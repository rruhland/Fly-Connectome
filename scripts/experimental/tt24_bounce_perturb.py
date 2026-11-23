"""Test closed-loop wrist bouncing after an unannounced horizontal flight kick."""

import argparse
import gzip
from hashlib import sha256
from itertools import product
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil
from .tt24_metric_transfer import load_transferred


ROOT = Path('artifacts/tt24')
STOCK = ROOT / 'bounce-ident-pupil.npz'
CORRECTION = ROOT / 'metric-transfer-map.npz'
OUTPUT = ROOT / 'bounce-perturb.json.gz'
GROUPS = {'stock': (1307819063, 24, 'standard'), 'alternate': (821172399, 16, 'changed')}
KICKS = tuple(product((0, 1), (-.24, -.16, .16, .24)))
QUALIFYING_RUN = 3
WINDOW = 120
MOTION_TOLERANCE = 1.e-4


def _design():
    groups = {}
    for name, (stream, count, regime) in GROUPS.items():
        rng = np.random.default_rng(stream)
        seeds = rng.integers(1, 2**31 - 1, count).tolist()
        allocated = rng.permutation(np.tile(KICKS, (count // len(KICKS), 1)))
        launches = []
        for seed, (axis, delta) in zip(seeds, allocated, strict=True):
            shift = [0., 0.]
            shift[int(axis)] = float(delta)
            launches.append(dict(seed=seed, velocity_delta_m_s=shift))
        groups[name] = dict(regime=regime, launches=launches, generator=stream)
    return groups


def _check_fresh(groups):
    seeds = [launch['seed'] for group in groups.values() for launch in group['launches']]
    if len(set(seeds)) != len(seeds):
        raise ValueError('perturbation course repeats a launch')
    pattern = re.compile(rb'"seed"\s*:\s*(?:' + b'|'.join(str(seed).encode() for seed in seeds) + rb')\b')
    for path in ROOT.iterdir():
        if path.name.startswith('bounce-perturb'):
            continue
        if path.name.endswith('.json.gz'):
            data = gzip.decompress(path.read_bytes())
        elif path.suffix == '.json':
            data = path.read_bytes()
        else:
            continue
        if pattern.search(data):
            raise ValueError(f'a supposedly fresh launch occurs in {path}')


def _model(arm):
    return WristPupil.load(STOCK) if arm == 'stock' else load_transferred(STOCK, CORRECTION)


def _state(arena):
    data = arena._data
    physical = np.r_[data.time, data.qpos, data.qvel, data.ctrl]
    return sha256(physical.astype('<f8').tobytes()).hexdigest()


def _candidate(arena, state, tick, delta):
    events = arena.judge_metrics()['rebound_events']
    indices = [row['impact_index'] for row in events[-QUALIFYING_RUN:]]
    if (len(indices) == QUALIFYING_RUN and
            indices[-1] - indices[0] == QUALIFYING_RUN - 1 and not arena._touching):
        return dict(tick=tick, time_s=float(arena._data.time), prior_rebounds=len(events),
                    last_qualified_impact=indices[-1], precut_state=state, velocity_delta_m_s=delta)
    return None


def _impulse(arena, state, kick):
    if (state != kick['precut_state'] or arena._touching or len(arena._rebounds) != kick['prior_rebounds'] or
            arena._data.time != kick['time_s']):
        raise ValueError('branches did not reach the same airborne kick state')
    before = arena._ball_velocity.copy()
    arena._ball_velocity[:2] += kick['velocity_delta_m_s']
    return dict(position_m=arena._ball_position.tolist(), before_m_s=before.tolist(),
                after_m_s=arena._ball_velocity.tolist(), time_s=kick['time_s'])


def _run(arena, regime, launch, *, model=None, tape=None, kick=None, cap=None, hold=False):
    obs = arena.reset(seed=launch['seed'], split='held', regime=regime, render=model is not None)
    limit = arena.max_steps if cap is None else cap
    if tape is not None and limit > len(tape):
        raise ValueError('the frozen control can only use explicitly recorded commands')
    states = sha256()
    camera = sha256() if model is not None else None
    actions = []
    candidate = packet = next_contact = None
    done = False
    for tick in range(limit):
        before = _state(arena)
        states.update(before.encode())
        if model is not None and kick is None and candidate is None:
            candidate = _candidate(arena, before, tick, launch['velocity_delta_m_s'])
        if kick is not None and tick == kick['tick']:
            packet = _impulse(arena, before, kick)
        if not hold and model is not None:
            camera.update(obs['rgb'])
            action = model.command({'rgb': obs['rgb'], 'joints': obs['joints']})
        elif hold and packet is not None:
            action = np.zeros(arena.action_size)
        else:
            action = np.asarray(tape[tick])
        actions.append(action.tolist())
        contacts = arena._paddle_contacts
        obs, _, done = arena.step(action, render=model is not None and tick + 1 < limit)
        if packet is not None and next_contact is None and arena._paddle_contacts > contacts:
            next_contact = tick + 1
        if done:
            break
    if kick is not None and packet is None:
        raise ValueError('a perturbed branch never reached its kick')
    states.update(_state(arena).encode())
    metrics = arena.judge_metrics()
    data = dict(metrics=metrics, actions=actions, physics_sha256=states.hexdigest(), kick=packet,
                next_contact_frame=next_contact)
    if camera is not None:
        data['wrist_rgb_sha256'] = camera.hexdigest()
        data['kick_candidate'] = candidate
    return data


def _fingerprint(row):
    keys = ('metrics', 'actions', 'physics_sha256', 'kick', 'next_contact_frame')
    return digest_json({key: row[key] for key in keys})


def _verify(arena, regime, launch, row, kick=None):
    tape = row['actions']
    evidence = _run(arena, regime, launch, tape=tape, kick=kick, cap=len(tape))
    if _fingerprint(row) != _fingerprint(evidence):
        raise ValueError(f'physical replay changed the trajectory on seed {launch["seed"]}')


def _after(row, kick):
    metrics = row['metrics']
    events = metrics['rebound_events'][kick['prior_rebounds']:]
    previous = kick['last_qualified_impact']
    streak = 0
    for event in events:
        if event['impact_index'] != previous + 1:
            break
        streak += 1
        previous += 1
    fault = metrics['termination'] not in (None, 'timeout')
    elapsed = metrics['elapsed_s'] - kick['time_s']
    return dict(immediate_consecutive=streak, total_qualified=len(events),
                observed_survival_s=elapsed, actual_loss_s=elapsed if fault else None,
                right_censored=not fault, loss_reason=metrics['termination'] if fault else None)


def _case(arena, arm, regime, launch, model_factory=None):
    if model_factory is None:
        model_factory = lambda: _model(arm)
    nominal = _run(arena, regime, launch, model=model_factory())
    _verify(arena, regime, launch, nominal)
    kick = nominal['kick_candidate']
    row = dict(launch, nominal=nominal, eligible=kick is not None, verified_tapes=1)
    if kick is None:
        return row
    tape = nominal['actions']
    cap = min(len(tape), kick['tick'] + WINDOW)
    intact = _run(arena, regime, launch, tape=tape, cap=cap)
    branches = {'nominal': intact}
    for name in ('live', 'tape', 'hold'):
        model = model_factory() if name == 'live' else None
        trial = _run(arena, regime, launch, model=model, tape=tape, kick=kick, cap=cap, hold=name == 'hold')
        _verify(arena, regime, launch, trial, kick)
        branches[name] = trial
    live, fixed, hold = (branches[key] for key in ('live', 'tape', 'hold'))
    if (live['kick'] != fixed['kick'] or live['kick'] != hold['kick'] or
            live['actions'][:kick['tick']+1] != tape[:kick['tick']+1] or
            fixed['actions'][:kick['tick']+1] != tape[:kick['tick']+1]):
        raise ValueError('the live controller saw unequal pre-kick inputs or an instantaneous velocity signal')
    prefix_end = min(kick['tick'] + 20, len(live['actions']), len(fixed['actions']),
                     *(branch['next_contact_frame'] for branch in (live, fixed)
                       if branch['next_contact_frame'] is not None))
    difference = abs(np.asarray(live['actions'][kick['tick']:prefix_end]) -
                     np.asarray(fixed['actions'][kick['tick']:prefix_end]))
    frames = np.flatnonzero(np.any(difference > MOTION_TOLERANCE, axis=1))
    motor = dict(early_ticks=len(difference), changed_ticks=len(frames),
                 first_change_ticks_after_kick=None if not len(frames) else int(frames[0]),
                 max_rate_delta=float(difference.max()) if len(difference) else 0.)
    row |= dict(kick=kick, cap_frame=cap, capped_at_nominal_tape_end=cap == len(tape),
                after={name: _after(result, kick) for name, result in branches.items()},
                branches=branches, motor=motor, verified_tapes=4)
    return row


def _summary(rows):
    paired = [row for row in rows if row['eligible']]
    result = dict(fresh=len(rows), eligible=len(paired), checked_tapes=sum(row['verified_tapes'] for row in rows),
                  capped_at_nominal_tape_end=sum(row['capped_at_nominal_tape_end'] for row in paired))
    if not paired:
        return result
    fields = ('immediate_consecutive', 'total_qualified', 'observed_survival_s')
    conditions = ('nominal', 'live', 'tape', 'hold')
    for name in conditions:
        scores = [row['after'][name] for row in paired]
        result[name] = {key: float(np.mean([score[key] for score in scores])) for key in fields}
        result[name]['actual_losses'] = sum(not branch['right_censored'] for branch in scores)
        result[name]['consecutive_scores'] = [score['immediate_consecutive'] for score in scores]
    result['live_vs_tape'] = {}
    for field in fields:
        delta = np.array([row['after']['live'][field] - row['after']['tape'][field] for row in paired])
        result['live_vs_tape'][field] = dict(mean_delta=float(delta.mean()), better=int((delta > 1.e-10).sum()),
                                            tied=int((abs(delta) <= 1.e-10).sum()), worse=int((delta < -1.e-10).sum()))
    first = [row['motor']['first_change_ticks_after_kick'] for row in paired
             if row['motor']['first_change_ticks_after_kick'] is not None]
    result['early_rate_changes'] = dict(changed_pairs=len(first), median_first_tick=float(np.median(first)) if first else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    groups = _design()
    meta = dict(schema='tt24-bounce-kick-1', groups=groups, qualified_rebounds_before_kick=QUALIFYING_RUN,
                maximum_post_kick_frames=WINDOW, source_checkpoint_sha256=_hash(STOCK),
                alternate_public_correction_sha256=_hash(CORRECTION), dimensions=[192, 144],
                stock_training='four motor-labeled stock demonstrations and privileged perception labels',
                alternate_calibration='public paired camera RGB and encoders, 248 frames on four changed-body drops',
                runtime='wrist RGB and own joint angles only', training_runs=0,
                early_rate_threshold=MOTION_TOLERANCE, early_rate_window_frames=20)
    if args.output.exists():
        result = _read(args.output)
        if digest_json(meta) != digest_json(result['metadata']):
            raise ValueError('the frozen design has changed')
    else:
        _check_fresh(groups)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = dict(metadata=meta, cases={arm: [] for arm in groups})
        _write(args.output, result)
    began = perf_counter()
    for arm, group in groups.items():
        rows = result['cases'][arm]
        arena = BounceArena(view='wrist', width=192, height=144, arm=arm)
        try:
            for launch in group['launches'][len(rows):]:
                row = _case(arena, arm, group['regime'], launch)
                rows.append(row)
                _write(args.output, result)
                print(arm, len(rows), launch['seed'], row['eligible'],
                      {name: score['immediate_consecutive'] for name, score in row.get('after', {}).items()}, flush=True)
        finally:
            arena.close()
    result['summary'] = {arm: _summary(rows) for arm, rows in result['cases'].items()}
    result['wall_seconds_last_invocation'] = perf_counter() - began
    _write(args.output, result)
    print(result['summary'], flush=True)


if __name__ == '__main__':
    main()
