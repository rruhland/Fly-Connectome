"""Fit a shared correction to convert a clock stroke's physical opening contacts."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_factory import HOME, RATE
from .tt19_flight import FlightLaw
from .tt19_shoot import COURT
from .tt19_view_full_game import digest_json
from .tt22_duration_transfer import PONG, STREAMS, priors
from .tt22_joint_practice import READS, draw
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer import MotorPrimitive, lateral_goal
from .tt22_motor_transfer_trial import MACRO, MotorArena

OUTPUT = MACRO.parent.parent / 'tt23'
PRACTICE = OUTPUT / 'opening-motor-practice.json.gz'
CHECKPOINT = OUTPUT / 'opening-motor-checkpoint.json.gz'
HELD = OUTPUT / 'opening-motor-held.json.gz'
TRAIN_SEED = 1979332694
HELD_SEED = 478871166
SPLIT_PER_WIDTH = 24
SAMPLED = 6
HANDOVER = 25
HORIZON = 59
MODES = ('duration', 'neutral', 'permuted')
MODELS = ('stock', 'conditional', 'pooled')
MOVES = np.vstack((np.zeros((1, 4)), np.eye(4), -np.eye(4),
                   2. * qmc.Sobol(4, scramble=True, seed=1080482969).random_base2(3) - 1.))
SCALES = np.array((.100, .100, .080, .060, .037, .095, 4.))
CROSS = np.triu_indices(4, 1)


def source():
    gains = priors()
    gains['permuted'] = np.r_[_read(PONG)['frozen']['permuted'][1:], np.zeros(4)]
    return {name: gains[name].tolist() for name in MODES}


def course(seed):
    rng = np.random.default_rng(seed)
    return [dict(seed=s, lane=lane, width=name)
            for name, span in STREAMS.items() for s, lane in draw(rng, SPLIT_PER_WIDTH, span)]


def described(primitive, sight, theta, law):
    """Use only issued motor geometry and the shared observed optical flight."""
    if sight['mover'] is None:
        return None, None
    mover, velocity = np.asarray(sight['mover'])
    ready = primitive.ready_world.copy()
    goal, _ = lateral_goal(ready, mover, velocity, np.r_[0., theta[:2]])
    goals = primitive._translate(goal, ready)
    issued, peak = (np.asarray(frame['solved']) for frame in goals)
    flight, speed = law.trajectory(mover, velocity, 24, bounds=COURT)
    depth = .5 * (issued[1] + peak[1])
    gap = flight[:, 1] - depth
    indices = np.flatnonzero((gap[:-1] >= 0.) & (gap[1:] < 0.))
    if len(indices):
        i = indices[0]
        fraction = gap[i] / (gap[i] - gap[i + 1])
        position = (1. - fraction) * flight[i] + fraction * flight[i + 1]
        incoming = (1. - fraction) * speed[i] + fraction * speed[i + 1]
        eta = i + fraction
    else:
        i = int(np.argmin(abs(gap)))
        position, incoming, eta = flight[i], speed[i], i
    shift = issued[0] - ready[0]
    features = np.array((shift, abs(shift), position[0] - peak[0], position[2] - peak[2],
                         incoming[2], incoming[1], READS - 1 + eta - 18.))
    return goal, np.clip(features / SCALES, -4., 4.)


def correction(stock, option):
    """Change the clock and sagittal peak pose while preserving lateral placement."""
    if option == 0:
        return stock.parameters.copy(), 0.
    advance, forward, loft, tempo = MOVES[option]
    parameters = stock.parameters.copy()
    parameters[10] -= 2.4 * advance
    parameters[11] = min(parameters[11] + tempo, HANDOVER - 5. - parameters[10])
    if forward == loft == 0.:
        return parameters, 0.
    ready = HOME + RATE[:5] * parameters[:5]
    peak = ready + RATE[:5] * parameters[5:10]
    tool, normal, _, _ = stock.factory.pose(peak)
    desired = tool + np.array((0., .042 * forward, 0.))
    angle = .24 * loft
    facing = np.array((normal[0], np.cos(angle) * normal[1] - np.sin(angle) * normal[2],
                       np.sin(angle) * normal[1] + np.cos(angle) * normal[2]))
    solved = stock.factory.solve(desired, facing, peak)
    actual, oriented, _, _ = stock.factory.pose(solved)
    lateral_error = float(actual[0] - tool[0])
    if abs(lateral_error) > .0045 or np.linalg.norm(actual - desired) > .026 or oriented @ facing < .96:
        return None, lateral_error
    parameters[5:10] = (solved - ready) / RATE[:5]
    return parameters, lateral_error


def opening_result(arena):
    contact_tick = next((event['step'] for event in arena.events if event['kind'] == 'contact'
                         and event['side'] == 1 and event['step'] <= HANDOVER), None)
    landing = next((event['location'][:2] for event in arena.events if event['kind'] == 'legal_return'
                    and event['side'] == 1 and contact_tick is not None), None)
    placement = 0. if landing is None else max(0., 1. - abs(landing[0]) / .36 - abs(landing[1] - .48) / .42)
    return dict(contact=contact_tick is not None, legal=landing is not None, landing=landing,
                reward=float(landing is not None) + .20 * placement, point=arena.point_result())


def simulate(arena, sight, parameters, *, tape=None, keep_tape=True):
    arena.lane = sight['lane']
    arena.reset(seed=sight['seed'], server=-1, render=False)
    primitive = MotorPrimitive(parameters, None)
    actions = []
    for tick in range(HORIZON if tape is None else len(tape)):
        if tape is None:
            command = np.zeros(6) if tick < READS else primitive.command(tick)
        else:
            command = tape[tick]
        if keep_tape:
            actions.append(np.asarray(command).tolist())
        _, _, done, cap, _ = arena.step(command, render=False)
        if done or cap or arena.legal_returns:
            if tape is not None and tick + 1 != len(tape):
                raise ValueError('opening replay terminated at a different frame')
            break
    outcome = dict(opening_result(arena), frames=arena.steps, event_sha256=digest_json(arena.events))
    if keep_tape:
        outcome.update(actions=actions, events=arena.events.copy())
    return outcome


def action_basis(options):
    a = MOVES[np.asarray(options)]
    return np.column_stack((a, a * a, a[:, CROSS[0]] * a[:, CROSS[1]]))


def design(options, context, *, conditional):
    action = action_basis(options)
    if not conditional:
        return action
    coupling = (action[:, :4, None] * context[:, None, :]).reshape(len(action), -1)
    return np.column_stack((action, coupling))


def fit(cases):
    options, deltas, context = [], [], []
    for case in cases:
        for sample in case['policies'].values():
            if sample['context'] is None:
                continue
            base = sample['trial']['0']['reward']
            for item, out in sample['trial'].items():
                if int(item):
                    options.append(int(item))
                    context.append(sample['context'])
                    deltas.append(out['reward'] - base)
    predictors = {}
    for name in MODELS[1:]:
        matrix = design(options, np.asarray(context), conditional=name == 'conditional')
        penalty = np.r_[np.full(14, 8.), np.full(matrix.shape[1] - 14, 18.)]
        weights = np.linalg.solve(matrix.T @ matrix + np.diag(penalty), matrix.T @ deltas)
        predictors[name] = weights.tolist()
    return predictors, len(deltas)


def select(stock, context, model, name):
    if name == 'stock' or context is None:
        return 0, stock.parameters.copy(), 0.
    indexed = np.arange(len(MOVES))
    model_inputs = design(indexed, np.repeat(context[None, :], len(MOVES), axis=0),
                          conditional=name == 'conditional')
    ranking = model_inputs @ model[name] - .012 * np.linalg.norm(MOVES, axis=1)
    for option in np.argsort(-ranking, kind='stable'):
        parameters, error = correction(stock, option)
        if parameters is not None:
            return int(option), parameters, error
    raise ValueError('the original uncorrected stroke must be available')


def train(arena):
    macro = _read(MACRO)['parameters']
    law, gains = FlightLaw(), source()
    metadata = dict(schema='tt23-opening-motor-practice-1', seed=TRAIN_SEED, schedule=course(TRAIN_SEED),
                    gains=gains, options=MOVES.tolist(), macro_sha256=_hash(MACRO), source_sha256=_hash(PONG))
    practice = _read(PRACTICE) if PRACTICE.exists() else dict(metadata, cases=[])
    if any(practice[key] != value for key, value in metadata.items()):
        raise ValueError('opening practice archive metadata changed')
    rng = np.random.default_rng(TRAIN_SEED + 1)
    began = perf_counter()
    for step, start in enumerate(metadata['schedule']):
        choices = np.r_[0, rng.choice(np.arange(1, len(MOVES)), size=SAMPLED, replace=False)]
        if step < len(practice['cases']):
            if any(practice['cases'][step][key] != value for key, value in start.items()):
                raise ValueError('earlier practice row changed its launch')
            continue
        sight = prefix(arena, start['seed'], start['lane'])
        case = dict(start, sight=sight, sampled=choices.tolist(), policies={})
        for name, theta in gains.items():
            stock = MotorPrimitive(macro, None)
            goal, context = described(stock, sight, theta, law)
            outcomes = {}
            for option in choices if context is not None else [0]:
                params, error = correction(stock, option)
                if params is None:
                    continue
                result = simulate(arena, sight, params, keep_tape=False)
                outcomes[str(option)] = dict(result, changed_peak_x=error)
            case['policies'][name] = dict(goal=goal, context=None if context is None else context.tolist(), trial=outcomes)
        practice['cases'].append(case)
        _write(PRACTICE, practice)
        if (step + 1) % 4 == 0:
            print('practice', step + 1, round(perf_counter() - began, 1), flush=True)
    model, used = fit(practice['cases'])
    total = sum(len(sample['trial']) for case in practice['cases'] for sample in case['policies'].values())
    frames = sum(row['frames'] for case in practice['cases'] for sample in case['policies'].values()
                 for row in sample['trial'].values())
    result = dict(schema='tt23-opening-motor-checkpoint-1', practice=PRACTICE.name, practice_sha256=_hash(PRACTICE),
                  macro_sha256=_hash(MACRO), source_sha256=_hash(PONG), gains=gains, model=model,
                  observed_deltas=used, physical_attempts=total, physical_frames=frames,
                  context_scale=SCALES.tolist(), options=MOVES.tolist(), prefix_frames=READS, handover=HANDOVER)
    _write(CHECKPOINT, result)
    print('frozen', used, total, frames, flush=True)
    return result


def aggregate(cases):
    result = {}
    for name in MODES:
        result[name] = {}
        for method in MODELS:
            rows = [case['policies'][name][method] for case in cases]
            stock = [case['policies'][name]['stock'] for case in cases]
            scores = [(after['legal'], before['legal']) for after, before in zip(rows, stock, strict=True)]
            result[name][method] = dict(starts=len(rows), contacts=sum(row['contact'] for row in rows),
                                       legal=sum(row['legal'] for row in rows),
                                       learned_only=sum(after and not before for after, before in scores),
                                       stock_only=sum(before and not after for after, before in scores),
                                       reward=float(sum(row['reward'] for row in rows)),
                                       faults={reason:sum(row['point']['reason'] == reason for row in rows)
                                               for reason in ('own_half', 'long_past_endline', 'floor', 'fixture_contact')})
    return result


def assess(arena, state):
    macro, law = _read(MACRO)['parameters'], FlightLaw()
    schedule = course(HELD_SEED)
    if set(row['seed'] for row in schedule) & set(row['seed'] for row in course(TRAIN_SEED)):
        raise ValueError('opening training and held launches overlap')
    metadata = dict(schema='tt23-opening-motor-held-1', seed=HELD_SEED, schedule=schedule,
                    weights_sha256=_hash(CHECKPOINT))
    report = _read(HELD) if HELD.exists() else dict(metadata, cases=[])
    if any(report[key] != value for key, value in metadata.items()):
        raise ValueError('opening held archive metadata changed')
    began = perf_counter()
    for step, start in enumerate(schedule):
        if step < len(report['cases']):
            if any(report['cases'][step][key] != value for key, value in start.items()):
                raise ValueError('earlier held row changed its launch')
            continue
        sight = prefix(arena, start['seed'], start['lane'])
        row = dict(start, sight=sight, policies={})
        for name, theta in state['gains'].items():
            stock = MotorPrimitive(macro, None)
            goal, context = described(stock, sight, theta, law)
            results, cached = {}, {}
            for method in MODELS:
                i, params, error = select(stock, context, state['model'], method)
                if i not in cached:
                    outcome = simulate(arena, sight, params)
                    replay = simulate(arena, sight, params, tape=outcome['actions'], keep_tape=False)
                    if replay != {key: val for key, val in outcome.items() if key not in ('actions', 'events')}:
                        raise ValueError('opening physical replay diverged')
                    cached[i] = outcome
                results[method] = dict(cached[i], option=i, changed_peak_x=error, goal=goal)
            row['policies'][name] = results
        report['cases'].append(row)
        _write(HELD, report)
        if (step + 1) % 4 == 0:
            print('held', step + 1, round(perf_counter() - began, 1), flush=True)
    summary = report['summary'] = {name: aggregate([r for r in report['cases'] if r['width'] == name])
                                  for name in STREAMS}
    combined = summary['combined'] = aggregate(report['cases'])
    gain = sum(group['conditional']['legal'] - group['stock']['legal'] for group in combined.values())
    warm_gain = combined['duration']['conditional']['legal'] - combined['duration']['stock']['legal']
    report['full_point_gate'] = gain >= 8 and warm_gain >= 3
    report['replayed_segments'] = sum(len({row['option'] for row in modes.values()})
                                      for case in report['cases'] for modes in case['policies'].values())
    _write(HELD, report)
    print('combined', combined, 'gate', report['full_point_gate'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held-only', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    if args.held_only and args.train_only:
        parser.error('select at most one phase')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arena = MotorArena()
    try:
        state = _read(CHECKPOINT) if args.held_only else train(arena)
        if not args.train_only:
            if state['practice_sha256'] != _hash(PRACTICE) or state['macro_sha256'] != _hash(MACRO):
                raise ValueError('practice or standard macro changed')
            assess(arena, state)
    finally:
        arena.close()


if __name__ == '__main__':
    main()
