"""Choose a coherent wrist reference from public sightings of the launch area."""

import argparse
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import DESIGNS, DonorBounce, FUSED_OUTPUT, Native, ROOT, load_public
from .tt24_launch_context import draw
from .tt24_posture import WidePosture
from .tt24_public_body import sight
from .tt24_ready import prepare_step


SOURCE = ROOT / 'tt24_donor-refined.json.gz'
FAMILIAR = ROOT / 'posture-source.npz'
FIT = ROOT / 'ready-geometry-source.npz'
OUTPUT = ROOT / 'ready-geometry.json.gz'
MODES = ('familiar', 'conditional', 'shuffled', 'as_started')
COURSES = (('fixed', BounceArena, 12), ('roving', WidePosture, 16))
SEED = 1514287927
SHUFFLE_SEED = 312570155
MIN_SOURCE_STREAK = 5
MIN_SIGHTINGS = 3


def ball_position(decoder, obs):
    ellipse = sight(obs['rgb'])
    if np.isfinite(ellipse).all():
        return decoder.ball(ellipse, decoder.body(obs['joints']))
    return None


def source_data(decoder):
    source = _read(SOURCE)['held_donor']
    rows = []
    for i, (design, group) in enumerate(zip(DESIGNS, source, strict=True)):
        arena = DonorBounce(design, view='wrist', width=192, height=144)
        try:
            for case in group:
                streak = case['conditions']['clean']['metrics']['longest_consecutive_rebounds']
                obs = arena.reset(seed=case['seed'], split='held', regime='changed')
                measured = ball_position(decoder, obs)
                rows.append(dict(body=i, seed=case['seed'], streak=streak, q=obs['joints'].tolist(),
                                 seen=None if measured is None else measured.tolist()))
        finally:
            arena.close()
    return rows


def fit_model(rows):
    successes = [r for r in rows if r['streak'] >= MIN_SOURCE_STREAK and r['seen'] is not None]
    if len(successes) < 10:
        raise ValueError(f'only {len(successes)} successful launches have a public sphere sighting')
    q = np.array([r['q'] for r in successes])
    positions = np.array([r['seen'][:2] for r in successes])
    bodies = np.array([r['body'] for r in successes])
    center = positions.mean(axis=0)
    scale = np.maximum(.025, positions.std(axis=0))
    # Body-specific offsets are removed from the regressions so that a wrist
    # calibration offset cannot become a cross-body location association.
    x, y = positions.copy(), q.copy()
    perm = np.arange(len(q))
    rng = np.random.default_rng(SHUFFLE_SEED)
    for body in np.unique(bodies):
        match = np.flatnonzero(bodies == body)
        x[match] -= x[match].mean(axis=0)
        y[match] -= y[match].mean(axis=0)
        perm[match] = rng.permutation(match)
    x /= scale
    gram = x.T @ x + 2.4*np.eye(2)
    fit = np.linalg.solve(gram, x.T @ y)
    shuffled = np.linalg.solve(gram, x.T @ y[perm])
    return dict(center=center, scale=scale, fit=fit, shuffled=shuffled,
                bound=2.25*q.std(axis=0), seeds=np.array([r['seed'] for r in successes]), permutation=perm)


def infer(measured, model, familiar, mode):
    feature = np.clip((measured[:2] - model['center'])/model['scale'], -2.5, 2.5)
    change = feature @ model['fit' if mode == 'conditional' else 'shuffled']
    return familiar + np.clip(change, -model['bound'], model['bound'])


def begin_release(arena, velocity, spin):
    arena._ball_velocity[:] = velocity
    arena._ball_spin[:] = spin
    mujoco.mj_forward(arena._model, arena._data)


def episode(arena, server, learner_seed, server_seed, native, weights, familiar, model, mode):
    server.reset(seed=server_seed, split='held', render=False)
    location = server._ball_position.copy()
    velocity, spin = server._ball_velocity.copy(), server._ball_spin.copy()
    lead = int(np.random.default_rng(learner_seed ^ 11940).integers(14, 39))

    def reset():
        arena.reset(seed=learner_seed, split='held', render=False)
        arena._ball_position[:] = location
        arena._ball_velocity[:] = 0.
        arena._ball_spin[:] = 0.
        mujoco.mj_forward(arena._model, arena._data)

    reset()
    obs = arena._observation(True)
    sensor = load_public(FUSED_OUTPUT)
    q = obs['joints'].copy()
    reference = q.copy() if mode == 'as_started' else familiar.copy()
    sensor.home = sensor.decoder.body(reference)[12:15].copy()
    issued = (q-reference).astype(float)
    reference_tick, launch_estimate = None, None
    measurements, actions = [], []
    for tick in range(lead+arena.max_steps):
        if mode in ('conditional', 'shuffled') and tick < min(lead//2, 11) and launch_estimate is None:
            observed = ball_position(sensor.decoder, obs)
            if observed is not None:
                if measurements and np.linalg.norm(measurements[-1][1]-observed) > .05:
                    measurements.clear()
                measurements.append((tick, observed))
                if len(measurements) == MIN_SIGHTINGS:
                    reference_tick = tick
                    launch_estimate = np.median(np.stack([v for _,v in measurements]), axis=0)
                    new_reference = infer(launch_estimate, model, familiar, mode)
                    # The issued offset is carried into the new reference, so
                    # switching coordinates preserves the current servo target.
                    issued += reference-new_reference
                    reference = new_reference
                    sensor.home = sensor.decoder.body(reference)[12:15].copy()
        action = draw(native, sensor.observe(obs), weights, issued)
        actions.append(action.tolist())
        if tick < lead:
            prepare_step(arena, action, location)
            if tick+1 == lead:
                begin_release(arena, velocity, spin)
            obs = arena._observation(True)
        else:
            obs, _, done = arena.step(action)
            if done:
                break
    metrics = arena.judge_metrics()
    reset()
    for tick, action in enumerate(actions):
        if tick < lead:
            prepare_step(arena, np.asarray(action), location)
            if tick+1 == lead:
                begin_release(arena, velocity, spin)
        else:
            _, _, done = arena.step(action, render=False)
            if done != (tick == len(actions)-1):
                raise ValueError('physical replay changed its terminal tick')
    if digest_json(metrics) != digest_json(arena.judge_metrics()):
        raise ValueError('physical replay changed the referee events')
    return dict(reference=reference.tolist(), initial_joints=q.tolist(), reference_tick=reference_tick,
                estimate=None if launch_estimate is None else launch_estimate.tolist(),
                sightings=[dict(tick=tick, xyz=xyz.tolist()) for tick, xyz in measurements],
                preparation_ticks=lead, launch_xyz=location.tolist(), launch_velocity=velocity.tolist(),
                launch_spin=spin.tolist(), metrics=metrics, actions=actions)


def summary(cases):
    streaks = {m: np.array([r['modes'][m]['metrics']['longest_consecutive_rebounds'] for r in cases]) for m in MODES}
    groups = {m: dict(mean=float(v.mean()), maximum=int(v.max()), eight=int((v >= 8).sum()),
                      ten=int((v >= 10).sum()), streaks=v.tolist()) for m,v in streaks.items()}
    pairs = {}
    for m in ('familiar', 'shuffled', 'as_started'):
        diff = streaks['conditional'] - streaks[m]
        pairs[m] = dict(mean=float(diff.mean()), wins=int((diff>0).sum()),
                        ties=int((diff==0).sum()), losses=int((diff<0).sum()))
    localized = [r['modes']['conditional'] for r in cases if r['modes']['conditional']['estimate'] is not None]
    error = np.array([np.array(r['estimate'])[:2]-np.array(r['launch_xyz'])[:2] for r in localized])
    result = dict(groups=groups, conditional_pairs=pairs, eligible=len(localized), verified_tapes=len(cases)*len(MODES))
    if len(localized):
        result['horizontal_error_median_m'] = float(np.median(np.linalg.norm(error, axis=1)))
        result['reference_tick_median'] = float(np.median([r['reference_tick'] for r in localized]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--fit', type=Path, default=FIT)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    start = perf_counter()
    observations = source_data(load_public(FUSED_OUTPUT).decoder)
    model = fit_model(observations)
    for path in (args.fit, args.output):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fit, **model)
    print('source', len(observations), len(model['seeds']), {k:model[k].tolist() for k in ('center','scale','fit','bound')},
          flush=True)
    if args.source_only:
        return
    with np.load(FAMILIAR, allow_pickle=False) as data:
        familiar = data['center']
    with np.load(CHECKPOINT, allow_pickle=False) as data:
        weights = np.ascontiguousarray(data['weights'], dtype=float)
    rng = np.random.default_rng(SEED)
    schedule = {name: rng.integers(1, 2**31-1, (count,2)).tolist() for name, _, count in COURSES}
    report = dict(schema='tt24-ready-geometry-1', source_evidence_sha256=_hash(SOURCE), observations=observations,
                  familiar_sha256=_hash(FAMILIAR), fixed_head_sha256=_hash(CHECKPOINT), pose_sha256=_hash(FUSED_OUTPUT),
                  fit_sha256=_hash(args.fit), minimum_source_streak=MIN_SOURCE_STREAK, seed=SEED, schedule=schedule,
                  cases={key:[] for key in schedule}, results={}, unchanged_learner_head=True)
    _write(args.output, report)
    native = Native()
    arena = WidePosture(view='wrist', width=192, height=144, max_steps=360)
    try:
        for name, server_type, _ in COURSES:
            server = server_type(view='none')
            try:
                for learner_seed, server_seed in schedule[name]:
                    modes = {mode:episode(arena, server, learner_seed, server_seed, native, weights, familiar, model, mode)
                             for mode in MODES}
                    matched = modes['conditional']['estimate'] == modes['shuffled']['estimate']
                    tick = modes['conditional']['reference_tick']
                    matched &= modes['conditional']['actions'][:tick] == modes['shuffled']['actions'][:tick] if tick is not None else modes['conditional']['actions'] == modes['shuffled']['actions']
                    if not matched:
                        raise ValueError('conditional and shuffled conditions saw different physical prefixes')
                    report['cases'][name].append(dict(learner_seed=learner_seed, server_seed=server_seed, modes=modes))
                    print(name, learner_seed, server_seed, tick,
                          [row['metrics']['longest_consecutive_rebounds'] for row in modes.values()], flush=True)
                    _write(args.output, report)
            finally:
                server.close()
            report['results'][name] = summary(report['cases'][name])
            _write(args.output, report)
    finally:
        arena.close()
        native.working.cleanup()
    if _hash(CHECKPOINT) != report['fixed_head_sha256']:
        raise ValueError('the receiver head changed during the assessment')
    report['wall_seconds'] = perf_counter()-start
    _write(args.output, report)
    print(report['results'], flush=True)


if __name__ == '__main__':
    main()
