"""Compare size ranging and temporal ranging on matched one-camera SO-101 starts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt19_factory_serve_trial import load_checkpoint
from .tt19_flight import optical_start


MODES = ('size', 'temporal')
VIEWS = ('wrist', 'outside')
DATA = ROOT / 'artifacts/tt19'
OUTPUT = ROOT / 'artifacts/tt20'
CHECKPOINT = DATA / 'factory-serve-player.json.gz'
CATALOG = DATA / 'factory-dry-catalog.npz'
SEED = 2079001421
ORIGIN = np.array((0., 0., TABLE_HEIGHT))
SETTINGS = dict(venue='hall', strict_endline=True, width=480, height=360, detail='preview',
                mount='clamp', wrist_optics='outboard', marked=False, max_steps=190)


def play(arena, player, tracker, seed):
    arena.reset(seed=seed, render=False, server=-1)
    player.reset()
    actions, samples = [], []
    for tick in range(arena.max_steps):
        images = arena.render_cameras()
        if len(images) != 1:
            raise ValueError('the receiver requires exactly one RGB stream')
        tracker.update(images)
        xyz = None if tracker.ball is None or not tracker.ball.observed else tracker.ball.position
        speed = tracker.ball_velocity
        action = player.command(tick, xyz, -1, estimated_velocity=speed)
        actions.append(action.tolist())

        # The evaluator measures errors only after the command was chosen.
        actual = arena.ball.copy() - ORIGIN
        actual_velocity = arena.velocity.copy() * arena.control_dt
        decision = bool(-.81 < actual[1] < -.24 and actual_velocity[1] < -.031 and -.01 < actual[2] < .85)
        samples.append(dict(tick=tick, position=None if xyz is None else list(map(float, xyz)),
                            velocity=None if speed is None else list(map(float, speed)),
                            raw_size=None if tracker.raw_ball is None else tracker.raw_ball.tolist(),
                            source=tracker.sources.get('ball_range'),
                            actual=actual.tolist(), actual_velocity=actual_velocity.tolist(), decision=decision))
        _, _, done, timeout, _ = arena.step(action, render=False)
        if done or timeout:
            break
    outcome = arena.point_result()
    events = arena.events.copy()
    first = next((event['step'] for event in events if event['kind'] == 'legal_return'
                  and event.get('side') == 1), None)
    digest = sha256(np.asarray(actions, dtype='<f8').tobytes()).hexdigest()
    return dict(seed=seed, actions=actions, samples=samples, action_sha256=digest,
                result=outcome, first_return=first, plans=player.plans, events=events,
                rgb_sha256=tracker.digest.hexdigest())


def batch(view, schedule):
    from .tt20_fixture import TT20Arena
    from .tt20_natural import NaturalViewTransport

    arena = TT20Arena(**SETTINGS, view_mode=view)
    player = load_checkpoint(CHECKPOINT, CATALOG, 'factory').player
    pairs = []
    try:
        lens_fov = {name: entry['fovy'] for name, entry in arena.lens_intrinsics().items()}
        for index, seed in schedule:
            order = MODES if index % 2 == 0 else MODES[::-1]
            rows = {}
            for mode in order:
                transport = NaturalViewTransport(view, lens_fov=lens_fov, flight=(mode == 'temporal'))
                rows[mode] = play(arena, player, transport.tracker(), seed)
            pairs.append(dict(index=index, seed=seed, **rows))
            print(view, index + 1, seed, [rows[mode]['result']['legal_returns'] for mode in MODES], flush=True)
    finally:
        player.close()
        arena.close()
    return pairs


def replay(report):
    from .tt20_fixture import TT20Arena

    count = 0
    for view, pairs in report['pairs'].items():
        arena = TT20Arena(**report['arena'], view_mode=view)
        try:
            for pair in pairs:
                for mode in MODES:
                    row = pair[mode]
                    arena.reset(seed=pair['seed'], render=False, server=-1)
                    for action in row['actions']:
                        arena.step(action, render=False)
                    if arena.point_result() != row['result'] or json.dumps(arena.events) != json.dumps(row['events']):
                        raise ValueError(f'physical replay changed in {view} {mode} for seed {pair["seed"]}')
                    count += 1
        finally:
            arena.close()
    return count


def distances(values, units):
    return dict(count=len(values), units=units, median=float(np.median(values)) if values else None,
                p90=float(np.quantile(values, .9)) if values else None)


def optical_metrics(rows):
    position_errors, velocity_errors = [], []
    decision_frames = 0
    for row in rows:
        history = []
        for sample in row['samples']:
            xyz = sample['position']
            history.append((sample['tick'], xyz))
            if not sample['decision']:
                continue
            decision_frames += 1
            if xyz is None:
                continue
            position_errors.append(float(np.linalg.norm(np.subtract(xyz, sample['actual']))))
            speed = sample['velocity']
            if speed is None:
                state = optical_start(history, .0109)
                speed = None if state is None else state[1]
            if speed is not None:
                velocity_errors.append(float(30 * np.linalg.norm(np.subtract(speed, sample['actual_velocity']))))
    result = dict(position=distances(position_errors, 'm'), decision_frames=decision_frames,
                  velocity=distances(velocity_errors, 'm/s'))
    return result


def paired(pairs, measure):
    counts = {mode: sum(bool(measure(pair[mode])) for pair in pairs) for mode in MODES}
    size_only = sum(bool(measure(p['size'])) and not measure(p['temporal']) for p in pairs)
    temporal_only = sum(bool(measure(p['temporal'])) and not measure(p['size']) for p in pairs)
    differing = size_only + temporal_only
    return dict(counts=counts, size_only=size_only, temporal_only=temporal_only, starts=len(pairs),
                exact_p=binomtest(temporal_only, differing).pvalue if differing else 1.)


def metrics(pairs):
    return dict(returns=paired(pairs, lambda row: row['first_return'] is not None),
                wins=paired(pairs, lambda row: row['result']['points_for']),
                contacts=paired(pairs, lambda row: row['result']['contacts']),
                optical={mode: optical_metrics([p[mode] for p in pairs]) for mode in MODES})


def run(folder, count, workers, views):
    seeds = np.random.default_rng(SEED).integers(2**31, size=count).tolist()
    if len(set(seeds)) != count or set(seeds) & previous_seeds(DATA):
        raise ValueError('the start schedule repeats a prior point')
    schedule = list(enumerate(seeds))
    files = ('tt20_fixture.py', 'tt20_natural.py', 'tt20_natural_geometry.py', 'tt20_mono_flight.py',
             'tt20_mono_flight_trial.py', 'tt19_model_rollout.py', 'tt19_catalog.py', 'tt19_view.py',
             'tt19_so101.py')
    before = {name: _hash(Path(__file__).with_name(name)) for name in files}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = {view: [pool.submit(batch, view, schedule[i::workers]) for i in range(workers)] for view in views}
        pairs = {view: sorted([row for task in tasks for row in task.result()], key=lambda row: row['index'])
                 for view, tasks in futures.items()}
    if before != {name: _hash(Path(__file__).with_name(name)) for name in files}:
        raise ValueError('the sensor or play implementation changed during evaluation')
    report = dict(schema='tt20-mono-flight-v1', arena=SETTINGS, schedule_seed=SEED, point_seeds=seeds,
                  checkpoint_sha256=_hash(CHECKPOINT), catalogue_sha256=_hash(CATALOG), implementation=before,
                  target_practice_points=0, pairs=pairs, metrics={view: metrics(rows) for view, rows in pairs.items()})
    report['replayed_points'] = replay(report)
    name = 'mono-flight' if tuple(views) == VIEWS else 'mono-flight-' + '-'.join(views)
    _write(folder / f'{name}.json.gz', report)
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--starts', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--views', nargs='+', choices=VIEWS, default=VIEWS)
    parser.add_argument('--verify', type=Path)
    arguments = parser.parse_args()
    if arguments.verify:
        print('replayed', replay(_read(arguments.verify)))
    elif not 1 <= arguments.workers <= arguments.starts:
        parser.error('choose at least as many starts as workers')
    else:
        arguments.folder.mkdir(parents=True, exist_ok=True)
        run(arguments.folder, arguments.starts, arguments.workers, arguments.views)
