"""Practice incoming arcs with one camera and replay transfer to the other."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt19_factory_serve_trial import load_checkpoint
from .tt19_flight import optical_start
from .tt20_arc_carry import ArcPrior, CarryArc, SOURCE_FRAMES, TARGET_FRAMES
from .tt20_fixture import TT20Arena
from .tt20_mono_flight_trial import CATALOG, CHECKPOINT, DATA, ORIGIN, OUTPUT, SETTINGS, VIEWS, distances
from .tt20_natural import NaturalViewTransport


MODES = ('optics', 'cross', 'same', 'shifted')
SEED = 1072338028
HORIZONS = (3, 6)
OTHER = dict(wrist='outside', outside='wrist')
STREAM = dict(wrist='wrist', outside='left')


def camera(arena, view):
    return NaturalViewTransport(view, lens_fov=arena.lens_intrinsics(), flight=view == 'outside',
                                continuous_wrist=True)


def practice(view, seeds):
    arena = TT20Arena(**SETTINGS, view_mode=view)
    rows = []
    try:
        sensor = camera(arena, view)
        for seed in seeds:
            tracker = sensor.tracker()
            arena.reset(seed=seed, server=-1, render=False)
            history = []
            for tick in range(SOURCE_FRAMES.stop):
                tracker.update(arena.render_cameras(), tick=tick)
                if tick in SOURCE_FRAMES:
                    xyz = tracker.ball
                    visible = xyz is not None and xyz.observed
                    history.append((tick, list(xyz.position) if visible else None,
                                    list(tracker.ball_sigma) if visible else None))
                arena.step(np.zeros(6), render=False)
            rows.append(dict(seed=seed, frames=SOURCE_FRAMES.stop, history=history,
                             rgb=tracker.digest.hexdigest()))
    finally:
        arena.close()
    normal = ArcPrior.fit([item['history'] for item in rows])
    shifted = ArcPrior.fit([item['history'] for item in rows], timing_shuffled=True)
    if normal is None or shifted is None:
        raise ValueError(f'not enough usable {view} RGB arcs from {len(seeds)} practice starts')
    print('practice', view, normal.arcs, flush=True)
    return dict(rows=rows, normal=normal.state(), shifted=shifted.state())


def _forecast(state, law, bounds):
    if state is None:
        return None
    path, _ = law.trajectory(*state, max(HORIZONS), bounds=bounds)
    return path[list(HORIZONS)].tolist()


def play(arena, player, tracker, seed, view, prior):
    arena.reset(seed=seed, server=-1, render=False)
    player.reset()
    carry = CarryArc(prior) if prior is not None else None
    actions, samples, optical_history = [], [], []
    for tick in range(arena.max_steps):
        images = arena.render_cameras()
        if tuple(images) != (STREAM[view],):
            raise ValueError('a single-camera run received an unexpected view')
        tracker.update(images, tick=tick)
        raw = tracker.ball
        xyz = raw.position if raw is not None and raw.observed else None
        speed = tracker.ball_velocity
        optical_history.append((tick, xyz))
        optical_state = (xyz, speed) if xyz is not None and speed is not None else optical_start(
            optical_history, player.law.gravity)
        state = None
        if carry is not None:
            stream = STREAM[view]
            state = carry.observe(tick, tracker.fits.get(stream), tracker.detections['ball'].get(stream),
                                  tracker.raw_ball, tracker.raw_ball_sigma)
        if state is not None:
            xyz, speed = state
        action = player.command(tick, xyz, -1, estimated_velocity=speed)
        actions.append(action.tolist())
        observed = dict(tick=tick, actual=(arena.ball.copy() - ORIGIN).tolist(), used_carry=state is not None)
        if tick in TARGET_FRAMES:
            observed['forecast'] = _forecast(optical_state if state is None else state, player.law, tracker.transport.bounds)
            observed['optical_forecast'] = _forecast(optical_state, player.law, tracker.transport.bounds)
        samples.append(observed)
        _, _, done, timeout, _ = arena.step(action, render=False)
        if done or timeout:
            break
    events = arena.events.copy()
    result = arena.point_result()
    return dict(seed=seed, actions=actions, plans=player.plans, samples=samples, result=result, events=events,
                rgb=tracker.digest.hexdigest())


def test(view, schedule, banks):
    arena = TT20Arena(**SETTINGS, view_mode=view)
    player = load_checkpoint(CHECKPOINT, CATALOG, 'factory').player
    states = dict(optics=None, cross=ArcPrior.restore(banks[OTHER[view]]['normal']),
                  same=ArcPrior.restore(banks[view]['normal']),
                  shifted=ArcPrior.restore(banks[OTHER[view]]['shifted']))
    rows = []
    try:
        transports = {mode: camera(arena, view) for mode in MODES}
        for index, seed in schedule:
            order = MODES[index % len(MODES):] + MODES[:index % len(MODES)]
            # Recreate the view at each start so no mode inherits additional
            # camera registration from practice or earlier held points.
            rows.append(dict(index=index, seed=seed, **{
                mode: play(arena, player, transports[mode].tracker(), seed, view, states[mode])
                for mode in order}))
            for transport in transports.values():
                transport.registry.known.clear()
            print(view, index + 1, seed, [rows[-1][mode]['result']['legal_returns'] for mode in MODES], flush=True)
    finally:
        player.close()
        arena.close()
    return rows


def _first_contact(row):
    return min((event['step'] - 1 for event in row['events']
                if event['kind'] == 'contact' and event.get('side') == 1), default=len(row['samples']))


def metrics(rows):
    results = {}
    for mode in MODES:
        errors = {horizon: [] for horizon in HORIZONS}
        paired = {horizon: [[], []] for horizon in HORIZONS}
        branch = [row[mode] for row in rows]
        for row in branch:
            contact = _first_contact(row)
            for sample in row['samples']:
                prediction = sample.get('forecast')
                if prediction is None:
                    continue
                for prediction_index, horizon in enumerate(HORIZONS):
                    tick = sample['tick'] + horizon
                    if tick >= min(contact, len(row['samples'])):
                        continue
                    actual = row['samples'][tick]['actual']
                    error = float(np.linalg.norm(np.subtract(prediction[prediction_index], actual)))
                    errors[horizon].append(error)
                    optical = sample['optical_forecast']
                    if optical is not None and sample['used_carry']:
                        paired[horizon][0].append(float(np.linalg.norm(np.subtract(optical[prediction_index], actual))))
                        paired[horizon][1].append(error)
        results[mode] = dict(starts=len(branch), first_returns=sum(row['result']['legal_returns'] > 0 for row in branch),
                             points_won=sum(row['result']['points_for'] for row in branch),
                             planned_with_prior=sum(row['samples'][row['plans'][0]['tick']]['used_carry']
                                                    for row in branch if row['plans']),
                             forecast={str(horizon): distances(errors[horizon], 'm') for horizon in HORIZONS},
                             same_rgb={str(horizon): dict(optics=distances(paired[horizon][0], 'm'),
                                                          carried=distances(paired[horizon][1], 'm'))
                                       for horizon in HORIZONS})
    return results


def replay(report):
    replayed = 0
    for view, rows in report['held'].items():
        arena = TT20Arena(**report['settings'], view_mode=view)
        try:
            for item in rows:
                for mode in MODES:
                    row = item[mode]
                    arena.reset(seed=item['seed'], server=-1, render=False)
                    for action in row['actions']:
                        arena.step(action, render=False)
                    if arena.point_result() != row['result'] or json.dumps(arena.events) != json.dumps(row['events']):
                        raise ValueError(f'physical replay changed in {view} {mode} for seed {item["seed"]}')
                    replayed += 1
        finally:
            arena.close()
    return replayed


def run(folder, exercises, starts, workers):
    seeds = np.random.default_rng(SEED).integers(2**31, size=exercises + starts).tolist()
    if len(set(seeds)) != len(seeds) or set(seeds) & previous_seeds(DATA):
        raise ValueError('the schedule repeats other table-tennis trials')
    practice_seeds, trial_seeds = seeds[:exercises], seeds[exercises:]
    paths = ('tt20_arc_carry.py', 'tt20_arc_carry_trial.py', 'tt20_fixture.py', 'tt20_natural_geometry.py',
             'tt20_natural.py', 'tt20_mono_flight.py', 'tt19_model_rollout.py', 'tt19_catalog.py')
    code = {name: _hash(Path(__file__).with_name(name)) for name in paths}
    schedule = list(enumerate(trial_seeds))
    cache = folder / 'arc-carry-practice.json.gz'
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        if cache.exists():
            saved = _read(cache)
            if saved['practice_seeds'] != practice_seeds or saved['source_code'] != code:
                raise ValueError('arc-carry practice receipt has a different schedule or implementation')
            banks = saved['banks']
        else:
            futures = {view: pool.submit(practice, view, practice_seeds) for view in VIEWS}
            banks = {view: task.result() for view, task in futures.items()}
            _write(cache, dict(schema='tt20-arc-carry-practice-v1', practice_seeds=practice_seeds,
                              source_code=code, banks=banks))
        jobs = {view: [pool.submit(test, view, schedule[index::workers], banks)
                       for index in range(min(workers, len(schedule)))] for view in VIEWS}
        held = {view: sorted([row for future in futures for row in future.result()], key=lambda row: row['index'])
                for view, futures in jobs.items()}
    if code != {name: _hash(Path(__file__).with_name(name)) for name in paths}:
        raise ValueError('the implementation changed during arc transfer')
    report = dict(schema='tt20-arc-carry-v1', settings=SETTINGS, schedule_seed=SEED, practice_seeds=practice_seeds,
                  held_seeds=trial_seeds, source_code=code, checkpoint_sha256=_hash(CHECKPOINT),
                  catalogue_sha256=_hash(CATALOG), practice_sha256=_hash(cache),
                  banks={view: {key: row[key] for key in ('normal', 'shifted')} for view, row in banks.items()},
                  held=held, metrics={view: metrics(rows) for view, rows in held.items()})
    report['replayed_points'] = replay(report)
    _write(folder / 'arc-carry.json.gz', report)
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--practice', type=int, default=12)
    parser.add_argument('--starts', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify', type=Path)
    arguments = parser.parse_args()
    if arguments.verify is not None:
        print('replayed', replay(_read(arguments.verify)))
    elif arguments.practice < 4 or min(arguments.starts, arguments.workers) < 1:
        parser.error('choose at least four practice starts and positive held starts and workers')
    else:
        arguments.folder.mkdir(parents=True, exist_ok=True)
        run(arguments.folder, arguments.practice, arguments.starts, arguments.workers)
