"""Charge one optical motion stream and score the frozen TT20 command correction."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_camera import PerspectiveCamera
from .tt4_transfer_trial import _hash, _read, _write
from .tt18_view import ImageColors
from .tt19_direct_receive import make_players, play, previous_seeds
from .tt19_direct_receive_payload import PayloadArena
from .tt19_so101 import mujoco
from .tt19_view import detect_roles
from .tt19_view_trial import CAMERAS, SO101ViewArena
from .tt20_blade import CorrectedPlayer, TimingFit, fit_timing


PROBE_SEED = 43750163
FRESH_SEED = 1664305912
FRESH_COUNT = 6
TRAIN_FRAMES = 48
COLOURS = ('tip', 0, 1, 2, 3)
VIEWS = ('left', 'right')
MARKS = ('tt19_tip_mark', *(f'tt19_rim_{i}' for i in range(4)))


def program(cache):
    with np.load(cache) as archive:
        moves = archive['tapes']
    possible = np.flatnonzero((abs(moves.sum(axis=1)).max(axis=1) < 3.85) & (abs(moves).sum(axis=(1, 2)) > 7.))
    selected = np.random.default_rng(PROBE_SEED).choice(possible, 3, replace=False)
    sequences = [np.vstack((moves[i], np.repeat(-moves[i].sum(axis=0, keepdims=True) / 4., 4, axis=0)))
                 for i in selected]
    return np.vstack(sequences).astype(float), selected.tolist()


def pixels(arena):
    objects = detect_roles({name: ImageColors(image) for name, image in arena.render_cameras().items()})
    return np.asarray([objects[role][view].center if view in objects[role] else [np.nan, np.nan]
                       for view in VIEWS for role in COLOURS])


def pixel_projector(arena):
    cameras = []
    for view in VIEWS:
        _, eye, target, fov = CAMERAS[view]
        cameras.append(PerspectiveCamera(eye, target, arena.width, arena.height, fov))
    marks = [arena.model.geom(name).id for name in MARKS]

    def project():
        mujoco.mj_kinematics(arena.model, arena.data)
        positions = arena.data.geom_xpos[marks] - [0., 0., TABLE_HEIGHT]
        return np.asarray([camera.project(point)[:2] for camera in cameras for point in positions])

    return project


def capture(arena, actions, observe):
    """Drive one continuous bench movement and abstract what the observer can read."""
    arena.reset(seed=PROBE_SEED, render=False)
    ball = arena.model.geom('ball')
    ball.contype[0] = ball.conaffinity[0] = 0
    arena.ball[:] = (0., 0., -25.)
    arena.velocity[:] = 0.
    mujoco.mj_forward(arena.model, arena.data)
    controls = arena._arm_ids[1][2][:5]
    increments = arena._joint_increments[controls]
    low, high = arena._joint_low[controls], arena._joint_high[controls]
    strides = int(round(arena.control_dt / arena.model.opt.timestep))
    trace = [observe()]
    step = np.arange(1, strides + 1)[:, None]
    for action in actions:
        settings = np.clip(arena.data.ctrl[controls] + step * (action * increments), low, high)
        for values in settings:
            arena.data.ctrl[controls] = values
            mujoco.mj_step(arena.model, arena.data)
        trace.append(observe())
    return np.stack(trace)


def calibrate(arena_args, cache):
    tape, selected = program(cache)
    station = dict(arena_args, view_mode='fixed')
    target = PayloadArena(**station)
    nominal = SO101ViewArena(**station)
    try:
        observed = capture(target, tape, lambda: pixels(target))
        model_rgb = capture(nominal, tape, lambda: pixels(nominal))
        inspect_model = pixel_projector(nominal)

        def project(commands):
            return capture(nominal, commands, inspect_model)

        original = project(tape)
        fit, metrics = fit_timing(model_rgb, original, observed, tape, project, fit_frames=TRAIN_FRAMES)
    finally:
        target.close()
        nominal.close()
    def public(trace):
        return np.where(np.isfinite(trace), trace, None).tolist()

    evidence = dict(charged_target_frames=len(tape), fitting_frames=TRAIN_FRAMES, checking_frames=len(tape)-TRAIN_FRAMES,
                    selected_nominal_motions=selected, commands=tape.tolist(), nominal_pixels=public(model_rgb),
                    observed_pixels=public(observed), nominal_projections=original.tolist(), rng_seed=PROBE_SEED,
                    fit=fit.state(), metrics=metrics)
    print('calibrated', fit.state(), metrics, flush=True)
    return fit, evidence


def batch(cache, source, args, fit, schedule, fresh):
    arena = PayloadArena(**args)
    base, unused = make_players(cache, source)
    unused.close()
    corrected = CorrectedPlayer(base, fit)
    players = {'frozen': base, 'corrected': corrected}
    rows = []
    try:
        for index, seed in schedule:
            row = dict(seed=seed, index=index)
            if fresh:
                first, second = ('frozen', 'corrected') if index % 2 == 0 else ('corrected', 'frozen')
                frames = []
                row[first] = play(arena, players[first], seed, frames)
                row[second] = play(arena, players[second], seed, frames, row[first]['actions'])
                shared = row[second]['shared_rgb_frames']
                if row[first]['optical'][:shared] != row[second]['optical'][:shared]:
                    raise ValueError('the paired camera prefixes differ')
            else:
                row['corrected'] = play(arena, corrected, seed, [])
            rows.append(row)
            print('fresh' if fresh else 'matched', seed, {k: r['legal_return_frames'] for k, r in row.items()
                                                          if k in players}, flush=True)
    finally:
        base.close()
        arena.close()
    return rows


def replay(arena_args, groups):
    arena = PayloadArena(**arena_args)
    count = 0
    try:
        for group in groups:
            for row in group:
                for branch in ('frozen', 'corrected'):
                    if branch not in row:
                        continue
                    item = row[branch]
                    arena.reset(seed=row['seed'], server=-1, render=False)
                    for action in item['actions']:
                        arena.step(action, render=False)
                    if (item['result'] != arena.point_result() or
                            json.dumps(item['events']) != json.dumps(arena.events)):
                        raise ValueError(f"physical replay differs on {row['seed']} in {branch}")
                    count += 1
    finally:
        arena.close()
    return count


def metrics(pairs):
    result = {branch: dict(first_returns=sum(bool(p[branch]['legal_return_frames']) for p in pairs),
                           total_returns=sum(p[branch]['result']['legal_returns'] for p in pairs),
                           points_won=sum(p[branch]['result']['points_for'] for p in pairs),
                           frames=sum(p[branch]['frames'] for p in pairs))
              for branch in ('frozen', 'corrected')}
    channels = [tuple(bool(p[b]['legal_return_frames']) for b in ('frozen', 'corrected')) for p in pairs]
    result['new_return_seeds'] = [p['seed'] for p, pair in zip(pairs, channels, strict=True) if pair == (False, True)]
    result['lost_return_seeds'] = [p['seed'] for p, pair in zip(pairs, channels, strict=True) if pair == (True, False)]
    return result


def run(previous_folder, folder, cache, workers):
    source = previous_folder / 'flight-source.json.gz'
    before = previous_folder / 'direct-receive-payload.json.gz'
    frozen = _read(before)
    if _hash(cache) != frozen['catalogue_sha256'] or _hash(source) != frozen['source_priors_sha256']:
        raise ValueError('the receiving inputs differ from the original frozen mass comparison')
    path = folder / 'blade-correction.json.gz'
    seeds = frozen['point_seeds']
    fresh = np.random.default_rng(FRESH_SEED).integers(2**31, size=FRESH_COUNT).tolist()
    seen = previous_seeds(previous_folder) | set(seeds) | {PROBE_SEED}
    if len(set(fresh)) != FRESH_COUNT or seen.intersection(fresh):
        raise ValueError('a fresh point overlaps an earlier evaluation')
    fit, optics = calibrate(frozen['arena'], cache)
    report = dict(schema='tt20-blade-correction-v1', arena=frozen['arena'], camera_payload_kg=.08,
                  frozen_payload_receipt_sha256=_hash(before), nominal_catalogue_sha256=_hash(cache),
                  flight_priors_sha256=_hash(source), calibration=optics, fresh_seed=FRESH_SEED,
                  fresh_starts=fresh, published_matched_starts=seeds)
    _write(path, report)

    def evaluate(pool, starts, novel):
        order = list(enumerate(starts))
        pending = [pool.submit(batch, cache, source, frozen['arena'], fit, order[i::workers], novel)
                   for i in range(min(workers, len(order)))]
        return sorted([row for task in pending for row in task.result()], key=lambda row: row['index'])

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        matched = evaluate(pool, seeds, False)
        for row, earlier in zip(matched, frozen['pairs'], strict=True):
            if row['seed'] != earlier['seed']:
                raise ValueError('the published starting order changed')
            row['frozen'] = earlier['gantry']
        report.update(matched=matched, matched_metrics=metrics(matched))
        _write(path, report)
        fresh_rows = evaluate(pool, fresh, True)

    report.update(fresh=fresh_rows, fresh_metrics=metrics(fresh_rows),
                  physical_replays=replay(frozen['arena'], (matched, fresh_rows)))
    _write(path, report)
    print('matched', report['matched_metrics'], 'fresh', report['fresh_metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_folder', type=Path)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--verify', action='store_true')
    setup = parser.parse_args()
    if setup.verify:
        proof = _read(setup.folder / 'blade-correction.json.gz')
        print('replayed', replay(proof['arena'], (proof['matched'], proof['fresh'])))
    else:
        setup.folder.mkdir(parents=True, exist_ok=True)
        run(setup.previous_folder, setup.folder, setup.previous_folder / 'factory-dry-catalog.npz', setup.workers)
