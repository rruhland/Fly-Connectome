"""Test RGB-only camera registration throughout live compact-court rallies."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_rally_improve_optical import snapshot
from .tt27_compact import (CACHE, COMBINED, DIMS, EYES, MOVIE_SEED, OPTICAL, OUTPUT, CompactStation,
                           Options, audit, brief)
from .tt27_world_camera import CameraMounts, Registration, SOURCE_ROTATIONS, true_direction


BUDGET = 8
ROSTER_SEED = 53387310
REFERENCE_SEED = 74573552
IDENTITY = ((0, 0, 0),) * 2
TURNS = (
    ((6, -5, 4), (-6, 5, -4)),
    ((-7, 5, 4), (6, -6, 5)),
    ((5, 7, -6), (-5, -7, 6)),
    ((5, -4, -5), (-4, 5, 5)),
)
MODES = ('carried', 'reset', 'uncorrected', 'intact')


def modes_for(tracking):
    return MODES + ('carried_raster', 'reset_raster') if tracking == 'paired' else MODES


def roster(seed, blocks, count):
    rng = np.random.default_rng(seed)
    course = []
    for phase in range(blocks):
        cases = []
        for i in range(count):
            direction = -1 if (i + phase) % 3 < 2 else 1
            offset = (.03, .36) if i % 2 == 0 else (.27, .62)
            lane = .10 + direction*rng.uniform(*offset)
            cases.append(dict(seed=int(rng.integers(1, 2**31 - 1)), lane=float(lane)))
        course.append(cases)
    return course


def validate(course):
    proposed = {row['seed'] for block in course for row in block}
    for name in ('compact-development.json.gz', 'compact-assessment.json.gz'):
        receipt = _read(OUTPUT / name)
        seen = {row['seed'] for stream in receipt['cases'].values() for block in stream.values()
                for paired in block for row in paired.values()}
        if proposed & seen:
            raise ValueError('release already used in a compact point experiment')
    if REFERENCE_SEED in proposed:
        raise ValueError('reference scenery release overlaps assessment')
    return audit([{i: list(block) for i, block in enumerate(course)}])


def compact_guides(station, mounts):
    arena = station.arena
    arena.lane = 0.
    mounts.move(IDENTITY)
    arena.reset(seed=REFERENCE_SEED, server=-1, render=False)
    nominal = arena.render_cameras()
    guides = {eye: Registration(nominal[eye], arena.lens_intrinsics()[eye]['fovy']) for eye in EYES}
    training = []
    for turn in SOURCE_ROTATIONS:
        mounts.move(turn)
        frames = arena.render_cameras()
        support = {eye: guides[eye].practice(frames[eye])[1] for eye in EYES}
        training.append(dict(angles=turn, matches=support,
                             rgb={eye: sha256(frames[eye]).hexdigest() for eye in EYES}))
    return guides, training


def calibrate(station, mounts, guides, turn, start):
    arena = station.arena
    arena.lane = start['lane']
    mounts.move(turn)
    arena.reset(seed=start['seed'], server=-1, render=False)
    frames = arena.render_cameras()
    corrections, records = {}, {}
    for mode in ('carried', 'reset'):
        by_eye, stat = {}, {}
        for eye, angles in zip(EYES, turn, strict=True):
            transform, support, tiles = guides[eye].calibrate(frames[eye], mode == 'carried', BUDGET)
            by_eye[eye] = transform
            error = float(np.rad2deg(Rotation.from_matrix(transform @ true_direction(angles).T).magnitude()))
            stat[eye] = dict(rotation=transform.tolist(), agreement=support, tiles=tiles, angular_error_deg=error)
        corrections[mode], records[mode] = by_eye, stat
    prefix = sha256()
    for eye in EYES:
        prefix.update(frames[eye])
    return corrections, records, prefix.hexdigest()


def canonical_features(observation, guide, rotation):
    row = observation.copy()
    visible = np.isfinite(row).all(axis=1)
    if visible.any():
        # The tracker reports centers in its historical 480-wide convention.
        # Project the radius with the square root of the homography's local
        # area scale as well as projecting the center.
        scale = 480 / guide.width
        pixels = (row[visible, :2]+.5) / scale - .5
        rays = guide.rays(pixels) @ rotation.T
        row[visible, :2] = (guide.pixels(rays)+.5) * scale - .5
        row[visible, 2] *= np.abs(rays[:, 2])**-1.5
    return row


def correct_history(history, guides, matrices):
    original = history.read

    def corrected(observation, joints, target):
        canonical = np.stack([canonical_features(observation[:, index], guides[eye], matrices[eye])
                              for index, eye in enumerate(EYES)], axis=1)
        return original(canonical, joints, target)

    history.read = corrected


class CanonicalImages:
    """Inverse-map the output raster and run the unchanged tracker on it."""

    def __init__(self, guide, matrix):
        y, x = np.indices((guide.height, guide.width), dtype=float)
        rays = guide.rays(np.column_stack((x.ravel(), y.ravel())))
        projected = guide.pixels(rays @ matrix)
        self.coordinates = projected[:, ::-1].T.reshape(2, guide.height, guide.width)

    def __call__(self, rgb):
        return np.stack([map_coordinates(rgb[..., channel], self.coordinates, order=1,
                                         mode='constant', cval=0, prefilter=False)
                         for channel in range(3)], axis=-1)


class ImageController:
    def __init__(self, controller, remappers):
        self.controller = controller
        self.remappers = remappers

    def command(self, frames, capture, tick, joints, target):
        canonical = {eye: self.remappers[eye](frames[eye]) for eye in EYES}
        return self.controller.command(canonical, capture, tick, joints, target)


def block(task):
    index, turn, cases, stage, tracking = task
    station = CompactStation(stage)
    options = Options()
    mounts = CameraMounts(station.arena)
    modes = modes_for(tracking)
    try:
        guides, training = compact_guides(station, mounts)
        matrices, fits, fitting_rgb = calibrate(station, mounts, guides, turn, cases[0])
        warps = ({mode: {eye: CanonicalImages(guides[eye], matrices[mode][eye]) for eye in EYES}
                  for mode in matrices} if tracking != 'centers' else {})
        rows = []
        for case_index, start in enumerate(cases):
            branches = {}
            offset = case_index % len(modes)
            order = modes[offset:] + modes[:offset]
            for mode in order:
                mounts.move(IDENTITY if mode == 'intact' else turn)
                controller = options.new('combined')
                fit = mode.removesuffix('_raster')
                if fit in matrices:
                    if tracking == 'raster' or mode.endswith('_raster'):
                        controller = ImageController(controller, warps[fit])
                    else:
                        correct_history(controller.history, guides, matrices[fit])
                branches[mode] = station.play(start, controller)
                score = branches[mode]
                print('compact camera', index, start['seed'], mode, score['outcome']['legal_returns'],
                      score['outcome']['rival_returns'], score['longest'], flush=True)
            original = branches['uncorrected']['first_rgb']
            if any(branches[mode]['first_rgb'] != original for mode in modes if mode != 'intact'):
                raise ValueError('corrected and uncorrected branches received different first frames')
            if case_index == 0 and original != fitting_rgb:
                raise ValueError('mount correction did not use the first ordinary stereo observation')
            if original == branches['intact']['first_rgb']:
                raise ValueError('rotated and intact observations are identical')
            rows.append(dict(start, results=branches))
        return dict(index=index, angle_degrees=turn, practice=training, fits=fits, rows=rows,
                    first_calibration_rgb=fitting_rgb, summary=summarize(rows, modes))
    finally:
        station.close()


def summarize(rows, modes):
    return {mode: brief([row['results'][mode] for row in rows]) for mode in modes}


def film(path):
    from .tt23_sequence_coach_movie import render_tape

    receipt = _read(path)
    setting, point = max(((block['angle_degrees'], row['results']['carried'])
                          for block in receipt['blocks'] for row in block['rows']),
                         key=lambda item: item[1]['longest'])

    def arena_factory():
        arena = CompactStation.make_arena(receipt['metadata']['court'], width=DIMS[0], height=DIMS[1])
        CameraMounts(arena).move(setting)
        return arena

    render_tape(point, path, OUTPUT / 'compact_camera-live.mp4', branch='carried',
                subtitle='Solid compact court · Rotated stereo with carried registration',
                terminal_caption=f"{point['longest']} alternating returns · Normal speed at 30 fps",
                arena_factory=arena_factory)


def tracking_diagnostic(path):
    """Score detector/warp order on the same physical tape from an earlier course."""
    source = OUTPUT / 'compact-assessment.json.gz'
    receipt = _read(source)
    row = next(case['retained'] for stream in receipt['cases'].values() for case in stream['rebound']
               if case['retained']['seed'] == MOVIE_SEED)
    station = CompactStation('rebound')
    arena = station.arena
    mounts = CameraMounts(arena)
    try:
        guides, _ = compact_guides(station, mounts)
        turn = TURNS[0]
        matrices, fits, _ = calibrate(station, mounts, guides, turn, row)
        matrices['exact'] = {eye: true_direction(angles) for eye, angles in zip(EYES, turn, strict=True)}
        warps = {mode: {eye: CanonicalImages(guides[eye], fit[eye]) for eye in EYES}
                 for mode, fit in matrices.items()}
        samples = {'intact': [], 'uncorrected': []}
        for mode in matrices:
            samples[f'{mode}_centers'] = []
            samples[f'{mode}_raster'] = []
        arena.lane = row['lane']
        arena.reset(seed=row['seed'], server=-1, render=False)
        for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
            if tick % 4 == 0:
                mounts.move(IDENTITY)
                samples['intact'].append(snapshot(arena.render_cameras()))
                mounts.move(turn)
                frames = arena.render_cameras()
                raw = snapshot(frames)
                samples['uncorrected'].append(raw)
                for mode, fit in matrices.items():
                    features = [canonical_features(raw[:, index], guides[eye], fit[eye])
                                for index, eye in enumerate(EYES)]
                    samples[f'{mode}_centers'].append(np.stack(features, axis=1))
                    samples[f'{mode}_raster'].append(snapshot({eye: warps[mode][eye](frames[eye]) for eye in EYES}))
            rival = np.asarray(far)
            arena._aim = arena._command_to = lambda *args, rival=rival: rival
            _, reward, done, cap, _ = arena.step(near, render=False)
            if bool(done or cap) != (tick + 1 == len(row['actions'])):
                raise ValueError('tracking diagnostic changed the tape length')
        if reward != row['reward'] or digest_json(arena.events) != digest_json(row['events']):
            raise ValueError('tracking diagnostic changed the referee tape')
    finally:
        station.close()
    reference = np.asarray(samples.pop('intact'))
    metrics = {}
    for mode, candidate in samples.items():
        candidate = np.asarray(candidate)
        metrics[mode] = {}
        for i, target in enumerate(('ball', 'blade')):
            found = np.isfinite(candidate[:, i]).all(axis=-1)
            known = np.isfinite(reference[:, i]).all(axis=-1)
            matched = found & known
            error = (candidate[:, i] - reference[:, i])[matched] * (DIMS[0]/480)
            distance = np.linalg.norm(error[:, :2], axis=1)
            metrics[mode][target] = dict(common=int(matched.sum()), missing=int((known & ~found).sum()),
                                         extras=int((found & ~known).sum()),
                                         median_center_px=float(np.median(distance)),
                                         p90_center_px=float(np.quantile(distance, .9)),
                                         median_radius_abs_px=float(np.median(abs(error[:, 2]))))
    _write(path, dict(reference_tape_sha256=_hash(source), seed=MOVIE_SEED, angles=turn, fitting=fits,
                      sampled_frames=len(reference), interval=4, physical_replay_verified=True, summary=metrics))
    print(metrics, flush=True)


def run(path, count, changes, seed, tracking, workers, offset=0):
    course = roster(seed, changes, count)
    checks = validate(course)
    stage = 'rebound'
    modes = modes_for(tracking)
    meta = dict(seed=seed, count=count, changes=changes, court=stage, tracking=tracking, archives_checked=checks,
                source_reference_seed=REFERENCE_SEED, motor_training={p.name: _hash(p) for p in (CACHE, OPTICAL, COMBINED)},
                receiver='same frozen TT24 combined fit in every branch', patch_budget_per_eye=BUDGET,
                held_camera_observations=1, source_camera_poses=SOURCE_ROTATIONS, held_course=course)
    if offset:
        meta['turn_offset'] = offset
    report = _read(path) if path.exists() else dict(metadata=meta, blocks=[None]*changes)
    if report['metadata'] != meta:
        # Normalize tuples and integer keys through the same receipt writer.
        import json
        if report['metadata'] != json.loads(json.dumps(meta)):
            raise ValueError('existing compact camera receipt has a different roster')
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [(i, TURNS[i+offset], cases, stage, tracking) for i, cases in enumerate(course)
             if report['blocks'][i] is None]

    def save(result):
        report['blocks'][result['index']] = result
        complete = [row for record in report['blocks'] if record is not None for row in record['rows']]
        report['summary'] = summarize(complete, modes)
        report['verified_tapes'] = len(complete) * len(modes)
        _write(path, report)

    if workers == 1:
        for task in tasks:
            save(block(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
            futures = [pool.submit(block, task) for task in tasks]
            for future in as_completed(futures):
                save(future.result())
    print(report['summary'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--changes', type=int, choices=range(1, len(TURNS)+1), default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=ROSTER_SEED)
    p.add_argument('--tracking', choices=('centers', 'raster', 'paired'), default='centers')
    p.add_argument('--turn-offset', type=int, default=0)
    p.add_argument('--inspect-tracking', action='store_true')
    p.add_argument('--film', action='store_true')
    p.add_argument('--output', type=Path, default=OUTPUT / 'compact_camera-assessment.json.gz')
    args = p.parse_args()
    if args.count < 1:
        p.error('--count must be positive')
    if args.turn_offset < 0 or args.turn_offset + args.changes > len(TURNS):
        p.error('requested more camera turns than the held roster contains')
    if args.film:
        film(args.output)
    elif args.inspect_tracking:
        tracking_diagnostic(OUTPUT / 'compact_camera-tracking.json.gz')
    else:
        run(args.output, args.count, args.changes, args.seed, args.tracking, args.workers, args.turn_offset)
