"""Score one-camera RGB depth through the held TT18 camera-removal course."""

import argparse
from itertools import chain
import json
from pathlib import Path

import numpy as np

from .tt18_monocular import AppearanceDepth, Ray, StereoContinuation
from .tt18_view import ImageColors, ViewRegistry, object_blobs, triangulate_views
from .tt18_view_trial import ORIGIN, PHASES, ViewHall, hall_images, rig_schedule


HELD_CAMERA_SEEDS = (236026241, 258998363, 838991611, 453647027, 720213857, 174047849)
HELD_PATH_SEEDS = (918826991, 402379163, 535562741, 883039687, 916835247, 107650489)
DROP_START = next(tick for tick, phase in PHASES if phase == 'monocular')
METHODS = ('abstain', 'median_depth', 'appearance', 'qualified_appearance',
           'last_point', 'linear', 'linear_on_ray', 'gravity', 'stereo_ray')


def score(position, truth, ray, *, interval=None, source_frames=None):
    error = position - truth
    along_ray = abs(float(error @ ray.direction))
    return dict(estimate_m=position.tolist(), error_m=float(np.linalg.norm(error)),
                range_error_m=along_ray, interval_half_width_m=interval,
                interval_contains_range=None if interval is None else along_ray <= interval,
                source_frames=source_frames)


def score_continuation(continuation, tick, camera, blob, truth):
    ray = Ray.observed(camera, blob.center)
    stereo_ray, continuations = continuation.predict(tick, camera, blob)
    methods = {name: score(estimate, truth, ray) for name, estimate in continuations.items()}
    if stereo_ray is not None:
        methods['stereo_ray'] = score(stereo_ray.position, truth, ray, interval=stereo_ray.range_half_width,
                                     source_frames=stereo_ray.source_frames)
    return methods


def score_frame(appearance, continuation, tick, camera, blob, stream, truth):
    ray = Ray.observed(camera, blob.center)
    scale, median_depth = appearance.predict(stream, camera, blob)
    methods = score_continuation(continuation, tick, camera, blob, truth)
    if median_depth is not None:
        methods['median_depth'] = score(median_depth, truth, ray)
    if scale is not None:
        methods['appearance'] = score(scale.position, truth, ray, interval=scale.range_half_width,
                                      source_frames=scale.source_frames)
        if appearance.qualified(stream, scale, camera, blob):
            methods['qualified_appearance'] = methods['appearance']
    return methods


def run_course(camera_seed, path_seed, *, width, height, resize):
    arena = ViewHall(seed=path_seed, body='rotary', venue='hall', width=width, height=height,
                    resize=resize, max_steps=32, frameskip=8)
    schedule = rig_schedule(camera_seed)
    registry = ViewRegistry()
    appearance = AppearanceDepth()
    continuation = StereoContinuation(arena.control_dt)
    rng = np.random.default_rng(path_seed)
    amplitude = rng.uniform(.22, .53, 6)
    offset = rng.uniform(-np.pi, np.pi, 6)
    rows, source_ticks = [], []
    try:
        arena.reset(seed=path_seed, render=False)
        for tick in range(arena.max_steps):
            phase = next(name for start, name in reversed(PHASES) if tick >= start)
            if rows and phase != 'monocular':
                break
            images = {key: ImageColors(image) for key, image in hall_images(arena, schedule[phase]).items()}
            fits = registry.observe(images)
            blobs = object_blobs(images)['ball']
            if phase == 'monocular':
                truth = arena.ball - ORIGIN
                methods = {}
                if len(fits) == 1:
                    stream, camera = next(iter(fits.items()))
                    if stream in blobs:
                        methods = score_frame(appearance, continuation, tick, camera, blobs[stream], stream, truth)
                rows.append(dict(tick=tick, registered=list(fits), detected=list(blobs), truth_m=truth.tolist(),
                                 methods=methods))
            else:
                position = triangulate_views(fits, {key: blob.center for key, blob in blobs.items()})
                before = appearance.source_frames
                appearance.observe(fits, blobs, position)
                continuation.observe(tick, position)
                if appearance.source_frames > before:
                    source_ticks.append(tick)
            if arena.terminated or arena.truncated:
                break
            arena.step(amplitude * np.sin(.18 * tick + offset), render=False)
    finally:
        arena.close()
    return dict(camera_seed=camera_seed, trajectory_seed=path_seed, table_resized=resize,
                source_ticks=source_ticks, source_frames=appearance.source_frames,
                last_mount_source_frames={stream: len(entry[1]) for stream, entry in appearance.streams.items()},
                monocular=rows)


def metrics(rows, method):
    covered = [row['methods'][method] for row in rows if method in row['methods']]
    errors = [1000. * row['error_m'] for row in covered]
    interval_rows = [row for row in covered if row['interval_half_width_m'] is not None]
    return dict(frames=len(rows), coverage=len(covered),
                median_3d_mm=float(np.median(errors)) if errors else None,
                p95_3d_mm=float(np.quantile(errors, .95)) if errors else None,
                median_range_mm=float(1000. * np.median([row['range_error_m'] for row in covered])) if covered else None,
                median_range_half_width_mm=float(1000. * np.median([row['interval_half_width_m']
                                                                   for row in interval_rows])) if interval_rows else None,
                range_inside=sum(row['interval_contains_range'] for row in interval_rows) if interval_rows else None)


def summarize(rows):
    by_method = {method: metrics(rows, method) for method in METHODS}
    matched = {}
    for method, controls in (('appearance', ('median_depth',)),
                             ('qualified_appearance', ('median_depth', 'appearance')),
                             ('stereo_ray', ('last_point', 'linear', 'linear_on_ray', 'gravity', 'median_depth'))):
        common = [row for row in rows if method in row['methods']]
        matched[method] = {name: metrics(common, name) for name in (method, *controls)}
    return dict(frames=len(rows), methods=by_method, matched=matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tt18/mono-depth.json'))
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=192)
    args = parser.parse_args()
    courses = []
    for index, (camera_seed, path_seed) in enumerate(zip(HELD_CAMERA_SEEDS, HELD_PATH_SEEDS, strict=True)):
        course = run_course(camera_seed, path_seed, width=args.width, height=args.height, resize=bool(index % 2))
        courses.append(course)
        print(f"course {index + 1}, {course['source_frames']} stereo frames, "
              f"{len(course['monocular'])} one-camera frames", flush=True)
    rows = list(chain.from_iterable(course['monocular'] for course in courses))
    by_drop_frame = {tick - DROP_START + 1: summarize([row for row in rows if row['tick'] == tick])
                     for tick in sorted({row['tick'] for row in rows})}
    result = dict(schema='tt18-monocular-depth-v1', resolution=[args.width, args.height],
                  calibration_motor_commands=0, summary=summarize(rows), by_drop_frame=by_drop_frame,
                  courses=courses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
