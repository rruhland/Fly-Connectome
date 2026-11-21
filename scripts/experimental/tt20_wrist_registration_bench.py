"""Time wrist registration on the same native frames and verify the saved decisions."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from statistics import median
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np

from . import tt20_natural as sensor
from .tt4_transfer_trial import _read
from .tt19_factory_serve_trial import load_checkpoint
from .tt20_fixture import TT20Arena
from .tt20_mono_flight_trial import CATALOG, CHECKPOINT, OUTPUT


def capture(report, indexes):
    arena = TT20Arena(**report['arena'], view_mode='wrist')
    tapes = []
    try:
        for index in indexes:
            row = report['pairs'][index]['temporal']
            arena.reset(seed=row['seed'], server=-1, render=False)
            images = []
            for action in row['actions']:
                images.append(arena.render_cameras()['wrist'])
                arena.step(action, render=False)
            if arena.point_result() != row['result'] or json.dumps(arena.events) != json.dumps(row['events']):
                raise ValueError(f'point {row["seed"]} changed during physical tape replay')
            tapes.append((images, row['actions']))
        return tapes, arena.lens_intrinsics()
    finally:
        arena.close()


def registration(tapes, lenses):
    poses = []
    for images, _ in tapes:
        track = sensor.NaturalViewTransport('wrist', lens_fov=lenses, flight=False, continuous_wrist=True)
        active, earlier = {}, None
        for tick, image in enumerate(images):
            active = track.observe({'wrist': image}, tick, active, earlier)
            pose = active.get('wrist')
            if pose is None:
                poses.append(None)
                continue
            earlier = pose, tick
            poses.append((pose.eye.tobytes(), pose.rotation.tobytes(), pose.rms, pose.markers))
    return poses


def decisions(tapes, lenses):
    player = load_checkpoint(CHECKPOINT, CATALOG, 'factory').player
    try:
        for images, expected in tapes:
            tracker = sensor.NaturalViewTransport('wrist', lens_fov=lenses, flight=False,
                                                  continuous_wrist=True).tracker()
            player.reset()
            for tick, (image, target) in enumerate(zip(images, expected, strict=True)):
                tracker.update({'wrist': image})
                ball = tracker.ball
                position = ball.position if ball is not None and ball.observed else None
                action = player.command(tick, position, -1)
                np.testing.assert_array_equal(action, target)
    finally:
        player.close()


def benchmark(path, indexes, reference, repeats):
    tapes, lenses = capture(_read(path), indexes)
    fitters = {'current': sensor.register_table}
    if reference is not None:
        spec = spec_from_file_location(f'{__package__}._wrist_reference', reference)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        fitters['reference'] = module.register_table
    timings = {name: [] for name in fitters}
    expected = None
    for repeat in range(repeats):
        order = list(fitters) if repeat % 2 == 0 else list(reversed(fitters))
        for name in order:
            with patch.object(sensor, 'register_table', fitters[name]):
                wall, cpu = perf_counter(), process_time()
                poses = registration(tapes, lenses)
                timings[name].append(dict(wall_s=perf_counter() - wall, cpu_s=process_time() - cpu))
            if expected is None:
                expected = poses
            elif poses != expected:
                raise ValueError(f'{name} produced different camera fits')
    for register in fitters.values():
        with patch.object(sensor, 'register_table', register):
            decisions(tapes, lenses)
    return dict(frames=len(expected), fitted_frames=sum(pose is not None for pose in expected),
                exact_decision_tapes=len(tapes), replayed_points=len(tapes), timing=timings,
                median_cpu_frames_per_s={name: len(expected) / median(item['cpu_s'] for item in rows)
                                         for name, rows in timings.items()})


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=OUTPUT / 'natural-wrist-continuity-play.json.gz')
    cli.add_argument('--reference', type=Path, help='earlier tt20_natural_geometry.py to compare')
    cli.add_argument('--indexes', type=int, nargs='+', default=[0, 1, 5])
    cli.add_argument('--repeats', type=int, default=3)
    args = cli.parse_args()
    if args.repeats < 1 or min(args.indexes) < 0:
        cli.error('choose positive repeats and zero-based point indexes')
    print(json.dumps(benchmark(args.source, args.indexes, args.reference, args.repeats), indent=2))
