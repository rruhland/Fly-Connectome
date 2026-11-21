"""Test a probe-learned RGB background under paired first-person MuJoCo play."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TT4Arena
from .tt4_native_rgb import native_tracker
from .tt4_native_temporal import modal_image, temporal_tracker
from .tt4_transfer import (ArmAdapter, CalibrationSample, TransferPlayer, TransferPrior,
                           ZERO, calibration_commands)
from .tt4_transfer_trial import (ROOT, Wiring, _counts, _hash, _metrics, _observed,
                                 _point, _seeds, _summary, _write, load_sources)


CONDITIONS = ('legacy_mini', 'first_person_mini', 'pong_only', 'fresh')
VIEW = 'first_person'


def _probe(arena, schedule, horizon):
    observations, calibration = [], []
    for index, seed in enumerate(schedule['calibration']):
        frames = arena.reset(seed=seed)
        history = [(ZERO, frames)]
        for action in calibration_commands(schedule['player_seed'] + index, horizon):
            frames, _, terminated, truncated, _ = arena.step(action)
            history.append((action, frames))
            if terminated or truncated:
                break
        observations.append(history)
        calibration.append(dict(seed=seed, steps=len(history) - 1, frames=len(history),
                                actions=[action.tolist() for action, _ in history[1:]]))
    return observations, calibration


def _fit(observations, factory):
    trajectories = []
    for history in observations:
        tracker = factory()
        trajectories.append([CalibrationSample(action, _observed(tracker, frames), tracker.appearances)
                             for action, frames in history])
    try:
        return ArmAdapter(trajectories), None
    except ValueError as error:
        return None, str(error)


def _course(job):
    schedule, native, sources, geometry = job
    start = perf_counter()
    arena = TT4Arena(schedule['arena_seed'], view_seed=schedule['view_seed'], view=VIEW, **geometry['arena'])
    wiring = Wiring(arena, schedule['arena_seed'])
    rig = arena.cameras
    try:
        observations, calibration = _probe(wiring, schedule, geometry['probe_horizon'])
        reference_frames = [frames for history in observations for _, frames in history[::2]]
        reference = modal_image(reference_frames)

        def factory():
            return temporal_tracker(rig, reference)

        adapter, error = _fit(observations, factory)
        original, original_error = _fit(observations, lambda: native_tracker(rig))
        course = dict(schedule=schedule, view=VIEW, calibration=calibration, evaluator_wiring=wiring.state(),
                      reference_frames=len(reference_frames), reference_sha256=sha256(reference.tobytes()).hexdigest(),
                      original_adapter=original.stats() if original else None, original_calibration_error=original_error,
                      adapter=adapter.stats() if adapter else None, calibration_error=error, runs={})
        if adapter is not None:
            for condition in CONDITIONS:
                source = sources.get(condition, sources['legacy_mini'])
                prior = TransferPrior(condition, native, schedule['player_seed'], mini=source)
                player = TransferPlayer(adapter, prior, factory)
                initial = prior.state()
                points = []
                for seed in schedule['tests']:
                    row = _point(wiring, player, seed)
                    points.append(dict(**row, **_counts(row)))
                course['runs'][condition] = dict(points=points, initial_prior=initial, final_prior=prior.state())
        course['wall_seconds'] = perf_counter() - start
        return course
    finally:
        arena.close()


def _run(args):
    native, sources, hashes = load_sources(args.first_person_mini)
    schedules = [_seeds(args.seed, index, args.probes, args.points) for index in range(args.installations)]
    geometry = dict(probe_horizon=args.probe_steps, arena=dict(
        width=192, height=144, baseline=.28, venue=args.venue, max_steps=140, frameskip=8,
        render_mode='mujoco', distractor=False,
    ))
    jobs = [(schedule, native, sources, geometry) for schedule in schedules]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    implementation = {path.name: _hash(path) for path in sorted(
        (ROOT / 'scripts' / 'experimental').glob('tt4*')) if path.is_file()}
    start = perf_counter()
    report = dict(schema='tt4-paired-native-temporal-v1', seed=args.seed, schedules=schedules,
                  conditions={VIEW: CONDITIONS}, inputs_sha256=hashes, implementation_sha256=implementation,
                  geometry=geometry, courses={VIEW: []}, summaries={})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for course in pool.map(_course, jobs):
            report['courses'][VIEW].append(course)
            finished = report['courses'][VIEW]
            accepted = [row for row in finished if row['adapter'] is not None]
            summary = _summary(accepted, CONDITIONS, args.points, args.seed) if accepted else {}
            summary['scheduled_points_per_condition'] = args.installations * args.points
            summary['calibration_failures'] = [row['schedule']['index'] for row in finished if row['adapter'] is None]
            summary['calibration_frames_including_failures'] = sum(
                probe['frames'] for row in finished for probe in row['calibration'])
            report['summaries'][VIEW] = summary
            report['wall_seconds'] = perf_counter() - start
            _write(args.output, report)
            print(json.dumps(dict(installation=course['schedule']['index'], adapter=course['adapter'],
                                  original_adapter=course['original_adapter'], error=course['calibration_error'],
                                  results={name: _metrics(run['points']) for name, run in course['runs'].items()})),
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--first-person-mini', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=258951348)
    parser.add_argument('--installations', type=int, default=2)
    parser.add_argument('--points', type=int, default=4)
    parser.add_argument('--probes', type=int, default=8)
    parser.add_argument('--probe-steps', type=int, default=34)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--venue', choices=('studio', 'hall'), default='studio')
    args = parser.parse_args()
    if min(args.points, args.installations, args.workers) < 1 or args.probes < 2 or args.probe_steps < 1:
        parser.error('expected positive counts and at least two common probe points')
    _run(args)


if __name__ == '__main__':
    main()
