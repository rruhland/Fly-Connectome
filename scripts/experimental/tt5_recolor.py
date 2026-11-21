"""Compare blade tracking and scored play after recoloring the receiving racket.

Run the seven-color study from the repository root.

    python -m scripts.experimental.tt5_recolor recolor-outline.json.gz

Use --colors dusty_blue --scored-points 1 for a smaller native-rendered check.
Only the receiving racket's paint changes. The two trackers share the same
charged RGB probes and use identical held point seeds wherever calibration works.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TABLE_HEIGHT, TT4Arena
from .tt4_native_temporal_trial import _probe
from .tt4_transfer import TransferPrior, calibration_commands
from .tt4_transfer_trial import ROOT, Wiring, _counts, _hash, _metrics, _point, _write, load_sources
from .tt5_player import RallyPlayer, fit_arm
from .tt5_vision import hall_tracker_factory, motion_tracker_factory


COLORS = {
    'blue': (69, 118, 195),
    'periwinkle': (107, 113, 202),
    'dusty_blue': (74, 130, 171),
    'teal': (40, 171, 109),
    'matte_ochre': (151, 111, 68),
    'original_red': (192, 46, 48),
    'original_black': (43, 37, 39),
}
SCORED_SEEDS = {'dusty_blue': range(69292, 69296), 'teal': range(69296, 69300)}
FACTORIES = {'frozen': hall_tracker_factory, 'outline': motion_tracker_factory}
GEOMETRY = dict(venue='hall', render_mode='mujoco', view='first_person', width=192,
                height=144, baseline=.28, frameskip=8, max_steps=140)
PROBE_HORIZON = 28
ACCURACY_THRESHOLD = .045


class _RecoloredArena(TT4Arena):
    def __init__(self, paint, *args, **kwargs):
        self.paint = np.array(paint) / 255.
        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, render=True):
        super().reset(seed=seed, render=False)
        self.model.geom('learner_racket').rgba[:3] = self.paint
        return self.render_cameras() if render else None


def _blade_audit(arena, wiring, seed):
    frames = wiring.reset(seed=seed)
    commands = calibration_commands(seed, PROBE_HORIZON)
    samples = []
    for tick in range(len(commands) + 1):
        if tick:
            frames, _, terminated, truncated, _ = wiring.step(commands[tick - 1])
            if terminated or truncated:
                break
        center = arena.paddle - np.array((0, 0, TABLE_HEIGHT))
        pitch = float(np.arcsin(arena._scene_pose(1)[3][2]))
        samples.append((frames, center, pitch))
    return samples, dict(seed=seed, commands=commands.tolist())


def _tracking_metrics(factory, episodes):
    errors, angles = [], []
    accurate = 0
    for samples in episodes:
        tracker = factory()
        for frames, target, pitch in samples:
            tracker.update(frames)
            actor = tracker.actor
            if actor is None or not actor.observed:
                continue
            error = np.abs(np.subtract(actor.position, target))
            accurate += int(np.linalg.norm(error) <= ACCURACY_THRESHOLD)
            errors.append(error)
            if tracker.actor_pitch is not None:
                angles.append(abs(tracker.actor_pitch - pitch))
    return dict(frames=sum(map(len, episodes)), detections=len(errors), centers_within_45mm=accurate,
                median_absolute_axis_error_mm=(np.median(errors, axis=0) * 1000).tolist() if errors else None,
                median_abs_pitch_error_rad=float(np.median(angles)) if angles else None)


def _scored_play(wiring, schedule, probes, factory, seeds):
    try:
        adapter = fit_arm(probes, factory)
    except ValueError as error:
        return dict(calibration_error=str(error), scheduled=len(seeds), completed=0, played=[])
    native, states, _ = load_sources()
    prior = TransferPrior('legacy_mini', native, schedule['player_seed'], mini=states['legacy_mini'])
    player = RallyPlayer(adapter, prior, factory)
    points = []
    for seed in seeds:
        row = _point(wiring, player, seed)
        points.append(dict(**row, **_counts(row)))
    return dict(adapter=adapter.stats(), scheduled=len(seeds), completed=len(points),
                metrics=_metrics(points), played=points)


def _compare(job):
    name, scored_points = job
    index = tuple(COLORS).index(name)
    first = 69222 + 5 * index
    color = COLORS[name]
    schedule = dict(arena_seed=69244 + index, view_seed=69259 + index, player_seed=69224 + index,
                    calibration=list(range(first, first + 3)))
    tests = [first + 3, first + 4]
    scored = list(SCORED_SEEDS.get(name, ()))[:scored_points]
    started = perf_counter()
    arena = _RecoloredArena(color, schedule['arena_seed'], view_seed=schedule['view_seed'], **GEOMETRY)
    wiring = Wiring(arena, schedule['arena_seed'])
    result = dict(variant=name, color=color, schedule=schedule, tracking_seeds=tests, scored_seeds=scored,
                  evaluator_wiring=wiring.state(), counts={}, play={})
    try:
        probes, calibration = _probe(wiring, schedule, PROBE_HORIZON)
        result['calibration'] = calibration
        result['paid_probe_frames'] = sum(row['frames'] for row in calibration)
        episodes, tapes = zip(*(_blade_audit(arena, wiring, seed) for seed in tests), strict=True)
        result['tracking_tapes'] = tapes
        for condition, fit in FACTORIES.items():
            factory = fit(arena.cameras, probes)
            result['counts'][condition] = _tracking_metrics(factory, episodes)
            if scored:
                result['play'][condition] = _scored_play(wiring, schedule, probes, factory, scored)
    finally:
        arena.close()
    result['wall_seconds'] = perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output', type=Path, help='write a JSON or JSON.GZ receipt')
    parser.add_argument('--colors', choices=list(COLORS), nargs='+', default=list(COLORS))
    parser.add_argument('--scored-points', type=int, choices=range(5), default=4,
                        help='play this many held points at each of the dusty blue and teal stations')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.colors)) != len(args.colors):
        parser.error('expected positive workers and unique colors')

    folder = ROOT / 'scripts' / 'experimental'
    modules = ('tt5_recolor.py', 'tt5_vision.py', 'tt5_player.py', 'tt5_player_body.py',
               'tt4_arena.py', 'tt4_arena.xml', 'tt4_camera.py', 'tt4_hall.py',
               'tt4_native_temporal.py', 'tt4_native_temporal_trial.py', 'tt4_transfer.py',
               'tt4_transfer_trial.py')
    report = dict(schema='tt5-regulation-hall-recolor-outline-v1', geometry=dict(**GEOMETRY, standard_shading=True),
                  implementation_sha256={name: _hash(folder / name) for name in modules},
                  source_sha256=load_sources()[2], probe_horizon=PROBE_HORIZON,
                  accuracy_threshold_m=ACCURACY_THRESHOLD, contested_play_source='legacy_mini', settings={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(name, args.scored_points) for name in args.colors]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for row in pool.map(_compare, jobs):
            report['settings'][row['variant']] = row
            _write(args.output, report)
            play = {condition: {key: run[key] for key in ('completed', 'metrics', 'calibration_error') if key in run}
                    for condition, run in row['play'].items()}
            print(json.dumps(dict(variant=row['variant'], counts=row['counts'], play=play)), flush=True)


if __name__ == '__main__':
    main()
