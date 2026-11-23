"""Measure anonymous visual binding and paired physical wrist rebound scores."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from .tt24_anonvision import AnonymousVision
from .tt24_bounce import BounceArena, WRIST_CAMERA, mujoco
from .tt24_motor import MotorEffects, MotorPolicy, optical_sightings


CHECKPOINT = Path('artifacts/tt24/motor-rate-continued.json.gz')
OUTPUT = Path('artifacts/tt24/tt24_anonvision.json.gz')
HELD_SEED = 3102026
REPAINT = ((.58, .07, .93, 1.), (.055, .82, .23, 1.))
WRIST_SHIFT = Rotation.from_euler('yx', (6., -4.), degrees=True).as_quat(scalar_first=True)


class _VisionAudit:
    def __init__(self, readout, *, previous_action=True):
        self.detector = AnonymousVision() if readout == 'anonymous' else None
        self.previous_action = previous_action
        self.sightings = (None, None)
        self.seconds = 0.
        self.calls = 0

    def observe(self, rgb, previous_action=None):
        start = perf_counter()
        if self.detector is not None:
            action = previous_action if self.previous_action else None
            self.sightings = self.detector.observe(rgb, action)
        else:
            self.sightings = optical_sightings(rgb)
        self.seconds += perf_counter() - start
        self.calls += 1
        return self.sightings


def _truth(arena):
    # The segmentation pass is audit-only. Policy has already issued its action.
    renderer = arena._renderer
    renderer.enable_segmentation_rendering()
    try:
        labels = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
    ids = np.where(labels[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, labels[..., 0], -1)
    result = []
    for index in (arena._ball_geom, arena._blade):
        rows, columns = np.nonzero(ids == index)
        if len(rows) < 3:
            result.append(None)
        else:
            result.append((np.array((columns.mean(), rows.mean())) -
                           ((arena.width - 1) / 2, (arena.height - 1) / 2)) / arena.height)
    return result


def _configure_view(arena, appearance, base_colors, base_rotation):
    # Change raster-only properties without changing the physical camera body.
    colors = REPAINT if appearance == 'repainted' else base_colors
    arena._model.geom_rgba[arena._ball_geom] = colors[0]
    arena._model.geom_rgba[arena._blade] = colors[1]
    rotation = np.empty(4)
    if appearance == 'shifted':
        mujoco.mju_mulQuat(rotation, base_rotation, WRIST_SHIFT)
    else:
        rotation[:] = base_rotation
    arena._model.camera(WRIST_CAMERA).quat[:] = rotation


def _play(arena, seed, checkpoint, *, split, readout='anonymous', appearance='native', replay=None,
          previous_action=True):
    effects = MotorEffects(arena.action_size)
    effects.readout[:] = checkpoint['motor_effects']
    audit = _VisionAudit(readout, previous_action=previous_action)
    policy = MotorPolicy(np.asarray(checkpoint['weights']), effects, readout=checkpoint['readout'],
                         retinal=checkpoint['retinal'], detector=audit)
    stride = checkpoint['stride']
    blind, passive = readout == 'blind', readout == 'passive'
    visual = not (blind or passive)
    obs = arena.reset(seed=seed, split=split, render=visual)
    actions = []
    detections = [dict(visible=0, returned=0, localized=0, false_visible=0, errors_px=[]) for _ in range(2)]
    tick = 0
    done = False
    while not done:
        sampled = tick % stride == 0
        if visual and sampled and appearance == 'permuted':
            obs['rgb'] = obs['rgb'][..., (2, 0, 1)]
        if passive:
            action = np.zeros(arena.action_size)
        else:
            action = policy.command(obs, sampled=sampled, blind=blind)
        if replay is not None:
            action = replay[tick]
            # Action-conditioned visual inference during replay observes the
            # motor command actually applied, the same as during live control.
            policy.last_action = action
        actions.append(action)
        if visual and sampled:
            for report, seen, true in zip(detections, audit.sightings, _truth(arena), strict=True):
                report['visible'] += int(true is not None)
                report['returned'] += int(seen is not None)
                report['false_visible'] += int(true is None and seen is not None)
                if true is not None and seen is not None:
                    error = float(np.linalg.norm(seen.center - true) * arena.height)
                    report['localized'] += int(error <= 2. * arena.height / checkpoint['height'])
                    report['errors_px'].append(error)
        tick += 1
        obs, _, done = arena.step(action, render=visual and tick % stride == 0)
    actions = np.array(actions)
    report = arena.judge_metrics() | dict(mover=detections[0], anchor=detections[1],
                                          calls=audit.calls, detector_seconds=audit.seconds,
                                          action_sha256=sha256(actions.astype('<f8').tobytes()).hexdigest())
    return report, actions


def _summarize(rows):
    result = dict(mean_rebounds=float(np.mean([row['rebounds'] for row in rows])),
                  rebound_counts=[row['rebounds'] for row in rows])
    calls = sum(row['calls'] for row in rows)
    result['detector_ms'] = round(1000. * sum(row['detector_seconds'] for row in rows) / calls, 3) if calls else None
    for role in ('mover', 'anchor'):
        reports = [row[role] for row in rows]
        errors = [error for report in reports for error in report['errors_px']]
        counts = {key: sum(report[key] for report in reports)
                  for key in ('visible', 'returned', 'localized', 'false_visible')}
        counts['localized_fraction'] = counts['localized'] / counts['visible'] if counts['visible'] else None
        counts['median_centroid_error_px'] = float(np.median(errors)) if errors else None
        result[role] = counts
    return result


def _write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        json.dump(result, stream, separators=(',', ':'))


def _paired(rows, reference):
    delta = np.array([a['rebounds'] - b['rebounds'] for a, b in zip(rows, reference, strict=True)])
    return dict(mean_difference=float(delta.mean()), better=int((delta > 0).sum()), tied=int((delta == 0).sum()),
                worse=int((delta < 0).sum()),
                identical_actions=sum(a['action_sha256'] == b['action_sha256']
                                      for a, b in zip(rows, reference, strict=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--episodes', type=int, default=14)
    parser.add_argument('--practice', action='store_true')
    parser.add_argument('--independent-held', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    started = perf_counter()
    with gzip.open(args.checkpoint, 'rt') as stream:
        checkpoint = json.load(stream)
    split = 'practice' if args.practice else 'held'
    if args.practice:
        launches = list(dict.fromkeys(checkpoint['practice_seeds']))[:args.episodes]
        schedule = 'checkpoint_practice'
    elif args.independent_held:
        launches = np.random.default_rng(HELD_SEED).integers(1, 2 ** 31 - 1, size=args.episodes).tolist()
        schedule = 'independent_held'
    else:
        launches = [row['seed'] for row in checkpoint['held']['live'][:args.episodes]]
        schedule = 'checkpoint_held'
    settings = (
        ('passive', 'passive', 'native', True),
        ('blind', 'blind', 'native', True),
        ('hue_native', 'hue', 'native', True),
        ('anonymous_native', 'anonymous', 'native', True),
        ('anonymous_native_no_actions', 'anonymous', 'native', False),
        ('hue_permuted', 'hue', 'permuted', True),
        ('anonymous_permuted', 'anonymous', 'permuted', True),
        ('hue_repainted', 'hue', 'repainted', True),
        ('anonymous_repainted', 'anonymous', 'repainted', True),
        ('hue_shifted', 'hue', 'shifted', True),
        ('anonymous_shifted', 'anonymous', 'shifted', True),
    )
    result = dict(schema='tt24-anonymous-visual-binding-1', checkpoint=str(args.checkpoint), split=split,
                  checkpoint_sha256=sha256(args.checkpoint.read_bytes()).hexdigest(), held_seed=HELD_SEED,
                  schedule=schedule, launches=launches, stride=checkpoint['stride'],
                  width=checkpoint['width'], height=checkpoint['height'],
                  repaint_rgba=REPAINT, wrist_camera_delta_xyzw=WRIST_SHIFT[[1, 2, 3, 0]].tolist(),
                  rows={}, summary={})
    native_actions = []
    arena = BounceArena(width=checkpoint['width'], height=checkpoint['height'])
    try:
        base_colors = arena._model.geom_rgba[[arena._ball_geom, arena._blade]].copy()
        base_rotation = arena._model.camera(WRIST_CAMERA).quat.copy()
        for name, readout, appearance, previous_action in settings:
            _configure_view(arena, appearance, base_colors, base_rotation)
            rows = []
            for seed in launches:
                report, actions = _play(arena, seed, checkpoint, split=split, readout=readout, appearance=appearance,
                                        previous_action=previous_action)
                rows.append(report)
                if name == 'anonymous_native':
                    native_actions.append(actions)
            result['rows'][name] = rows
            result['summary'][name] = _summarize(rows)
            print(name, result['summary'][name], flush=True)
            _write(args.output, result)
    finally:
        arena.close()

    # With a substantially different viewpoint, score semantic association
    # against the same physical trajectories rather than an untrained remapping.
    arena = BounceArena(view='external', width=2 * checkpoint['width'], height=2 * checkpoint['height'])
    try:
        rows = [_play(arena, seed, checkpoint, split=split, replay=actions)[0]
                for seed, actions in zip(launches, native_actions, strict=True)]
    finally:
        arena.close()
    result['rows']['anonymous_external_replay'] = rows
    result['summary']['anonymous_external_replay'] = _summarize(rows)
    comparisons = (
        ('anonymous_native', 'hue_native'), ('anonymous_native', 'blind'), ('anonymous_native', 'passive'),
        ('anonymous_native_no_actions', 'anonymous_native'), ('anonymous_permuted', 'anonymous_native'),
        ('anonymous_permuted', 'hue_permuted'), ('anonymous_repainted', 'hue_repainted'),
        ('anonymous_shifted', 'hue_shifted'), ('anonymous_external_replay', 'anonymous_native'),
    )
    result['paired'] = {f'{name}_vs_{reference}': _paired(result['rows'][name], result['rows'][reference])
                        for name, reference in comparisons}
    result['external_resolution'] = [arena.width, arena.height]
    result['localized_tolerance_native_px'] = 2
    result['wall_seconds'] = perf_counter() - started
    _write(args.output, result)
    print('anonymous_external_replay', result['summary']['anonymous_external_replay'], flush=True)


if __name__ == '__main__':
    main()
