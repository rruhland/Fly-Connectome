"""Recalibrate the transferred wrist pupil using target RGB pairs only."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ceiling import PrivilegedCeiling
from .tt24_bounce_ident import WristPupil, episode, summarize
from .tt24_bounce_ident_transfer import DEMOS, TARGET_HELD, TARGET_REGIME, _verify_tape, _write
from .tt24_predict_reference import (DATA, DEMOS as SOURCE_PHOTO_SEEDS, Reference, TrainingEyes, WristReadout,
                                     fit, image_read, upper)


class SourceBridge:
    """Align the fixed public cameras to the source pupil's inferred coordinates."""

    def __init__(self, pupil, reference, examples):
        decoder = pupil.sensor.decoder
        angles = np.concatenate([row['enc'][:, :5] for row in examples])
        fixed = reference.read(np.concatenate([row['fixed'] for row in examples]))
        wrist = np.concatenate([row['wrist'] for row in examples])
        body = np.asarray([decoder.body(q) for q in angles])
        good_ball = np.isfinite(wrist[:, :3]).all(axis=1) & np.isfinite(fixed[:, :3]).all(axis=1)
        good_blade = np.isfinite(fixed[:, 3:]).all(axis=1)
        inferred = np.asarray([decoder.ball(ellipse, pose)
                               for ellipse, pose in zip(wrist[good_ball, :3], body[good_ball], strict=True)])
        self.center = np.nanmedian(fixed, axis=0)
        self.scale = np.maximum(np.nanstd(fixed, axis=0), .015)
        self.ball_weights = fit(self._basis(fixed[good_ball], slice(0, 3)), inferred, .10)
        self.paddle_weights = fit(self._basis(fixed[good_blade], slice(3, 8)), body[good_blade, 12:17], .20)

    def _basis(self, values, bounds):
        x = np.clip((values[:, bounds] - self.center[bounds]) / self.scale[bounds], -6., 6.)
        return np.column_stack((np.ones(len(x)), x, upper(x)))

    def read(self, values):
        ball = self._basis(values, slice(0, 3))[0] @ self.ball_weights
        blade = self._basis(values, slice(3, 8))[0] @ self.paddle_weights
        return ball, blade


class PairedRGBState:
    """Use a calibrated public reference, then the original source motion fit."""

    def __init__(self, optics, bridge, motion, default):
        self.optics, self.bridge, self.motion = optics, bridge, motion
        self.default = default
        self.home = None
        self.history = []
        self.last = None
        self.velocity = np.zeros(3)

    def observe(self, obs):
        visual = image_read(obs['rgb'])
        enc = np.r_[obs['joints'], obs['joint_velocity']]
        measured = np.isfinite(visual[:3]).all()
        if not measured:
            # This fallback supplies a finite placeholder for the pose decoder.
            visual[:3] = self.default
        predicted = self.optics.read(visual[None, :], enc[None, :])
        ball, blade = self.bridge.read(predicted)
        if not measured and self.last is not None:
            ball = self.last + self.velocity / 60
        if self.home is None:
            self.home = blade[:3].copy()
        self.history.append(ball)
        self.history = self.history[-4:]
        history = [(ball - self.history[max(0, len(self.history) - 1 - lag)]) * (60 / lag)
                   for lag in (1, 2, 3)]
        self.velocity = self.motion.predict(np.r_[ball - self.home, *history])
        self.last = ball
        return np.r_[ball - self.home, self.velocity, blade[:3] - self.home, blade[3:]]


def _target_photographs(primary):
    prototype = TrainingEyes()
    arena = BounceArena(view='wrist', width=192, height=144, arm='alternate')
    arena._model.vis.global_.fovy = prototype._model.vis.global_.fovy
    arena.eyes = prototype.eyes
    prototype.close()
    episodes = []
    try:
        for expected in primary['demonstrations']:
            obs = arena.reset(seed=expected['seed'], split='held', regime=TARGET_REGIME)
            teacher = PrivilegedCeiling(arena)
            wrist, fixed, encoders = [], [], []
            for _ in range(arena.max_steps):
                wrist.append(image_read(obs['rgb']))
                fixed.append([image_read(frame) for frame in TrainingEyes.fixed_images(arena)])
                encoders.append(np.r_[obs['joints'], obs['joint_velocity']])
                obs, _, done = arena.step(teacher.command())
                if done:
                    break
            assert json.loads(json.dumps(arena.judge_metrics())) == expected
            episodes.append(dict(wrist=np.asarray(wrist), fixed=np.asarray(fixed), enc=np.asarray(encoders)))
            print('paired RGB recalibration example', expected['seed'], len(wrist), flush=True)
    finally:
        arena.close()
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24'))
    parser.add_argument('--source-optics', type=Path, default=DATA)
    args = parser.parse_args()
    pupil = WristPupil.load(args.out / 'bounce-ident-pupil.npz')
    with gzip.open(args.out / 'bounce-ident-transfer.json.gz', 'rt', encoding='utf-8') as file:
        primary = json.load(file)
    with np.load(args.source_optics, allow_pickle=False) as archive:
        stock = [{key: archive[f'{seed}_{key}'] for key in ('wrist', 'fixed', 'enc')}
                 for seed in SOURCE_PHOTO_SEEDS]
    reference = Reference(stock, calibrated=True)
    bridge = SourceBridge(pupil, reference, stock)
    target = _target_photographs(primary)
    evaluation = BounceArena(view='wrist', width=192, height=144, arm='alternate')
    evidence = dict(source='source wrist checkpoint distilled with privileged supervision',
                    target_calibration='paired public wrist and fixed RGB plus own encoders',
                    motor_labels_used_for_target_recalibration=False, target_seeds=DEMOS,
                    held_seeds=TARGET_HELD, evaluation='wrist RGB and own encoders', cohorts={})
    try:
        for count in (1, 4):
            recordings = target[:count]
            optics = WristReadout(recordings, reference)
            default = np.nanmedian(np.concatenate([data['wrist'][:, :3] for data in recordings]), axis=0)
            sensor = lambda _: PairedRGBState(optics, bridge, pupil.sensor.motion, default)
            rows = [episode(evaluation, seed, actor=pupil.actor, regime=TARGET_REGIME,
                            sensor_factory=sensor)[0] for seed in TARGET_HELD]
            top = max(rows, key=lambda row: row['longest_consecutive_rebounds'])
            _verify_tape(evaluation, top, TARGET_REGIME)
            summary = summarize(rows)
            evidence['cohorts'][count] = dict(summary=summary, episodes=rows, replay_seed=top['seed'],
                                             frames=sum(len(data['wrist']) for data in recordings))
            print('target public RGB recalibration', count, summary, flush=True)
    finally:
        evaluation.close()
    _write(args.out / 'bounce-ident-transfer-public.json.gz', evidence)


if __name__ == '__main__':
    main()
