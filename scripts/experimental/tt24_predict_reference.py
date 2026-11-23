"""Learn bounce feedback in a reference made from fixed training images."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena, mujoco
from .tt24_bounce_ceiling import PrivilegedCeiling
from .tt24_bounce_ident import DirectPupil, _ellipse, pack, select_pair, summarize
from .tt24_motor import optical_sightings


DATA = Path('artifacts/tt24/predict-reference-data.npz')
RESULT = Path('artifacts/tt24/predict-reference.json.gz')
DEMOS = tuple(range(249001, 249005))
HELD = tuple(range(250001, 250013))
HEIGHT = 144
WIDTH = 192
TICK = 1. / 60
COMMAND_UNIT = 5.2 / 60


class TrainingEyes(BounceArena):
    """Two stationary training viewpoints aimed at the table-end workspace."""

    def __init__(self, *, view='wrist'):
        super().__init__(view=view, height=HEIGHT, width=WIDTH, max_steps=360)
        self._model.vis.global_.fovy = 22.
        self.eyes = []
        for azimuth in (0., 90.):
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = (0., -.94, .92)
            camera.distance = 1.60
            camera.elevation = -12.
            camera.azimuth = azimuth
            self.eyes.append(camera)

    def fixed_images(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, width=self.width, height=self.height)
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_camlight(self._model, self._data)
        images = []
        for eye in self.eyes:
            self._renderer.update_scene(self._data, camera=eye, scene_option=self._scene_option)
            images.append(self._renderer.render())
        return images


def image_read(rgb):
    ball = _ellipse(rgb)
    _, blade = optical_sightings(rgb)
    ball = np.full(3, np.nan) if ball is None else ball
    blade_values = np.full(4, np.nan) if blade is None else np.array((blade.center[0] * HEIGHT,
        -blade.center[1] * HEIGHT, blade.radius * HEIGHT, blade.half_span * HEIGHT))
    return np.r_[ball, blade_values]


def photograph(arena, obs, *, wrist=True, fixed=True):
    moving = image_read(obs['rgb']) if wrist else np.full(7, np.nan)
    still = np.asarray([image_read(rgb) for rgb in arena.fixed_images()]) if fixed else np.full((2, 7), np.nan)
    return moving, still


def collect(path):
    arena = TrainingEyes()
    try:
        archive = {}
        for seed in DEMOS:
            obs = arena.reset(seed=seed, split='held')
            teacher = PrivilegedCeiling(arena)
            wrist, fixed, enc, commands = [], [], [], []
            done = False
            learned_target = np.zeros(5)
            while not done:
                optical, stationary = photograph(arena, obs)
                wrist.append(optical)
                fixed.append(stationary)
                enc.append(np.r_[obs['joints'], obs['joint_velocity']])
                action = teacher.command()
                learned_target = learned_target + COMMAND_UNIT * action
                commands.append(learned_target)
                obs, _, done = arena.step(action)
            info = arena.judge_metrics()
            labels = dict(wrist=wrist, fixed=fixed, enc=enc, target=commands,
                          rebounds=np.array([info['longest_consecutive_rebounds']]))
            archive.update({f'{seed}_{key}': np.asarray(value) for key, value in labels.items()})
            senses = np.asarray(fixed)
            print('reference demonstration', seed, len(commands), info['longest_consecutive_rebounds'],
                  'eye ball coverage', np.isfinite(senses[:, :, 0]).mean(axis=0).tolist(),
                  'paddle coverage', np.isfinite(senses[:, :, 3]).mean(axis=0).tolist(), flush=True)
        np.savez_compressed(path, **archive)
    finally:
        arena.close()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return [{key: archive[f'{seed}_{key}'] for key in ('wrist', 'fixed', 'enc', 'target', 'rebounds')}
                for seed in DEMOS]


def fit(design, output, ridge):
    reg = np.eye(design.shape[1]) * ridge
    reg[0, 0] *= .03
    return np.linalg.solve(design.T @ design + reg, design.T @ output)


def upper(x):
    i, j = np.triu_indices(x.shape[1])
    return x[:, i] * x[:, j]


class Reference:
    """Compose image coordinates from two upright views measured in silhouette units."""

    def __init__(self, examples):
        pictures = np.concatenate([example['fixed'] for example in examples])
        self.scale = .020 / np.nanmedian(pictures[:, :, 2], axis=0)
        self.default = np.nanmedian(self.read(pictures), axis=0)

    def read(self, pictures):
        ball, blade = pictures[:, :, :2], pictures[:, :, 3:5]
        ball = ball * self.scale[None, :, None]
        blade = blade * self.scale[None, :, None]
        ball_xyz = np.c_[ball[:, :, 0], ball[:, :, 1].mean(axis=1)]
        blade_xyz = np.c_[blade[:, :, 0], blade[:, :, 1].mean(axis=1)]
        aspect = (pictures[:, :, 5] / pictures[:, :, 6])**2
        return np.c_[ball_xyz, blade_xyz, aspect * .18]


class WristReadout:
    """Predict fixed-eye locations from a wrist silhouette and joint observations."""

    def __init__(self, examples, reference):
        stock = np.concatenate([example['enc'] for example in examples])
        wrist = np.concatenate([example['wrist'] for example in examples])
        target = reference.read(np.concatenate([example['fixed'] for example in examples]))
        self.center = stock.mean(axis=0)
        self.scale = np.maximum(stock.std(axis=0), np.r_[np.full(5, .055), np.full(5, .45)])
        self.retinal_center = np.nanmedian(self.optics(wrist), axis=0)
        self.retinal_scale = np.maximum(np.nanstd(self.optics(wrist), axis=0), .08)
        good_ball = np.isfinite(wrist[:, :3]).all(axis=1) & np.isfinite(target[:, :3]).all(axis=1)
        good_blade = np.isfinite(target[:, 3:]).all(axis=1)
        self.pose = fit(self.body_design(stock[good_blade]), target[good_blade, 3:], .035)
        self.ball = fit(self.ball_design(wrist[good_ball], stock[good_ball]), target[good_ball, :3], .10)
        self.bounds = np.column_stack((np.nanpercentile(target, 1, axis=0) - .12,
                                       np.nanpercentile(target, 99, axis=0) + .12))

    @staticmethod
    def optics(wrist):
        radius = wrist[:, 2]
        return np.c_[wrist[:, :2] / radius[:, None], HEIGHT / radius]

    def body_design(self, encoders):
        q = np.clip((encoders - self.center) / self.scale, -4., 4.)
        return np.c_[np.ones(len(q)), q[:, :5], .30 * q[:, 5:], upper(q[:, :5])]

    def ball_design(self, wrist, encoders):
        forms = self.body_design(encoders)
        visual = np.clip((self.optics(wrist) - self.retinal_center) / self.retinal_scale, -6., 6.)
        coupling = (visual[:, :, None] * forms[:, None, 1:6]).reshape(len(wrist), -1)
        optical = upper(visual)
        return np.column_stack((forms, visual, optical, coupling))

    def read(self, wrist, encoders):
        ball = self.ball_design(wrist, encoders) @ self.ball
        paddle = self.body_design(encoders) @ self.pose
        read = np.c_[ball, paddle]
        return np.clip(read, self.bounds[:, 0], self.bounds[:, 1])


class SingleEye:
    """Retain optical coordinates and proximity without an outside camera."""

    def __init__(self, examples):
        pictures = np.concatenate([example['wrist'] for example in examples])
        self.radius = np.nanmedian(pictures[:, 2])
        self.paddle_default = np.nanmedian(pictures[:, 3:], axis=0)
        self.default = np.nanmedian(self.read(pictures), axis=0)

    def read(self, wrist):
        wrist = np.asarray(wrist)
        optics = WristReadout.optics(wrist)
        ball = .022 * np.c_[optics[:, 0], optics[:, 2], optics[:, 1]]
        paddle = np.where(np.isfinite(wrist[:, 3:]), wrist[:, 3:], self.paddle_default)
        blade = .022 * np.c_[paddle[:, 0] / self.radius, np.zeros(len(wrist)), paddle[:, 1] / self.radius]
        geometry = .18 * np.c_[(paddle[:, 2] / paddle[:, 3])**2, paddle[:, 3] / self.radius]
        return np.c_[ball, blade, geometry]


def replace_gaps(samples, default):
    """Apply the same causal missing-data rule to data and live observations."""
    values = []
    recent = default.copy()
    for row in samples:
        recent = np.where(np.isfinite(row), row, recent)
        values.append(recent)
    return np.asarray(values)


def velocities(positions):
    """Supervise motion from centered optical differences without labels from physics."""
    n = len(positions)
    i = np.arange(n)
    differences = (positions[np.minimum(i + 1, n - 1)] - positions[np.maximum(i - 1, 0)]) / (2. * TICK)
    p = np.clip(differences, -5., 5.)
    smoothed = p.copy()
    smoothed[1:-1] = .70 * p[1:-1] + .15 * (p[:-2] + p[2:])
    return smoothed


class Motion:
    """Learn current reference velocity from causal differences of visual positions."""

    def __init__(self, observations, reference_velocity):
        inp = np.concatenate([self.history(row[:, :3] - row[0, 3:6]) for row in observations])
        target = np.concatenate(reference_velocity)
        self.center = inp.mean(axis=0)
        self.scale = np.maximum(inp.std(axis=0), np.r_[np.full(3, .032), np.full(12, .16)])
        self.weights = fit(self.design(inp), target, 1.6)

    @staticmethod
    def history(positions):
        index = np.arange(len(positions))
        return np.c_[positions, *[(positions - positions[np.maximum(index - lag, 0)]) / (lag * TICK)
                                  for lag in (1, 2, 3, 5)]]

    def design(self, inputs):
        x = np.clip((inputs - self.center) / self.scale, -5., 5.)
        return np.c_[np.ones(len(x)), x, .50 * upper(x)]

    def predict(self, inputs):
        return np.clip(self.design(inputs) @ self.weights, -4.7, 4.7)


def pupil_inputs(read, velocity, encoders):
    anchor = read[0, 3:6]
    joints = encoders[:, :5]
    home = np.broadcast_to(joints[0], joints.shape)
    return np.c_[read[:, :3] - anchor, velocity, read[:, 3:6] - anchor, read[:, 6:], home, joints - home]


def references(examples):
    ref = Reference(examples)
    readout = WristReadout(examples, ref)
    ref_read = [replace_gaps(ref.read(item['fixed']), ref.default) for item in examples]
    wrist_read = [replace_gaps(readout.read(item['wrist'], item['enc']), ref.default) for item in examples]
    optical_targets = [velocities(read[:, :3]) for read in ref_read]
    ref_motion = Motion(ref_read, optical_targets)
    wrist_motion = Motion(wrist_read, optical_targets)
    lone = SingleEye(examples)
    lone_read = [replace_gaps(lone.read(item['wrist']), lone.default) for item in examples]
    lone_motion = Motion(lone_read, [velocities(read[:, :3]) for read in lone_read])
    return dict(ref=ref, readout=readout, ref_motion=ref_motion, wrist_motion=wrist_motion,
                lone=lone, lone_motion=lone_motion, ref_read=ref_read, lone_read=lone_read,
                wrist_read=wrist_read, optical_targets=optical_targets)


def visual_validation(examples):
    errors = []
    for i, test in enumerate(examples):
        models = references([item for j, item in enumerate(examples) if i != j])
        ref, decoder, motion = (models[key] for key in ('ref', 'readout', 'wrist_motion'))
        loc = replace_gaps(ref.read(test['fixed']), ref.default)
        mapped = replace_gaps(decoder.read(test['wrist'], test['enc']), ref.default)
        vel = velocities(loc[:, :3])
        predicted = motion.predict(motion.history(mapped[:, :3] - mapped[0, 3:6]))
        finite = np.isfinite(test['fixed'][:, :, :3]).all(axis=(1, 2)) & np.isfinite(test['wrist'][:, :3]).all(axis=1)
        errors.append(dict(seed=DEMOS[i], frames=int(finite.sum()),
                           apparent_position_rmse=np.sqrt(np.mean((loc[finite, :3] - mapped[finite, :3])**2, axis=0)).tolist(),
                           apparent_motion_rmse=np.sqrt(np.mean((vel[finite] - predicted[finite])**2, axis=0)).tolist()))
    return errors


def make_actor(examples, reads, vels):
    groups = tuple((pupil_inputs(scene, vel, item['enc']), item['target'])
                   for scene, vel, item in zip(reads, vels, examples, strict=True))
    pair, scores = select_pair(groups)
    return DirectPupil(*pack(groups), pair=pair), scores


def play(arena, seed, mode, models, actor):
    use_wrist = mode in ('transfer', 'single')
    obs = arena.reset(seed=seed, split='held', render=use_wrist)
    issued = np.zeros(5)
    history = []
    actions = []
    gaps = 0
    done = False
    while not done:
        if mode == 'hold':
            action = np.zeros(5)
        else:
            wrist, fixed = photograph(arena, obs, wrist=use_wrist, fixed=mode == 'reference')
            encoded = np.r_[obs['joints'], obs['joint_velocity']][None]
            if mode == 'reference':
                read = models['ref'].read(fixed[None])[0]
                motion = models['ref_motion']
                default = models['ref'].default
            elif mode == 'transfer':
                read = models['readout'].read(wrist[None], encoded)[0]
                motion = models['wrist_motion']
                default = models['ref'].default
            else:
                read = models['lone'].read(wrist[None])[0]
                motion = models['lone_motion']
                default = models['lone'].default
            if not history:
                home = obs['joints'].copy()
                anchor = np.where(np.isfinite(read), read, default)[3:6]
                last = default
            gaps += int(not np.isfinite(read[:3]).all())
            last = np.where(np.isfinite(read), read, last)
            history.append(last[:3] - anchor)
            history = history[-6:]
            movement = motion.predict(motion.history(np.asarray(history))[-1:])[0]
            features = np.r_[history[-1], movement, last[3:6] - anchor, last[6:], home, obs['joints'] - home]
            action = actor.command(features, issued)
        issued += COMMAND_UNIT * action
        actions.append(action.tolist())
        obs, _, done = arena.step(action, render=use_wrist)
    return arena.judge_metrics() | dict(visual_gap_ticks=gaps, actions=actions)


def qualify(data, output):
    examples = load(data)
    models = references(examples)
    validation = visual_validation(examples)
    print('cross-view visual validation', validation, flush=True)
    debug = [models['lone_motion'].predict(Motion.history(row[:, :3] - row[0, 3:6]))
             for row in models['lone_read']]
    ref_actor, ref_scoring = make_actor(examples, models['ref_read'], models['optical_targets'])
    single_actor, single_scoring = make_actor(examples, models['lone_read'], debug)
    print('selected channel pairs', ref_actor.pair, single_actor.pair, flush=True)
    evaluation = {}
    arena = TrainingEyes()
    try:
        for mode in ('hold', 'reference', 'transfer', 'single'):
            actor = single_actor if mode == 'single' else ref_actor
            evaluation[mode] = []
            for seed in HELD:
                row = play(arena, seed, mode, models, actor)
                evaluation[mode].append(row)
                print('image reference held', mode, seed, row['longest_consecutive_rebounds'], row['visual_gap_ticks'], flush=True)
    finally:
        arena.close()
    with gzip.open(output, 'wt') as stream:
        json.dump(dict(schema='tt24-visual-reference-1', visual_validation=validation,
                       teacher_streaks=[item['rebounds'][0].item() for item in examples],
                       policy_pair_scores=dict(reference=ref_scoring, single=single_scoring), held=evaluation,
                       summary={key: summarize(rows) for key, rows in evaluation.items()}),
                  stream, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'qualify'))
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=RESULT)
    args = parser.parse_args()
    if args.stage == 'collect':
        collect(args.data)
    else:
        qualify(args.data, args.output)


if __name__ == '__main__':
    main()
