"""Fit bounce control from metric optical tracks and deploy from wrist images."""

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt24_bounce import mujoco
from .tt24_bounce_ident import WristPupil, WristState, _unit_camera_ray, summarize
from .tt24_predict_reference import (COMMAND_UNIT, DATA, HEIGHT, Motion, TrainingEyes, fit, load,
                                     make_actor, photograph, replace_gaps, upper, velocities)


RESULT = Path('artifacts/tt24/metric-stereo.json.gz')
DIAGNOSTIC = Path('artifacts/tt24/metric-stereo-frozen-head.json.gz')
MOTION_DIAGNOSTIC = Path('artifacts/tt24/metric-stereo-frozen-motion.json.gz')
PUPIL = Path('artifacts/tt24/bounce-ident-pupil.npz')
HELD = tuple(range(254001, 254013))
DIAGNOSTIC_HELD = tuple(range(255001, 255009))
WRIST_FOCAL = HEIGHT / (2. * np.tan(np.deg2rad(108.) / 2.))


@dataclass(frozen=True)
class Camera:
    center: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    focal: float
    principal: np.ndarray

    def rays(self, positions):
        p = (positions - self.principal) / self.focal
        direction = self.forward + p[:, :1] * self.right + p[:, 1:] * self.up
        return direction / np.linalg.norm(direction, axis=1, keepdims=True)

    def project(self, points):
        vector = points - self.center
        depth = vector @ self.forward
        image = np.column_stack((vector @ self.right, vector @ self.up))
        return self.principal + self.focal * image / depth[:, None], depth

    @classmethod
    def from_fixed_renderer(cls, scene):
        # This reads the calibrated stationary mount and lens only. Moving wrist
        # poses, reconstructed scene contents and velocities are never read.
        cam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
        fwd, up = (np.array(field, dtype=float) for field in (cam.forward, cam.up))
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        focal = HEIGHT * cam.frustum_near / (cam.frustum_top - cam.frustum_bottom)
        principal = -focal / cam.frustum_near * np.array((cam.frustum_center,
                                                         (cam.frustum_top + cam.frustum_bottom) / 2.))
        return cls(np.array(cam.pos, dtype=float), right, up, fwd, focal, principal)


def calibrated_cameras(arena):
    arena.reset(seed=249001, split='held', render=False)
    if arena._renderer is None:
        arena._renderer = mujoco.Renderer(arena._model, width=arena.width, height=arena.height)
    eyes = []
    for eye in arena.eyes:
        arena._renderer.update_scene(arena._data, camera=eye, scene_option=arena._scene_option)
        eyes.append(Camera.from_fixed_renderer(arena._renderer.scene))
    return tuple(eyes)


def triangulate(cameras, pixels, tolerance):
    a, b = cameras
    rd0, rd1 = a.rays(pixels[:, 0]), b.rays(pixels[:, 1])
    offset = b.center - a.center
    mutual = np.sum(rd0 * rd1, axis=1)
    denom = np.maximum(1. - mutual**2, 1.e-8)
    proj0, proj1 = rd0 @ offset, rd1 @ offset
    t0 = (proj0 - mutual * proj1) / denom
    t1 = (mutual * proj0 - proj1) / denom
    point0 = a.center + t0[:, None] * rd0
    point1 = b.center + t1[:, None] * rd1
    gap = np.linalg.norm(point1 - point0, axis=1)
    good = (t0 > 0.) & (t1 > 0.) & (gap < tolerance)
    point = .5 * (point0 + point1)
    return np.where(good[:, None], point, np.nan), gap, np.column_stack((t0, t1))


class Stereo:
    """Reconstruct ball and paddle position from perspective camera pairs."""

    def __init__(self, eyes, examples):
        self.eyes = eyes
        photos = np.concatenate([item['fixed'] for item in examples])
        points = self.read(photos)
        self.default = np.nanmedian(points, axis=0)
        radii = []
        for i, camera in enumerate(eyes):
            _, depths = camera.project(points[:, :3])
            pixels = photos[:, i, 2]
            radii.extend(depths * pixels / np.sqrt(camera.focal**2 + pixels**2))
        self.radius = float(np.nanmedian(radii))

    def read(self, photos):
        ball, _, _ = triangulate(self.eyes, photos[:, :, :2], tolerance=.028)
        blade, _, _ = triangulate(self.eyes, photos[:, :, 3:5], tolerance=.040)
        aspect = .18 * (photos[:, :, 5] / photos[:, :, 6])**2
        return np.column_stack((ball, blade, aspect))

    def check(self, photos):
        result = {}
        for name, columns in (('ball', slice(0, 2)), ('blade', slice(3, 5))):
            pixels = photos[:, :, columns]
            triangulated, separation, distance = triangulate(self.eyes, pixels, tolerance=np.inf)
            finite = np.isfinite(pixels).all(axis=(1, 2))
            good = np.isfinite(triangulated).all(axis=1)
            predicted = np.stack([eye.project(triangulated[good])[0] for eye in self.eyes], axis=1)
            error = (np.sum((predicted - pixels[good])**2, axis=2)).reshape(-1)
            result[name] = dict(paired_frames=int(finite.sum()), frontal_frames=int(good.sum()),
                                pixels_rmse=float(np.sqrt(error.mean())),
                                median_ray_gap=float(np.nanmedian(separation[good])),
                                median_depth=np.median(distance[good], axis=0).tolist(),
                                metric_span=np.ptp(triangulated[good], axis=0).tolist())
        return result


class WristRegistration:
    """Fit a moving rigid camera map from encoders and stereo RGB correspondences."""

    def __init__(self, reference, examples):
        wrist = np.concatenate([item['wrist'] for item in examples])
        enc = np.concatenate([item['enc'] for item in examples])
        points = reference.read(np.concatenate([item['fixed'] for item in examples]))
        self.radius = reference.radius
        self.center = enc[:, :5].mean(axis=0)
        self.scale = np.maximum(enc[:, :5].std(axis=0), .06)
        q = self.joints(enc)
        ball_valid = np.isfinite(wrist[:, :3]).all(axis=1) & np.isfinite(points[:, :3]).all(axis=1)
        target = points[ball_valid, :3]
        q_ball = q[ball_valid]
        camera = _unit_camera_ray(wrist[ball_valid, :3], WRIST_FOCAL) * self.radius
        camera_context = self.context(q_ball)
        translation_forms = self.translation_forms(q_ball)
        cov = (camera - camera.mean(axis=0)).T @ (target - target.mean(axis=0))
        u, _, v = np.linalg.svd(cov)
        rotate = v.T @ np.diag((1., 1., np.linalg.det(v.T @ u.T))) @ u.T
        initial_w = np.zeros((camera_context.shape[1], 3))
        initial_w[0] = Rotation.from_matrix(rotate).as_rotvec()
        initial_t = fit(translation_forms, target - camera @ rotate.T, .10)
        shape = initial_w.size

        def cost(parameter):
            w, t = parameter[:shape].reshape(initial_w.shape), parameter[shape:].reshape(initial_t.shape)
            estimate = self.apply(camera, camera_context, translation_forms, w, t)
            return np.r_[(estimate - target).ravel(), .028 * w[1:].ravel(), .22 * t[1:].ravel()]

        solved = least_squares(cost, np.r_[initial_w.ravel(), initial_t.ravel()], loss='soft_l1', f_scale=.007,
                               max_nfev=60, xtol=1.e-7, ftol=1.e-7)
        self.rotation = solved.x[:shape].reshape(initial_w.shape)
        self.translation = solved.x[shape:].reshape(initial_t.shape)
        self.cost = float(solved.cost)
        self.gradient = float(solved.optimality)
        blade_valid = np.isfinite(points[:, 3:]).all(axis=1)
        self.blade = fit(self.translation_forms(q[blade_valid]), points[blade_valid, 3:], .06)

    @staticmethod
    def apply(camera, context, forms, rotation, translation):
        return Rotation.from_rotvec(context @ rotation).apply(camera) + forms @ translation

    def joints(self, enc):
        return np.clip((enc[:, :5] - self.center) / self.scale, -4., 4.)

    @staticmethod
    def context(q):
        return np.c_[np.ones(len(q)), q]

    @staticmethod
    def translation_forms(q):
        return np.c_[np.ones(len(q)), q, .40 * upper(q)]

    def read(self, wrist, enc):
        q = self.joints(enc)
        good = np.isfinite(wrist[:, :3]).all(axis=1)
        ball = np.full((len(wrist), 3), np.nan)
        if np.any(good):
            ray = _unit_camera_ray(wrist[good, :3], WRIST_FOCAL) * self.radius
            ball[good] = self.apply(ray, self.context(q[good]), self.translation_forms(q[good]),
                                    self.rotation, self.translation)
        paddle = self.translation_forms(q) @ self.blade
        return np.column_stack((ball, paddle))


class ReplaceSourceBallMap:
    """Swap only the frozen pupil's source ball sighting map for the RGB fit."""

    image_height = HEIGHT

    def __init__(self, source, own):
        self.source, self.own = source, own
        self.default = source.default

    def body(self, joints):
        self.last = np.r_[joints, np.zeros(5)][None]
        return self.source.body(joints)

    def ball(self, ellipse, body):
        if ellipse is None:
            return None
        optics = np.r_[ellipse, np.zeros(4)][None]
        return self.own.read(optics, self.last)[0, :3]


class ReplaceSourceMotion(WristState):
    """Supply the source actor with causal velocity learned from stereo image motion."""

    def __init__(self, decoder, optical_motion):
        super().__init__(decoder)
        self.optical_motion = optical_motion
        self.optical_home = None
        self.optical_history = []

    def observe(self, obs):
        state = super().observe(obs)
        if self.optical_home is None:
            enc = self.decoder.last
            self.optical_home = (self.decoder.own.translation_forms(self.decoder.own.joints(enc)) @
                                 self.decoder.own.blade)[0, :3]
        self.optical_history.append(state[:3] + self.home - self.optical_home)
        self.optical_history = self.optical_history[-6:]
        design = Motion.history(np.asarray(self.optical_history))[-1:]
        self.velocity = self.optical_motion.predict(design)[0]
        return np.r_[state[:3], self.velocity, state[6:]]


def motion_inputs(estimator, locations):
    return [estimator.predict(Motion.history(row[:, :3] - row[0, 3:6])) for row in locations]


def train(examples, eyes):
    reference = Stereo(eyes, examples)
    wrist = WristRegistration(reference, examples)
    static = [replace_gaps(reference.read(item['fixed']), reference.default) for item in examples]
    moving = [replace_gaps(wrist.read(item['wrist'], item['enc']), reference.default) for item in examples]
    motion = [velocities(scene[:, :3]) for scene in static]
    fixed_motion = Motion(static, motion)
    wrist_motion = Motion(moving, motion)
    return reference, wrist, static, moving, motion, fixed_motion, wrist_motion


def validate(examples, eyes):
    checks = []
    for i, test in enumerate(examples):
        ref, wrist, _, _, _, _, motion = train([item for j, item in enumerate(examples) if i != j], eyes)
        expected = ref.read(test['fixed'])
        inferred = wrist.read(test['wrist'], test['enc'])
        usable = np.isfinite(expected[:, :3]).all(axis=1) & np.isfinite(inferred[:, :3]).all(axis=1)
        filled = replace_gaps(inferred, ref.default)
        velocity = motion_inputs(motion, [filled])[0]
        future = velocities(replace_gaps(expected, ref.default)[:, :3])
        checks.append(dict(frames=int(usable.sum()), position_rmse=np.mean((inferred[usable, :3] -
                                                                            expected[usable, :3])**2, axis=0)**.5,
                           velocity_rmse=np.mean((velocity[usable] - future[usable])**2, axis=0)**.5))
    return [{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in item.items()}
            for item in checks]


def play(arena, seed, mode, actor, stereo, wrist, motion):
    from_wrist = mode in ('transfer', 'aligned')
    obs = arena.reset(seed=seed, split='held', render=from_wrist)
    commands = []
    recent = []
    last = stereo.default.copy()
    home = obs['joints'].copy()
    issued = np.zeros(5)
    missing = 0
    done = False
    while not done:
        if mode == 'hold':
            action = np.zeros(5)
        else:
            retinal, fixed = photograph(arena, obs, wrist=from_wrist, fixed=not from_wrist)
            if from_wrist:
                enc = np.r_[obs['joints'], obs['joint_velocity']][None]
                read = wrist.read(retinal[None], enc)[0]
            else:
                read = stereo.read(fixed[None])[0]
            missing += int(not np.isfinite(read[:3]).all())
            last = np.where(np.isfinite(read), read, last)
            if not recent:
                blade_home = last[3:6].copy()
            recent.append(last[:3] - blade_home)
            recent = recent[-6:]
            velocity = motion.predict(Motion.history(np.asarray(recent))[-1:])[0]
            features = np.r_[recent[-1], velocity, last[3:6] - blade_home, last[6:], home, obs['joints'] - home]
            action = actor.command(features, issued)
        issued += COMMAND_UNIT * action
        commands.append(action.tolist())
        obs, _, done = arena.step(action, render=from_wrist)
    return arena.judge_metrics() | dict(commands=commands, visual_gaps=missing)


def run(path, output):
    examples = load(path)
    arena = TrainingEyes()
    try:
        eyes = calibrated_cameras(arena)
        stereo, wrist, static, moving, motion, fixed_motion, wrist_motion = train(examples, eyes)
        optical = stereo.check(np.concatenate([item['fixed'] for item in examples]))
        print('stereo optical checks', optical, 'ball radius', stereo.radius, flush=True)
        cross = validate(examples, eyes)
        print('cross-domain visual validation', cross, flush=True)
        actor, scores = make_actor(examples, static, motion)
        aligned, aligned_scores = make_actor(examples, moving, motion_inputs(wrist_motion, moving))
        print('channel pairs', actor.pair, aligned.pair, flush=True)
        held = {}
        for mode in ('hold', 'stereo', 'transfer', 'aligned'):
            rows = []
            for seed in HELD:
                velocity = wrist_motion if mode in ('transfer', 'aligned') else fixed_motion
                result = play(arena, seed, mode, aligned if mode == 'aligned' else actor, stereo, wrist, velocity)
                rows.append(result)
                print('metric held', mode, seed, result['longest_consecutive_rebounds'], result['visual_gaps'], flush=True)
            held[mode] = rows
    finally:
        arena.close()
    optics = [{key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in vars(eye).items()}
              for eye in eyes]
    with gzip.open(output, 'wt') as stream:
        json.dump(dict(schema='tt24-metric-stereo-1', stationary_lenses=optics, optical=optical,
                       sphere_radius_from_images=stereo.radius, cross_validation=cross, held=held,
                       pair_selection=dict(stereo=scores, wrist=aligned_scores),
                       summary={key: summarize(rows) for key, rows in held.items()}), stream, separators=(',', ':'))


def frozen_head(path, stored, output, *, optical_motion=False):
    examples = load(path)
    arena = TrainingEyes()
    arena.max_steps = 300
    held = {}
    try:
        eyes = calibrated_cameras(arena)
        _, lookup, _, _, _, _, motion = train(examples, eyes)
        modes = ('rgb_position_and_motion',) if optical_motion else ('original', 'rgb_stereo_ball_map')
        for mode in modes:
            rows = []
            for seed in DIAGNOSTIC_HELD:
                pupil = WristPupil.load(stored)
                if mode != 'original':
                    decoder = ReplaceSourceBallMap(pupil.sensor.decoder, lookup)
                    pupil.sensor = (ReplaceSourceMotion(decoder, motion) if optical_motion
                                    else WristState(decoder, pupil.sensor.motion))
                obs = arena.reset(seed=seed, split='held')
                done = False
                commands = []
                while not done:
                    action = pupil.command(obs)
                    commands.append(action.tolist())
                    obs, _, done = arena.step(action)
                row = arena.judge_metrics() | dict(commands=commands)
                rows.append(row)
                print('frozen metric diagnostic', mode, seed, row['longest_consecutive_rebounds'], flush=True)
            held[mode] = rows
    finally:
        arena.close()
    with gzip.open(output, 'wt') as stream:
        json.dump(dict(schema='tt24-metric-stereo-frozen-head-1', source=str(stored),
                       both_use_prior_privileged_body_and_head_supervision=True,
                       motion_learned_from_stereo=optical_motion,
                       held=held, summary={key: summarize(rows) for key, rows in held.items()}),
                  stream, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--frozen-head', action='store_true')
    parser.add_argument('--optical-motion', action='store_true')
    parser.add_argument('--source', type=Path, default=PUPIL)
    args = parser.parse_args()
    if args.frozen_head or args.optical_motion:
        default = MOTION_DIAGNOSTIC if args.optical_motion else DIAGNOSTIC
        frozen_head(args.data, args.source, args.out or default, optical_motion=args.optical_motion)
    else:
        run(args.data, args.out or RESULT)


if __name__ == '__main__':
    main()
