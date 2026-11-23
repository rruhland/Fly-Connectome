"""Separate clean-state policy identification from wrist-based latent inference."""

import argparse
import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from .tt19_so101 import WRIST_CAMERA
from .tt24_bounce import BounceArena, mujoco
from .tt24_bounce_ceiling import PrivilegedCeiling


DEMOS = tuple(range(247_001, 247_005))
DEVELOPMENT = tuple(range(247_101, 247_109))
FRESH = tuple(range(247_201, 247_217))
_INC = 5.2 / 60


class CleanState:
    """Privileged diagnostic of relative position, motion and paddle pose."""

    def __init__(self, arena):
        self.arena = arena
        self.blade_home = arena._data.geom_xpos[arena._blade].copy()

    def observe(self, obs):
        arena = self.arena
        data = arena._data
        mujoco.mj_kinematics(arena._model, data)
        normal = data.geom_xmat[arena._blade].reshape(3, 3)[:, 2]
        normal = normal[:2] if normal[2] >= 0. else -normal[:2]
        return np.r_[arena._ball_position - self.blade_home, arena._ball_velocity,
                     data.geom_xpos[arena._blade] - self.blade_home, normal]


def _recent_velocity(history):
    if len(history) == 1:
        return np.zeros(3)
    immediate = (history[-1] - history[-2]) * 60
    averaged = (history[-1] - history[0]) * (60 / (len(history) - 1))
    return np.r_[averaged[:2], immediate[2]]


class CleanDifference(CleanState):
    """Apply the raw wrist time window to perfect relative position samples."""

    def __init__(self, arena):
        super().__init__(arena)
        self.history = []

    def observe(self, obs):
        state = super().observe(obs)
        self.history.append(state[:3].copy())
        self.history = self.history[-4:]
        state[3:6] = _recent_velocity(self.history)
        return state


def _ellipse(rgb):
    red, green, blue = np.moveaxis(rgb, -1, 0)
    mask = (red > 48) & (green > .19 * red) & (green < .69 * red) & (blue < .36 * red)
    components, count = label(mask)
    if not count:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    chosen = sizes.argmax()
    if sizes[chosen] < 15:
        return None
    yy, xx = np.nonzero(components == chosen)
    height, width = rgb.shape[:2]
    x0, y0 = xx.mean(), yy.mean()
    radius = math.sqrt(len(xx) / math.pi)
    min_x = np.full(height, width)
    max_x = np.full(height, -1)
    min_y = np.full(width, height)
    max_y = np.full(width, -1)
    np.minimum.at(min_x, yy, xx)
    np.maximum.at(max_x, yy, xx)
    np.minimum.at(min_y, xx, yy)
    np.maximum.at(max_y, xx, yy)
    left = np.flatnonzero((min_x > 0) & (min_x < width))
    right = np.flatnonzero((max_x >= 0) & (max_x < width - 1))
    top = np.flatnonzero((min_y > 0) & (min_y < height))
    bottom = np.flatnonzero((max_y >= 0) & (max_y < height - 1))
    edge = np.concatenate((np.column_stack((min_x[left] - .5, left)), np.column_stack((max_x[right] + .5, right)),
                           np.column_stack((top, min_y[top] - .5)), np.column_stack((bottom, max_y[bottom] + .5))))
    if len(edge) >= 12:
        x, y = ((edge - (x0, y0)) / radius).T
        design = np.column_stack((x * x, x * y, y * y, x, y))
        a, b, c, d, e = np.linalg.lstsq(design, np.ones(len(x)), rcond=None)[0]
        quad = np.array(((a, b / 2.), (b / 2., c)))
        strength = np.linalg.eigvalsh(quad)
        if min(strength) > .04:
            offset = np.linalg.solve(quad, -.5 * np.array((d, e)))
            if np.linalg.norm(offset) < 2.5:
                x0, y0 = np.array((x0, y0)) + radius * offset
                radius *= math.sqrt((1. + offset @ quad @ offset) / max(strength))
    return np.array((x0 - (width - 1) / 2., (height - 1) / 2. - y0, radius))


def _unit_camera_ray(ellipse, focal):
    u, v, minor = np.asarray(ellipse, dtype=float).T
    depth_squared = 1. + (focal / np.maximum(minor, 1.))**2
    depth = np.sqrt(depth_squared)
    tangent = (depth_squared - 1.) / (focal * depth)
    return np.stack((u * tangent, v * tangent, -depth), axis=-1)


class CalibrationState(CleanState):
    """Capture privileged pose targets only on the four demonstration trajectories."""

    def __init__(self, arena, records):
        super().__init__(arena)
        self.records = records

    def observe(self, obs):
        state = super().observe(obs)
        arena = self.arena
        camera = arena._data.camera(WRIST_CAMERA)
        rotation = camera.xmat.reshape(3, 3)
        ball_cam = rotation.T @ (arena._ball_position - camera.xpos)
        body = np.r_[camera.xpos, rotation.ravel(), arena._data.geom_xpos[arena._blade], state[9:11]]
        pixels = _ellipse(obs['rgb'])
        self.records.append((obs['joints'].copy(), body, ball_cam, pixels, state))
        return state


class WristDecoder:
    """Fit scene coordinates from silhouette optics and own encoder readings."""

    def __init__(self, records, focal, image_height=144):
        self.focal = focal
        self.image_height = image_height
        self.default = np.mean([sequence[0][-1][:3] for sequence in records], axis=0)
        samples = [record for sequence in records for record in sequence]
        joints, body, ball, ellipses, _ = zip(*samples, strict=True)
        self.center = np.mean(joints, axis=0)
        self.scale = np.maximum(.06, np.std(joints, axis=0))
        body_forms = self._body_forms(joints)
        regularizer = 1.e-4 * np.eye(body_forms.shape[1])
        self.body_weights = np.linalg.solve(body_forms.T @ body_forms + regularizer, body_forms.T @ body)
        valid = np.array([ellipse is not None for ellipse in ellipses])
        rays = _unit_camera_ray([ellipse for ellipse in ellipses if ellipse is not None], focal)
        actual = np.asarray(ball)[valid]
        self.radius = float(np.median(actual[:, 2] / rays[:, 2]))
        forms = self._camera_forms(rays)
        self.camera_weights = np.linalg.solve(forms.T @ forms + .1 * np.eye(forms.shape[1]), forms.T @ actual)

    def _body_forms(self, joints):
        x = np.clip((np.atleast_2d(joints) - self.center) / self.scale, -6., 6.)
        return np.column_stack((np.ones(len(x)), x, _power(x, 2), _power(x, 3)))

    def _camera_forms(self, rays):
        x = np.atleast_2d(rays) * self.radius / .20
        return np.column_stack((np.ones(len(x)), x, _power(x, 2)))

    def body(self, joints):
        return self._body_forms(joints)[0] @ self.body_weights

    def ball(self, ellipse, body):
        if ellipse is None:
            return None
        local = self._camera_forms(_unit_camera_ray(ellipse, self.focal))[0] @ self.camera_weights
        return body[:3] + body[3:12].reshape(3, 3) @ local


class WristState:
    """Runtime input adapter whose only inputs are wrist RGB and own encoders."""

    def __init__(self, decoder, motion=None):
        self.decoder = decoder
        self.motion = motion
        self.home = None
        self.previous = None
        self.velocity = np.zeros(3)
        self.history = []
        self.gaps = 0

    def observe(self, obs):
        rgb = obs.get('rgb')
        ellipse = _ellipse(rgb) if rgb is not None else None
        if ellipse is not None:
            ellipse *= getattr(self.decoder, 'image_height', 144) / rgb.shape[0]
        return self.read(ellipse, obs['joints'])

    def read(self, ellipse, joints):
        body = self.decoder.body(joints)
        if self.home is None:
            self.home = body[12:15].copy()
        ball = self.decoder.ball(ellipse, body)
        if ball is None:
            self.gaps += 1
            ball = self.home + self.decoder.default if self.previous is None else self.previous + self.velocity / 60
        self.history.append(ball)
        self.history = self.history[-4:]
        self.motion_input = np.concatenate((ball - self.home, *[
            (ball - self.history[max(0, len(self.history) - 1 - lag)]) * (60 / lag) for lag in (1, 2, 3)]))
        self.velocity = (self.motion.predict(self.motion_input) if self.motion is not None
                         else _recent_velocity(self.history))
        self.previous = ball
        return np.r_[ball - self.home, self.velocity, body[12:15] - self.home, body[15:17]]


class MotionFit:
    """Learn a causal velocity estimate from recent position differences."""

    def __init__(self, decoder, records, *, ridge=3., quadratic=True):
        xx, yy = [], []
        for sequence in records:
            sensor = WristState(decoder)
            for q, _, _, ellipse, state in sequence:
                sensor.read(ellipse, q)
                xx.append(sensor.motion_input)
                yy.append(state[3:6])
        x, y = np.asarray(xx), np.asarray(yy)
        self.center = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), np.r_[[.03] * 3, [.15] * 9])
        self.quadratic = quadratic
        values = self.design(x)
        self.weights = np.linalg.solve(values.T @ values + ridge * np.eye(values.shape[1]), values.T @ y)

    def design(self, x):
        value = np.clip((np.atleast_2d(x) - self.center) / self.scale, -6., 6.)
        forms = (np.ones(len(value)), value)
        if self.quadratic:
            forms += (_power(value, 2),)
        return np.column_stack(forms)

    def predict(self, x):
        return self.design(x)[0] @ self.weights


def _power(x, degree):
    terms = list(itertools.combinations_with_replacement(range(x.shape[1]), degree))
    return np.prod(x[:, np.asarray(terms)], axis=2)


class DirectPupil:
    """A smooth direct fit to demonstrated joint targets with no movement plan."""

    def __init__(self, x, y, ridge=.1, family='separable', width=.65, pair=None):
        self.family = family
        self.width = width
        self.pair = (2, 5) if pair is None else tuple(pair)
        self.context = tuple(idx for idx in range(6) if idx not in self.pair)
        if pair is None:
            self.center = np.r_[np.zeros(11), x[:, 11:16].mean(axis=0), np.zeros(5)]
            self.scale = np.r_[[.045, .045, .15, .35, .35, 1.25], [.04, .04, .04, .13, .13],
                               [.1] * 5, [.13] * 5]
            options = (np.linspace(-.10, 2., 6), np.linspace(-2., 1.8, 8))
        else:
            self.center = x.mean(axis=0)
            floor = np.r_[[.04] * 3, [.28] * 3, [.035] * 3, [.075] * 2, [.10] * 10]
            self.scale = np.maximum(1.55 * x.std(axis=0), floor)
            scaled_pair = ((x - self.center) / self.scale)[:, self.pair]
            bounds = np.quantile(scaled_pair, (.015, .985), axis=0)
            options = tuple(np.linspace(start - .18, end + .18, 7) for start, end in bounds.T)
        self.centers = np.array(list(itertools.product(*options)))
        if self.family == 'fourier':
            rng = np.random.default_rng(641993)
            self.projection = rng.normal(size=(x.shape[1], 256)) * rng.uniform(.6, 2.4, 256) / np.sqrt(x.shape[1])
            self.phase = rng.uniform(-np.pi, np.pi, 256)
        design = self.design(x)
        self.weights = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ y)

    def design(self, x):
        x = np.clip((np.atleast_2d(x) - self.center) / self.scale, -4., 4.)
        bias = np.ones(len(x))
        if self.family == 'fourier':
            return np.column_stack((bias, x, np.cos(x @ self.projection + self.phase)))
        coordinates = x[:, self.pair]
        distance = coordinates[:, None, :] - self.centers[None, :, :]
        radial = np.exp(-.5 * np.sum(distance**2, axis=2) / self.width**2)
        context = x[:, self.context]
        interaction = radial[:, :, None] * context[:, None, :]
        joints = x[:, 11:16]
        pair = context[:, :, None] * joints[:, None, :]
        return np.column_stack((bias, x, radial, interaction.reshape(len(x), -1), pair.reshape(len(x), -1)))

    def target(self, x):
        return self.design(x)[0] @ self.weights

    def command(self, x, issued):
        return np.clip((self.target(x) - issued) / _INC, -.70, .70)


class WristPupil:
    """Deploy the fitted sequence with access only to public observations."""

    def __init__(self, actor, decoder, motion):
        self.actor = actor
        self.sensor = WristState(decoder, motion)
        self.home = None
        self.issued = np.zeros(5)

    def command(self, obs):
        if self.home is None:
            self.home = obs['joints'].copy()
        x = _features(self.sensor.observe(obs), obs['joints'], self.home)
        action = self.actor.command(x, self.issued)
        self.issued += _INC * action
        return action

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as checkpoint:
            actor = _restore(DirectPupil, checkpoint, 'actor_')
            actor.pair, actor.context = tuple(actor.pair), tuple(actor.context)
            decoder = _restore(WristDecoder, checkpoint, 'decoder_')
            motion = _restore(MotionFit, checkpoint, 'motion_')
        return cls(actor, decoder, motion)


def _restore(agent_type, archive, prefix):
    agent = agent_type.__new__(agent_type)
    items = {key.removeprefix(prefix): archive[key] for key in archive.files if key.startswith(prefix)}
    agent.__dict__.update({key: value.item() if value.ndim == 0 else value for key, value in items.items()})
    return agent


def _features(state, joints, home):
    return np.r_[state, home, joints - home]


def wrist_features(decoder, motion, records):
    result = []
    for sequence in records:
        sensor = WristState(decoder, motion)
        home = sequence[0][0]
        result.append(np.array([_features(sensor.read(ellipse, q), q, home)
                                for q, _, _, ellipse, _ in sequence]))
    return result


def episode(arena, seed, *, actor=None, teacher_fraction=0., collect=False, sensor_factory=CleanState):
    obs = arena.reset(seed=seed, split='held')
    home = obs['joints'].copy()
    sensor = sensor_factory(arena)
    initial_control = arena._control[:arena.action_size].copy() if collect else None
    teacher = PrivilegedCeiling(arena) if collect else None
    issued = np.zeros(arena.action_size)
    features, targets = [], []
    trace = []
    for _ in range(arena.max_steps):
        state = sensor.observe(obs)
        x = _features(state, obs['joints'], home)
        action = actor.command(x, issued) if actor else np.zeros(arena.action_size)
        if collect:
            teacher_action = teacher.command(trace=True)
            desired = _INC * np.asarray(teacher.track['normalized_requested_rates'])
            desired += arena._control[:arena.action_size] - initial_control
            features.append(x)
            targets.append(desired)
            action = teacher_fraction * teacher_action + (1. - teacher_fraction) * action
        obs, _, done = arena.step(action)
        issued += _INC * action
        trace.append(action.tolist())
        if done:
            break
    result = arena.judge_metrics()
    result['commands'] = trace
    return result, (np.asarray(features), np.asarray(targets))


def pack(pairs):
    return tuple(np.concatenate(columns, axis=0) for columns in zip(*pairs, strict=True))


def select_pair(pairs, *, ridge=.1, width=.65):
    scores = []
    for pair in itertools.combinations(range(6), 2):
        errors = []
        for hold, (vx, vy) in enumerate(pairs):
            x, y = pack([group for idx, group in enumerate(pairs) if idx != hold])
            actor = DirectPupil(x, y, ridge=ridge, width=width, pair=pair)
            errors.append(float(np.mean((actor.design(vx) @ actor.weights - vy)**2)))
        scores.append((float(np.mean(errors)), pair))
    return min(scores)[1], sorted(scores)


def summarize(rows):
    run = np.array([row['longest_consecutive_rebounds'] for row in rows])
    return dict(counts=run.tolist(), mean=float(run.mean()), median=float(np.median(run)),
                maximum=int(run.max()), at_least_five=int(np.count_nonzero(run >= 5)),
                at_least_ten=int(np.count_nonzero(run >= 10)))


def _training_demos(wrist):
    records, pairs, rows = [], [], []
    for seed in DEMOS:
        sequence = []
        result, data = episode(wrist, seed, teacher_fraction=1., collect=True,
                               sensor_factory=lambda arena: CalibrationState(arena, sequence))
        rows.append(result)
        pairs.append(data)
        records.append(sequence)
    return rows, pairs, records


def _lens_error(wrist, decoder, motion):
    errors = {key: [] for key in ('finite_differences', 'fitted_motion')}
    for seed in DEVELOPMENT[:2]:
        sequence = []
        episode(wrist, seed, teacher_fraction=1., collect=True, sensor_factory=lambda arena: CalibrationState(arena, sequence))
        for key, lookup in (('finite_differences', None), ('fitted_motion', motion)):
            sensor = WristState(decoder, lookup)
            per_step = [sensor.read(ellipse, q) - value for q, _, _, ellipse, value in sequence]
            errors[key].extend(per_step[3:])
    return {key: dict(rms=np.sqrt(np.mean(np.square(values), axis=0)).tolist(), samples=len(values))
            for key, values in errors.items()}


def _save_pupil(path, actor, decoder, motion):
    weights = {}
    for prefix, obj in (('actor_', actor), ('decoder_', decoder), ('motion_', motion)):
        weights.update({f'{prefix}{key}': val for key, val in vars(obj).items()})
    np.savez_compressed(path, **weights)


def run_deployed(arena, seed, checkpoint):
    policy = WristPupil.load(checkpoint)
    obs = arena.reset(seed=seed, split='held')
    commands = []
    for _ in range(arena.max_steps):
        command = policy.command(obs)
        commands.append(command.tolist())
        obs, _, done = arena.step(command)
        if done:
            break
    result = arena.judge_metrics()
    result['commands'] = commands
    return result


def _compare_resolution(folder):
    checkpoint = folder / 'bounce-ident-pupil.npz'
    with gzip.open(folder / 'bounce-ident.json.gz', 'rt', encoding='utf-8') as file:
        reference = json.load(file)['cohorts']['wrist_fitted_motion']['summary']
    wrist = BounceArena(view='wrist', width=384, height=288)
    try:
        rows = [run_deployed(wrist, seed, checkpoint) for seed in FRESH]
        evidence = dict(privileged_training_supervision=True, test_time='wrist RGB and own encoders',
                        held_seeds=FRESH, baseline_dimensions=[192, 144], baseline=reference,
                        doubled_dimensions=[wrist.width, wrist.height], doubled=summarize(rows), episodes=rows)
        with gzip.open(folder / 'bounce-ident-resolution.json.gz', 'wt', encoding='utf-8') as output:
            json.dump(evidence, output, separators=(',', ':'))
        print('doubled resolution', evidence['doubled'], flush=True)
    finally:
        wrist.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24'))
    parser.add_argument('--compare-resolution', action='store_true', help='evaluate the saved wrist pupil at 2× resolution')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.compare_resolution:
        _compare_resolution(args.out)
        return
    wrist = BounceArena(view='wrist', width=192, height=144)
    clean = BounceArena(view='none', width=192, height=144)
    try:
        demos, pairs, records = _training_demos(wrist)
        x, y = pack(pairs)
        pair, ranks = select_pair(pairs)
        generic = DirectPupil(x, y, pair=pair)
        supplied_pair = DirectPupil(x, y)
        fourier = DirectPupil(x, y, family='fourier', ridge=.4)
        focal = wrist.height / (2 * math.tan(math.radians(wrist._model.camera(WRIST_CAMERA).fovy[0]) / 2))
        decoder = WristDecoder(records, focal)
        motion = MotionFit(decoder, records, quadratic=False)
        lens_error = _lens_error(wrist, decoder, motion)
        modes = (
            ('stationary', clean, None, CleanState),
            ('clean_generic', clean, generic, CleanState),
            ('clean_differenced_motion', clean, generic, CleanDifference),
            ('clean_supplied_axes', clean, supplied_pair, CleanState),
            ('clean_fourier', clean, fourier, CleanState),
            ('wrist_direct_differences', wrist, generic, lambda _: WristState(decoder)),
            ('wrist_fitted_motion', wrist, generic, lambda _: WristState(decoder, motion)),
            ('wrist_fitted_motion_blind', clean, generic, lambda _: WristState(decoder, motion)),
        )
        cohorts = {}
        for name, arena, actor, sensor in modes:
            rows = [episode(arena, seed, actor=actor, sensor_factory=sensor)[0] for seed in FRESH]
            cohorts[name] = dict(summary=summarize(rows), episodes=rows)
            print(name, cohorts[name]['summary'], flush=True)
        report = dict(teacher='privileged MuJoCo analytic controller, four motor-labeled demonstrations',
                      demo_seeds=DEMOS, held_seeds=FRESH, demos=summarize(demos), width=wrist.width, height=wrist.height,
                      pair=pair, held_demo_pair_errors=ranks, held_perception_errors=lens_error,
                      perception_training='privileged training supervision of camera geometry and scene state from the same four demonstrations',
                      sighted_runtime='wrist RGB and own joint angles; fixed fitted calibration',
                      cohorts=cohorts)
        with gzip.open(args.out / 'bounce-ident.json.gz', 'wt', encoding='utf-8') as output:
            json.dump(report, output, separators=(',', ':'))
        _save_pupil(args.out / 'bounce-ident-pupil.npz', generic, decoder, motion)
    finally:
        wrist.close()
        clean.close()


if __name__ == '__main__':
    main()
