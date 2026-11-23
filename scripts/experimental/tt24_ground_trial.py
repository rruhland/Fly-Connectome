"""Ground a planar-earned controller in measured wrist-paddle motor effects."""

import argparse
from collections import deque
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import MotionFit, WristDecoder, WristState
from .tt24_bounce_trial import _write, seeds, verify_replay
from .tt24_motor import RewardSearch
from .tt24_planar_learning import RATE_VALUES, sensory_basis
from .tt24_source import CHECKPOINT as NATIVE, VisualConsequence


ROOT = Path('artifacts/tt24')
VISION = ROOT / 'bounce-ident-pupil.npz'
SOURCE = ROOT / 'planar-powered.json.gz'
OUTPUT = ROOT / 'ground-planar.json.gz'
STRIDE = 4
CALIBRATION = 4
SPAN = .076
MAX_SPEED = np.array((.20, .20, .50))
VELOCITY_SCALE = np.array((.26, .26, .45, .13, .13))
POSE_SCALE = np.array((.035, .035, .040, .009, .009))
OFFSET = np.array((.016, .016, .027))
MODES = ('powered', 'cold', 'shuffled')


def load_senses(path):
    """Read only frozen wrist perception parameters, leaving motor labels unopened."""
    specifications = (
        (WristDecoder, 'decoder_', ('focal', 'default', 'center', 'scale', 'body_weights', 'radius', 'camera_weights')),
        (MotionFit, 'motion_', ('center', 'scale', 'quadratic', 'weights')),
    )
    fits = []
    with np.load(path, allow_pickle=False) as checkpoint:
        for typ, prefix, keys in specifications:
            fit = typ.__new__(typ)
            for key in keys:
                value = checkpoint[prefix + key]
                setattr(fit, key, value.item() if value.ndim == 0 else value)
            fits.append(fit)
    return tuple(fits)


def paddle_pose(state):
    return state[6:11] * np.r_[np.ones(3), SPAN, SPAN]


class Effects:
    """Fit short joint-command blocks to actual latent paddle motion."""

    def __init__(self, observations):
        rows = np.vstack(observations)
        action, speed, effect = rows[:, :5], rows[:, 5:10], rows[:, 10:]
        design = np.column_stack((action, speed))
        regularization = np.diag(np.r_[np.full(5, .12), np.full(5, .40)])
        self.map = np.linalg.solve(design.T @ design + regularization, design.T @ effect)
        response = self.map[:5]
        self.inverse = response.T @ np.linalg.inv(response @ response.T + .05 * np.eye(5))
        self.error = float(np.sqrt(np.mean((design @ self.map - effect) ** 2)))
        self.singular = np.linalg.svd(response, compute_uv=False)
        self.samples = len(design)

    def command(self, desired, measured):
        coast = measured @ self.map[5:]
        command = (desired / VELOCITY_SCALE - coast) @ self.inverse
        return np.clip(command, -.43, .43)


class PoseEffects:
    """Fit small realized encoder offsets to simultaneous decoded paddle offsets."""

    def __init__(self, observations):
        rows = np.vstack(observations)
        angles, pose = rows[:, :5], rows[:, 5:]
        weight = 1. / (1. + .25 * np.sum(pose ** 2, axis=1))
        design = angles.T * weight
        self.map = np.linalg.solve(design @ angles + .0004 * np.eye(5), design @ pose)
        self.inverse = self.map.T @ np.linalg.inv(self.map @ self.map.T + .08 * np.eye(5))
        self.error = float(np.sqrt(np.mean((angles @ self.map - pose) ** 2)))
        self.singular = np.linalg.svd(self.map, compute_uv=False)
        self.samples = len(angles)

    def command(self, desired, measured):
        step = (desired / POSE_SCALE) @ self.inverse
        return np.clip(6. * step - .038 * measured, -.24, .24)


def calibrate(arena, senses, launches, seed, *, pose=False):
    """Collect complete rewarded drops driven by task-independent rate babble."""
    rng = np.random.default_rng(seed)
    all_rows, evidence = [], []
    for launch in launches:
        obs = arena.reset(seed=launch, split='practice')
        sensor = WristState(*senses)
        origin = obs['joints'].copy()
        state = sensor.observe(obs)
        home = paddle_pose(state)
        previous = np.zeros(5)
        points = []
        trial_reward = 0.
        done = False
        while not done:
            old = paddle_pose(state)
            limits = (.055, .115) if pose else (.11, .27)
            drive = rng.choice((-1., 1.), 5) * rng.uniform(*limits)
            action = np.clip(drive + 1.5 * (origin - obs['joints']), -.33, .33)
            for tick in range(STRIDE):
                obs, reward, done = arena.step(action)
                trial_reward += reward
                state = sensor.observe(obs)
                if done:
                    break
            if tick == STRIDE - 1:
                if pose:
                    points.append(np.r_[obs['joints'] - origin, (paddle_pose(state) - home) / POSE_SCALE])
                else:
                    later = (paddle_pose(state) - old) / (arena.dt * STRIDE) / VELOCITY_SCALE
                    points.append(np.r_[action, previous, later])
                    previous = later
        all_rows.append(np.asarray(points))
        evidence.append(arena.judge_metrics() | dict(observed_reward=trial_reward, effect_samples=len(points)))
    return (PoseEffects(all_rows) if pose else Effects(all_rows)), evidence


class Portable:
    """Combine two projections of a planar policy in a gravity-aligned visual frame."""

    def __init__(self, weights, prior):
        self.weights = weights
        self.prior = prior
        self.paths = deque(maxlen=5)

    def track(self, state):
        self.paths.append((state[6:9].copy(), state[:3].copy()))

    def command(self, dt, gain):
        if len(self.paths) < 4 or gain <= 0:
            return np.zeros(3)
        arm, ball = (np.asarray(track) / SPAN for track in zip(*self.paths, strict=True))
        candidate = np.array(list(np.ndindex(3, 3)))
        directions = RATE_VALUES[candidate]
        held = []
        for dim in (0, 1):
            own = arm[:, (dim, 2)] * (1., -1.)
            mover = ball[:, (dim, 2)] * (1., -1.)
            feat = sensory_basis(mover[-1] - own[-1], own[-1] - own[0], mover[-1] - mover[0], np.tanh(own[-1] / 3.))
            logits = self.weights @ feat
            if self.prior is not None:
                own_move = directions * MAX_SPEED[[dim, 2]] * (1., -1.) * (dt / SPAN)
                arm_path = np.concatenate((np.broadcast_to(own[-4:], (len(candidate), 4, 2)),
                                           (own[-1] + own_move)[:, None, :]), axis=1)
                # One-step local flow extrapolation from actual decoded samples.
                ball_step = mover[-1] + (mover[-1] - mover[-3]) / 2.
                ball_path = np.concatenate((mover[-4:], ball_step[None, :]), axis=0)
                score = self.prior.score_history(arm_path, np.broadcast_to(ball_path, arm_path.shape), dt=dt, scale=1.)
                for axis in range(len(logits)):
                    logits[axis] += 6. * np.array([np.mean(score[candidate[:, axis] == j]) - score.mean() for j in range(3)])
            probability = np.exp(2.4 * (logits - logits.max(axis=1, keepdims=True)))
            rates = (probability @ RATE_VALUES) / probability.sum(axis=1)
            held.append(rates)
        return gain * np.array((held[0][0], held[1][0], .5 * (held[0][1] + held[1][1])))


def carrier(path):
    with gzip.open(path, 'rt') as stream:
        source = json.load(stream)
    selected = [entry['stages']['damped']['modes'] for entry in source['cohorts']]
    means = {mode: np.mean([entry[mode]['actor'] for entry in selected], axis=0) for mode in ('pong', 'shuffled')}
    return {'powered': (means['pong'], VisualConsequence.load('pong')),
            'shuffled': (means['shuffled'], VisualConsequence.load('shuffled')),
            'cold': (np.zeros_like(means['pong']), None)}


def play(arena, launch, senses, body, donor, readout, gain, *, practice=False, hold=False, blind=False, record=False):
    obs = arena.reset(seed=launch, split='practice' if practice else 'held', render=not (hold or blind))
    sensor = WristState(*senses)
    visual = Portable(*donor)
    original = obs['joints'].copy()
    latest = sensor.observe(obs)
    home = paddle_pose(latest)
    repeat = np.zeros(5)
    actions, earned = [], 0.
    finished = False
    while not finished:
        before = paddle_pose(latest)
        if hold:
            action = np.zeros(5)
        else:
            if blind:
                feature = np.r_[1., np.zeros(9)]
                carried = np.zeros(3)
            else:
                feature = np.r_[1., np.tanh((latest[:3] - latest[6:9]) / .13), np.tanh(latest[3:6] / 2.8),
                                np.tanh(latest[6:9] / .055)]
                carried = visual.command(arena.dt, gain)
            target = before - home
            motor = carried + np.tanh(readout @ feature)
            if isinstance(body, PoseEffects):
                request = np.r_[OFFSET * motor, 0., 0.]
                action = body.command(request - target, obs['joint_velocity']) if np.any(request) else np.zeros(5)
            else:
                translation = motor * MAX_SPEED - 3.6 * target[:3]
                stabilization = -9. * target[3:]
                action = body.command(np.r_[translation, stabilization], repeat)
                own = obs['joints'] - original
                action = np.clip(action - .72 * own, -.43, .43)
        for frame in range(STRIDE):
            obs, reward, finished = arena.step(action, render=not (hold or blind))
            earned += reward
            latest = sensor.observe(obs)
            if not blind and not hold:
                visual.track(latest)
            if record:
                actions.append(action.tolist())
            if finished:
                break
        if frame == STRIDE - 1:
            repeat = (paddle_pose(latest) - before) / (STRIDE * arena.dt) / VELOCITY_SCALE
    result = arena.judge_metrics() | dict(score=earned)
    if record:
        result['actions'] = actions
    return result


def summary(rows):
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    runs = [row['longest_consecutive_rebounds'] for row in rows]
    return dict(rebounds=mean('rebounds'), consecutive=mean('longest_consecutive_rebounds'), highest=max(runs),
                longest=runs, elapsed=mean('elapsed_s'), counts=[row['rebounds'] for row in rows])


def train(arena, launches, senses, body, donor, *, seed, gain):
    learner = RewardSearch(3, seed, radius=.36, features=10)
    options, receipt = [], []
    for k in range(0, len(launches), 2):
        weights = learner.propose()
        # The same bounded scale exploration is charged to each treatment.
        scale = gain * (.66, 1., 1.35)[(k // 2) % 3]
        if isinstance(body, PoseEffects):
            if k <= 2:
                weights = np.zeros_like(weights)
            scale = gain * (0., 1., .55, 1.45)[(k // 2) % 4]
        rows = [play(arena, case, senses, body, donor, weights, scale, practice=True) for case in launches[k:k + 2]]
        utility = float(np.mean([row['score'] + min(.20, .05 * row['elapsed_s']) for row in rows]))
        options.append((utility, weights.copy(), scale))
        learner.update(weights, utility)
        receipt += [row | dict(proposed=weights.tolist(), drive_scale=scale) for row in rows]
    _, weights, scale = max(options, key=lambda row: row[0])
    return weights, scale, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=51178398)
    parser.add_argument('--practice', type=int, default=12)
    parser.add_argument('--held', type=int, default=12)
    parser.add_argument('--resolution', type=int, choices=(96, 192), default=192)
    parser.add_argument('--gain', type=float, default=1.)
    parser.add_argument('--mapping', choices=('velocity', 'pose'), default='velocity')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.practice not in (4, 8, 12) or args.held < 1 or args.gain < 0:
        parser.error('use 4, 8, or 12 practice drops in addition to 4 body probes')
    arena = BounceArena(view='wrist', width=args.resolution, height=3 * args.resolution // 4)
    sight = load_senses(VISION)
    donors = carrier(SOURCE)
    training = seeds(args.seed + 1, 2) * (args.practice // 2)
    calibration = seeds(args.seed + 2, CALIBRATION)
    held = seeds(args.seed + 3, args.held)
    if set(held) & set(training + calibration):
        raise ValueError('practice and held launches overlap')
    try:
        body, trace = calibrate(arena, sight, calibration, args.seed, pose=args.mapping == 'pose')
        report = dict(schema='tt24-grounded-planar-1', seed=args.seed, resolution=args.resolution,
                      mapping=args.mapping, target_drop_budget=CALIBRATION + args.practice,
                      delivered=dict(vision=str(VISION), vision_sha256=sha256(VISION.read_bytes()).hexdigest(),
                                     planar=str(SOURCE), planar_sha256=sha256(SOURCE.read_bytes()).hexdigest(),
                                     pong_sha256=sha256(NATIVE.read_bytes()).hexdigest()),
                      prior_training='privileged calibration of the frozen wrist decoder and motion map',
                      practice_seeds=training, held_seeds=held, calibration=trace, effect_map=body.map.tolist(),
                      fit_samples=body.samples, fit_error=body.error, singular_values=body.singular.tolist(), modes={})
        blank = np.zeros((3, 10))
        passive = [play(arena, case, sight, body, donors['cold'], blank, 0., hold=True, record=True) for case in held]
        report['hold'] = dict(held=passive, summary=summary(passive))
        print('stationary', report['hold']['summary'], 'effect singular', body.singular.round(3), flush=True)
        for mode in MODES:
            donor = donors[mode]
            weights, scale, history = train(arena, training, sight, body, donor, seed=args.seed + 4, gain=args.gain)
            results = {name: [play(arena, case, sight, body, donor, weights, scale, blind=blind, record=True) for case in held]
                       for name, blind in (('live', False), ('blind', True))}
            if mode == 'powered':
                results['source_removed'] = [play(arena, case, sight, body, donors['cold'], weights, scale, record=True)
                                             for case in held]
            scores = {name: summary(rows) for name, rows in results.items()}
            for name, rows in results.items():
                winner = max(rows, key=lambda row: row['longest_consecutive_rebounds'])
                verify_replay(arena, winner)
            report['modes'][mode] = dict(weights=weights.tolist(), scale=scale, training=history, held=results, summary=scores)
            print(mode, [row['rebounds'] for row in history], scores, flush=True)
            _write(args.output, report)
    finally:
        arena.close()


if __name__ == '__main__':
    main()
