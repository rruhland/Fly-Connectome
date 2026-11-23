"""Learn bounce control by watching wrist and encoder transitions without motor labels."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ceiling import PrivilegedCeiling
from .tt24_motor import optical_sightings


DATA = Path('artifacts/tt24/predict-witness-data.npz')
OUTPUT = Path('artifacts/tt24/predict-witness.json.gz')
RATES_OUTPUT = Path('artifacts/tt24/predict-witness-rates.json.gz')
BODY_SEEDS = tuple(range(246001, 246009))
DEMO_SEEDS = tuple(range(245001, 245005))
HELD_SEEDS = tuple(range(247001, 247013))
RATES_SEEDS = tuple(range(248001, 248013))
STEPS = 360


class BodyPlay(BounceArena):
    """Independent body experience using the same arm with ball contacts disabled."""

    def __init__(self):
        super().__init__(view='none', max_steps=240)
        self._model.geom_contype[self._ball_geom] = 0
        self._model.geom_conaffinity[self._ball_geom] = 0

    def _judge_substep(self, incoming):
        return 0.


def sight(obs):
    ball, paddle = optical_sightings(obs.get('rgb'))
    if ball is None or paddle is None:
        return np.full(5, np.nan)
    return np.r_[ball.center - paddle.center, np.log(ball.radius), paddle.center]


def encoders(obs):
    return np.r_[obs['joints'], obs['joint_velocity']]


def body_record(arena, seed):
    rng = np.random.default_rng(seed + 25314)
    obs = arena.reset(seed=seed, split='held', render=False)
    sensed = [encoders(obs)]
    action_history = []
    hold = obs['joints'].copy()
    noise = np.zeros(5)
    for tick in range(arena.max_steps):
        amplitude = (.075, .20, .031, .105)[(tick // 27) % 4]
        noise = .63 * noise + rng.normal(0., amplitude, 5)
        action = np.clip(noise - 1.4 * (obs['joints'] - hold), -.32, .32)
        obs, _, _ = arena.step(action, render=False)
        sensed.append(encoders(obs))
        action_history.append(action)
    return np.asarray(sensed), np.asarray(action_history)


def watch(arena, seed):
    obs = arena.reset(seed=seed, split='held')
    teacher = PrivilegedCeiling(arena)
    vision, sensed = [sight(obs)], [encoders(obs)]
    done = False
    while not done:
        obs, _, done = arena.step(teacher.command())
        vision.append(sight(obs))
        sensed.append(encoders(obs))
    evidence = arena.judge_metrics()
    return np.c_[vision, sensed], evidence['longest_consecutive_rebounds']


def collect(path):
    body = BodyPlay()
    arena = BounceArena(view='wrist', max_steps=STEPS)
    try:
        archive = {}
        for seed in BODY_SEEDS:
            sensed, action = body_record(body, seed)
            archive[f'body_obs_{seed}'] = sensed
            archive[f'body_action_{seed}'] = action
        for seed in DEMO_SEEDS:
            observed, longest = watch(arena, seed)
            archive[f'witness_{seed}'] = observed
            archive[f'witness_score_{seed}'] = np.array([longest])
            print('witness', seed, 'rebound sequence', longest, 'frames', len(observed), flush=True)
        np.savez_compressed(path, **archive)
    finally:
        body.close()
        arena.close()


def centered_temporal_window(state, index):
    index = np.asarray(index)
    taps = np.clip(index[:, None] + np.array((-4, -2, -1, 0, 1, 2, 3, 5)), 0, len(state) - 1)
    sampled = state[taps].copy()
    sampled[:, :, :5] -= state[0, :5]
    sampled[:, :, 5:] *= .16
    return np.c_[state[index, :5], sampled.reshape(len(index), -1)]


class BodyReadout:
    """Estimate accumulated motor intent from independent joint-response examples."""

    def __init__(self, pairs, *, shuffle=False):
        designs, goals = [], []
        rng = np.random.default_rng(654256)
        for sensed, action in pairs:
            index = np.arange(len(action))
            total = np.cumsum(action, axis=0)
            if shuffle:
                total = np.roll(total, rng.integers(25, len(total) - 25), axis=0)
            designs.append(centered_temporal_window(sensed, index))
            goals.append(total)
        motion = np.vstack(designs)
        self.center = motion.mean(axis=0)
        self.scale = np.maximum(motion.std(axis=0), .016)
        body = self.design(motion)
        reg = .9 * np.eye(body.shape[1])
        reg[0, 0] = 0.
        self.readout = np.linalg.solve(body.T @ body + reg, body.T @ np.vstack(goals))

    def design(self, x):
        x = np.clip((x - self.center) / self.scale, -6., 6.)
        return np.c_[np.ones(len(x)), x]

    def predict(self, sensed):
        x = centered_temporal_window(sensed, np.arange(len(sensed) - 1))
        return self.design(x) @ self.readout


class SensorHistory:
    """A short live image window plus current joint position and velocity."""

    def __init__(self, blind=False):
        self.history = []
        self.recent = None
        self.home = None
        self.blind = blind

    def update(self, rgb, sensed):
        if self.home is None:
            self.home = sensed[:5].copy()
        observed = bool(np.isfinite(rgb).all() and not self.blind)
        if observed:
            self.recent = rgb.copy()
        vision = np.zeros(5) if self.recent is None else self.recent
        self.history.append(vision)
        self.history = self.history[-8:]
        before = self.history[max(0, len(self.history) - 4)]
        long = self.history[0]
        fast = (vision - before) / max(1, min(3, len(self.history) - 1))
        slow = (vision - long) / max(1, len(self.history) - 1)
        return np.r_[vision, fast[:3], slow[:3], float(observed),
                     sensed[:5] - self.home, sensed[5:], self.home]


def frames(observed, *, blind=False):
    memory = SensorHistory(blind=blind)
    return np.asarray([memory.update(row[:5], row[5:]) for row in observed[:-1]])


class Pupil:
    """Identify local continuous motor goals from instantaneous sensory states."""

    def __init__(self, states, goals, *, neighbors=32, ridge=.14, visual_gain=1.):
        from scipy.spatial import cKDTree

        self.center = states.mean(axis=0)
        self.scale = np.maximum(states.std(axis=0), .015)
        self.scale[12:17] *= 2.8
        self.scale[17:22] *= 3.3
        self.scale[22:] *= 5.
        self.scale[12:] *= visual_gain
        self.state = (states - self.center) / self.scale
        self.tree = cKDTree(self.state)
        self.target = goals
        self.neighbors = min(len(states), neighbors)
        self.ridge = ridge

    def predict(self, state):
        normalized = (state - self.center) / self.scale
        distances, idx = self.tree.query(normalized, k=self.neighbors)
        delta = self.state[idx] - normalized
        weight = np.exp(-.5 * ((distances - distances[0]) / max(.30, .45 * distances[-1]))**2)
        design = np.c_[np.ones(len(idx)), delta]
        weighted = design * weight[:, None]
        reg = np.eye(design.shape[1]) * self.ridge
        reg[0, 0] = 0.
        update = np.linalg.solve(design.T @ weighted + reg, weighted.T @ self.target[idx])[0]
        return np.clip(update, -3.6, 3.6)


def configure(body, examples, mode):
    blind = mode == 'blind'
    states = np.vstack([frames(observed, blind=blind) for observed in examples])
    if mode == 'scratch':
        inferred = []
        for observed in examples:
            future = np.minimum(np.arange(len(observed) - 1) + 3, len(observed) - 1)
            inferred.append(observed[future, 5:10] - observed[0, 5:10])
    else:
        decoder = BodyReadout(body, shuffle=mode == 'shuffled')
        inferred = [decoder.predict(observed[:, 5:]) for observed in examples]
    if mode in ('shifted', 'rate_shifted'):
        inferred = [np.roll(target, 13 + 2 * i, axis=0) for i, target in enumerate(inferred)]
    if mode in ('rates', 'rate_shifted'):
        from scipy.signal import savgol_filter

        smoothed = [savgol_filter(target, 7, 2, axis=0) for target in inferred]
        targets = np.vstack([np.c_[target, np.diff(target, axis=0, prepend=target[:1])]
                             for target in smoothed])
        return Pupil(states, targets, neighbors=44, ridge=.28, visual_gain=1.8)
    targets = np.vstack(inferred)
    return Pupil(states, targets)


def play(arena, seed, pupil, *, blind=False, scratch=False):
    obs = arena.reset(seed=seed, split='held', render=not blind and pupil is not None)
    memory = SensorHistory(blind=blind)
    issued = np.zeros(5)
    actions = []
    done = False
    while not done:
        if pupil is None:
            action = np.zeros(5)
        else:
            state = memory.update(sight(obs), encoders(obs))
            goal = pupil.predict(state)
            if scratch:
                measured = obs['joints'] - memory.home
                action = np.clip(2.8 * (goal - measured) - .055 * obs['joint_velocity'], -.54, .54)
            elif len(goal) == 10:
                target, rate = np.split(goal, 2)
                action = np.clip(.44 * (target - issued) + .62 * rate, -.54, .54)
            else:
                action = np.clip(.72 * (goal - issued), -.54, .54)
        issued += action
        actions.append(action.tolist())
        obs, _, done = arena.step(action, render=not blind and pupil is not None)
    return arena.judge_metrics() | dict(actions=actions)


def body_validation(body):
    results = []
    for test in range(0, len(body), 2):
        train = [pair for i, pair in enumerate(body) if i != test]
        normal, shuffled = BodyReadout(train), BodyReadout(train, shuffle=True)
        y = np.cumsum(body[test][1], axis=0)
        results.append(dict(actual_rmse=float(np.mean((y - normal.predict(body[test][0]))**2)**.5),
                            shuffled_rmse=float(np.mean((y - shuffled.predict(body[test][0]))**2)**.5),
                            null_rmse=float(np.mean(y**2)**.5)))
    return results


def qualify(data, output, modes, seeds):
    with np.load(data) as archive:
        body = [(archive[f'body_obs_{seed}'], archive[f'body_action_{seed}']) for seed in BODY_SEEDS]
        examples = [archive[f'witness_{seed}'] for seed in DEMO_SEEDS]
        demo_runs = {seed: archive[f'witness_score_{seed}'][0].item() for seed in DEMO_SEEDS}
    tests = body_validation(body)
    print('body playback target validation', tests, flush=True)
    results = {}
    arena = BounceArena(view='wrist', max_steps=STEPS)
    try:
        for mode in modes:
            pupil = None if mode == 'hold' else configure(body, examples, mode)
            runs = []
            for seed in seeds:
                row = play(arena, seed, pupil, blind=mode == 'blind', scratch=mode == 'scratch')
                print('observer held', mode, seed, row['rebounds'], row['longest_consecutive_rebounds'], flush=True)
                runs.append(row)
            results[mode] = runs
    finally:
        arena.close()
    with gzip.open(output, 'wt') as stream:
        json.dump(dict(schema='tt24-observational-bounce-1', body_validation=tests,
                       demo_rebounds=demo_runs, held=results, independent_body_steps=len(body) * 240),
                  stream, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'qualify', 'rates'))
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'collect':
        collect(args.data)
    elif args.stage == 'rates':
        modes = ('hold', 'body', 'rates', 'rate_shifted')
        qualify(args.data, args.output or RATES_OUTPUT, modes, RATES_SEEDS)
    else:
        modes = ('hold', 'body', 'scratch', 'shuffled', 'shifted', 'blind')
        qualify(args.data, args.output or OUTPUT, modes, HELD_SEEDS)


if __name__ == '__main__':
    main()
