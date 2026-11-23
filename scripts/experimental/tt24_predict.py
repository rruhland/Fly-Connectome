"""Identify wrist-visual action effects and plan new five-channel commands."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_motor import optical_sightings
from .tt24_source import VisualConsequence


PATH = Path('artifacts/tt24/predict-data.json.gz')
RESULT = Path('artifacts/tt24/predict-result.json.gz')
PRACTICE_SEED = 85241
HOLD_SEED = 85242
BLOCK = 3


class Senses:
    """Retain appearance, motion, elapsed visibility and own actuator history."""

    def __init__(self):
        self.home = None
        self.last = None
        self.motion = np.zeros(3)
        self.previous = np.zeros(5)

    def read(self, obs, *, blind=False):
        q = np.asarray(obs['joints'])
        if self.home is None:
            self.home = q.copy()
        ball, paddle = optical_sightings(None if blind else obs.get('rgb'))
        if ball is not None:
            visual = np.r_[5. * ball.center, 2. * np.log(ball.radius / .08)]
            if self.last is not None:
                self.motion = .48 * self.motion + .52 * (visual - self.last)
            self.last = visual
        elif self.last is not None:
            self.last = np.clip(self.last + self.motion, -3., 3.)
            self.motion *= .5
        track = np.array((0., 0., .18)) if paddle is None else np.r_[paddle.center, paddle.radius]
        return np.r_[self.last if self.last is not None else np.zeros(3), self.motion * 5.,
                     float(ball is not None), (q - self.home) * 5., obs['joint_velocity'] / 3.,
                     self.previous, track]


def seeds(seed, n):
    return np.random.default_rng(seed).integers(1, 2 ** 31 - 1, n).tolist()


def practice(arena, launch, index, rng):
    sense = Senses()
    obs = arena.reset(seed=launch)
    state = sense.read(obs)
    trace = []
    exploration = np.zeros(5)
    done = False
    while not done:
        if index < 8:
            action = np.zeros(5)
        else:
            # Smooth independent exploration around the observed reset posture.
            exploration = .60 * exploration + rng.normal(0., .23, 5)
            action = np.clip(exploration - .30 * state[7:12] - .13 * state[12:17], -.55, .55)
        initial = state
        score = 0.
        elapsed = 0
        for _ in range(BLOCK):
            obs, reward, done = arena.step(action)
            sense.previous = action
            state = sense.read(obs)
            score += reward
            elapsed += 1
            if done:
                break
        trace.append([initial.tolist(), action.tolist(), state.tolist(), score, done, elapsed])
    return trace


def collect(arena, output):
    exploration = np.random.default_rng(PRACTICE_SEED - 1)
    rows = []
    for i, launch in enumerate(seeds(PRACTICE_SEED, 16)):
        trace = practice(arena, launch, i, exploration)
        rows.append(dict(seed=launch, trace=trace))
        print('practice', i + 1, launch, sum(row[3] for row in trace), sum(row[-1] for row in trace), flush=True)
    with gzip.open(output, 'wt') as stream:
        json.dump(dict(schema='tt24-predict-1', practice=rows), stream, separators=(',', ':'))


def arrays(episodes):
    state, action, next_state, reward, done, elapsed = zip(
        *(row for episode in episodes for row in episode['trace']), strict=True
    )
    return tuple(np.asarray(value) for value in (state, action, next_state, reward, done, elapsed))


def _kernel(left, right, right_norm):
    distance = (left * left).sum(axis=1)[:, None] + right_norm[None, :] - 2. * left @ right.T
    return np.exp(-np.maximum(distance, 0.) * .5)


class VisualDynamics:
    """Ridge-regularized visual and encoder dynamics, with local residuals."""

    def __init__(self, episodes, *, use_command=True):
        state, command, following, reward, done, elapsed = arrays(episodes)
        self.use_command = use_command
        self.center = np.mean(state, axis=0)
        self.scale = np.maximum(np.std(state, axis=0), .035)
        self.scale[22:25] = np.maximum(self.scale[22:25], .025)
        self.output = np.c_[following[:, :17] - state[:, :17], following[:, 22:25] - state[:, 22:25],
                            reward, (~done).astype(float)]
        body_weights = np.maximum(np.linalg.norm(command, axis=1), .20)
        # Keep short terminal fragments as outcome witnesses. Ignore the
        # unequal-time sensor differences while fitting regular block dynamics.
        dynamic_weights = body_weights * (elapsed == BLOCK)
        self.support = self._context(state)
        self.norm = (self.support * self.support).sum(axis=1)
        x = self._linear(state, command)
        self.linear = self._fit(x, self.output, dynamic_weights, ridge=1.5)
        residual = self.output - x @ self.linear
        gram = _kernel(self.support, self.support, self.norm)
        self.adjustment = self._fit(gram, residual, dynamic_weights, ridge=1.6)
        self.outcome = self._fit(np.c_[x, gram], self.output[:, -2:], np.ones(len(state)), ridge=3.2)
        span = np.ptp(state, axis=0)
        self.low, self.high = state.min(axis=0) - .10 * span, state.max(axis=0) + .10 * span
        self.low[6], self.high[6] = 0., 1.

    @staticmethod
    def _fit(design, target, weights, ridge):
        penalty = ridge * np.eye(design.shape[1])
        penalty[0, 0] *= .05
        weighted = design * weights[:, None]
        return np.linalg.solve(design.T @ weighted + penalty, weighted.T @ target)

    def _context(self, state):
        normalized = (state - self.center) / self.scale
        return normalized[:, :17] * np.r_[[.5, 1., .65, .5, .9, .6, .5], np.full(10, .14)]

    def _linear(self, state, command):
        normalized = np.clip((state - self.center) / self.scale, -5., 5.)
        terms = [np.ones(len(state)), normalized[:, :17], normalized[:, 22:25]]
        if self.use_command:
            terms.extend((2. * command, 2. * state[:, 17:22],
                          normalized[:, 4, None] * command, normalized[:, 1, None] * command))
        return np.column_stack(terms)

    def predict(self, state, command):
        local = _kernel(self._context(state), self.support, self.norm)
        x = self._linear(state, command)
        result = x @ self.linear + local @ self.adjustment
        outcome = np.c_[x, local] @ self.outcome
        following = np.c_[state[:, :17] + result[:, :17], command, state[:, 22:25] + result[:, 17:20]]
        following = np.clip(following, self.low, self.high)
        reward = np.clip(outcome[:, 0], 0., .65)
        alive = np.clip(outcome[:, 1], .05, 1.)
        return following, reward, alive, local.max(axis=1)


class Planner:
    """Re-score continuous motor proposals from the latest sensor history."""

    def __init__(self, model, rng, prior, dt, *, horizon=5, samples=92):
        self.model, self.rng, self.prior, self.dt = model, rng, prior, dt
        self.horizon, self.samples = horizon, samples
        self.previous = np.zeros((horizon, 5))

    def _score(self, state, sequence):
        state = np.repeat(state[None], len(sequence), axis=0)
        value = np.zeros(len(sequence))
        alive = np.ones(len(sequence))
        history = [state]
        for step, command in enumerate(np.moveaxis(sequence, 1, 0)):
            following, reward, survival, support = self.model.predict(state, command)
            alive *= survival
            rate_cost = .16 * np.mean(command**2, axis=1) + .10 * np.mean((command - state[:, 17:22])**2, axis=1)
            posture_cost = .36 * np.mean(following[:, 7:12]**2, axis=1)
            value += .94**step * (alive * (reward + .18 + .12 * following[:, 6])
                                   - rate_cost - posture_cost - .035 * (1. - support))
            history.append(following)
            state = following
        if self.prior is not None:
            history = np.stack(history, axis=1)
            mover = history[:, :, :2] / 5.
            actor = history[:, :, 22:24]
            scale = np.maximum(np.mean(history[:, :, 24], axis=1), .02)
            source = self.prior.predict(actor, mover, dt=self.dt * BLOCK, scale=scale)
            value += .45 * alive * (source - self.prior.prior)
        return value

    def command(self, state):
        if state[6] == 0.:
            return np.clip(-.35 * state[7:12] - .18 * state[12:17], -.27, .27)
        start = self.previous.copy()
        scale = .18
        for _ in range(3):
            noise = self.rng.normal(0., scale, (self.samples, self.horizon, 5))
            sequence = np.clip(start + noise, -.32, .32)
            sequence[0], sequence[1] = start, np.zeros((self.horizon, 5))
            value = self._score(state, sequence)
            chosen = np.argpartition(value, -9)[-9:]
            start = np.mean(sequence[chosen], axis=0)
            scale *= .58
        self.previous = np.r_[start[1:], start[-1:]]
        return start[0]


def model_validation(episodes, use_command):
    errors = []
    for pair in ((2, 12), (6, 14), (4, 10)):
        model = VisualDynamics([row for i, row in enumerate(episodes) if i not in pair], use_command=use_command)
        x, u, y, reward, done, duration = arrays([episodes[i] for i in pair])
        pred, score, alive, _ = model.predict(x, u)
        good = (duration == BLOCK) & (y[:, 6] == 1.)
        errors.append(dict(appearance_mse=np.mean((y[good, :3] - pred[good, :3])**2, axis=0).tolist(),
                           motor_mse=float(np.mean((y[good, 7:17] - pred[good, 7:17])**2)),
                           reward_mae=float(np.mean(abs(np.clip(reward, 0., 1.) - score))),
                           termination_mae=float(np.mean(abs(~done - alive))), blocks=len(pred)))
    return errors


def evaluate(arena, launch, model, name, prior):
    sense = Senses()
    render = name not in ('blind', 'hold')
    obs = arena.reset(seed=launch, split='held', render=render)
    state = sense.read(obs)
    plan = Planner(model, np.random.default_rng(launch + 482), prior, arena.dt)
    done = False
    actions = []
    true_reward = 0.
    visible = 0
    while not done:
        action = np.zeros(5) if name == 'hold' else plan.command(state)
        for _ in range(BLOCK):
            obs, reward, done = arena.step(action, render=render)
            sense.previous = action
            state = sense.read(obs)
            visible += int(state[6])
            true_reward += reward
            actions.append(action.tolist())
            if done:
                break
    return arena.judge_metrics() | dict(command=actions, visible=visible, score=true_reward)


def qualify(arena, episodes, output):
    started = perf_counter()
    fits = {label: model_validation(episodes, enabled) for label, enabled in (('actual', True), ('omitted', False))}
    print('blocked prediction validation', fits, flush=True)
    model = VisualDynamics(episodes)
    held = seeds(HOLD_SEED, 12)
    results = {}
    for name in ('hold', 'plain', 'pong', 'shuffled', 'blind'):
        prior = VisualConsequence.load(name) if name in ('pong', 'shuffled') else None
        results[name] = []
        for i, launch in enumerate(held):
            row = evaluate(arena, launch, model, name, prior)
            results[name].append(row)
            print('held', name, i + 1, launch, row['rebounds'], row['steps'], flush=True)
    with gzip.open(output, 'wt') as stream:
        json.dump(dict(schema='tt24-predict-1', prediction_validation=fits, held=results,
                       wall_seconds=perf_counter() - started), stream, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('collect', 'qualify'))
    parser.add_argument('--data', type=Path, default=PATH)
    parser.add_argument('--output', type=Path, default=RESULT)
    args = parser.parse_args()
    arena = BounceArena(max_steps=360)
    try:
        if args.task == 'collect':
            collect(arena, args.data)
        else:
            with gzip.open(args.data, 'rt') as stream:
                episodes = json.load(stream)['practice']
            qualify(arena, episodes, args.output)
    finally:
        arena.close()


if __name__ == '__main__':
    main()
