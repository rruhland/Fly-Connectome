"""Recall a motor map by comparing delayed encoder predictions during wrist play."""

import argparse
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil
from .tt24_effects import transmission
from .tt24_inflight_adaptation import evaluate, set_load
from .tt24_ongoing_game import DIR
from .tt24_predict_witness import BODY_SEEDS, DATA as BODY_TRACE


OUTPUT = DIR / 'motor-recall.json.gz'
FRESH = DIR / 'motor-recall-fresh.json.gz'
STOCK = DIR / 'motor-recall-stock.npz'
CHECKPOINT = DIR / 'bounce-ident-pupil.npz'
BASELINE = DIR / 'inflight-native-experience.json.gz'
STOCK_SEEDS = tuple(range(257001, 257005))
TAPS = np.array((0, 1, 2, 3, 4, 6, 7, 9))
DELAY = 2
WINDOW = 5
STRIDE = 2
INC = 5.2 / 60


def sensor(obs):
    return np.r_[obs['joints'], obs['joint_velocity']].astype(float)


def features(trace, indices):
    i = np.asarray(indices)[:, None]
    tapped = trace[np.maximum(0, i - TAPS[None])]
    q = tapped[:, :, :5]
    v = tapped[:, :, 5:]
    displacement = q - trace[0, :5]
    changes = q[:, :-1] - q[:, -1:]
    return np.c_[q[:, 0], displacement[:, 0], changes.reshape(len(i), -1), v.reshape(len(i), -1)]


def example(states, commands):
    index = np.arange(max(TAPS), len(states))
    total = np.r_[np.zeros((1, 5)), np.cumsum(commands, axis=0)]
    latest = index - DELAY
    target = np.c_[total[latest] - total[latest - WINDOW], total[latest]]
    return features(states, index), target


class MotorObserver:
    """Infer recent effective action and actuator target from our own encoders."""

    def __init__(self, pairs):
        state, targets = map(np.concatenate, zip(*(example(x, u) for x, u in pairs), strict=True))
        self.center = state.mean(axis=0)
        self.scale = np.maximum(state.std(axis=0), .009)
        x = self.design(state)
        reg = .70 * np.eye(x.shape[1])
        reg[0, 0] = 0.
        self.weights = np.linalg.solve(x.T @ x + reg, x.T @ targets)

    def design(self, state):
        x = np.clip((state - self.center) / self.scale, -6., 6.)
        return np.c_[np.ones(len(x)), x]

    def estimate(self, states):
        ticks = np.array((len(states) - 1,))
        return (self.design(features(np.asarray(states), ticks)) @ self.weights)[0]


def collect(arena, checkpoint, path):
    record = {}
    for seed in STOCK_SEEDS:
        obs = arena.reset(seed=seed, split='practice')
        pupil = WristPupil.load(checkpoint)
        states = [sensor(obs)]
        actions = []
        done = False
        while not done:
            command = pupil.command(obs)
            obs, _, done = arena.step(command)
            actions.append(command)
            states.append(sensor(obs))
        record[f'obs_{seed}'] = np.asarray(states)
        record[f'act_{seed}'] = np.asarray(actions)
        print('stock ordinary', seed, len(actions), arena.judge_metrics()['rebounds'], flush=True)
    np.savez_compressed(path, **record)


def calibrate(path):
    with np.load(BODY_TRACE, allow_pickle=False) as body:
        probes = [(body[f'body_obs_{seed}'], body[f'body_action_{seed}']) for seed in BODY_SEEDS]
    with np.load(path, allow_pickle=False) as stock:
        ordinary = [(stock[f'obs_{seed}'], stock[f'act_{seed}']) for seed in STOCK_SEEDS]
    pairs = probes + ordinary
    residual = []
    audit = []
    for i, (state, actions) in enumerate(pairs):
        observer = MotorObserver([pair for j, pair in enumerate(pairs) if i != j])
        design, target = example(state, actions)
        failure = observer.design(design) @ observer.weights - target
        residual.append(failure[:, :5])
        audit.append(dict(equivalent_rmse=np.sqrt(np.mean(failure[:, :5]**2, axis=0)).tolist(),
                          cumulative_rmse=np.sqrt(np.mean(failure[:, 5:]**2, axis=0)).tolist(),
                          frames=len(actions)))
    all_errors = np.vstack(residual)
    scale = np.maximum(np.sqrt(np.mean(all_errors**2, axis=0)), .025)
    return MotorObserver(pairs), scale, audit


class MapFit:
    """Regularized action maps that learn with different memory horizons."""

    def __init__(self, decay):
        self.decay = decay
        self.moment = np.zeros((5, 5))
        self.cross = np.zeros((5, 5))
        self.transform = np.eye(5)

    def update(self, command, effect, weight=1.):
        self.moment *= self.decay
        self.cross *= self.decay
        self.moment += weight * np.outer(command, command)
        self.cross += weight * np.outer(command, effect)
        regularization = np.eye(5) * .024
        self.transform = np.linalg.solve(self.moment + regularization, self.cross + regularization).T


class MotorRecall:
    """Select a retained or recently learned inverse from causal prediction errors."""

    def __init__(self, observer, error_scale):
        self.observer, self.scale = observer, error_scale
        self.fast = MapFit(.936)
        self.retained = MapFit(.993)
        self.identity = np.eye(5)
        self.loss = np.zeros(3)
        self.mode = 0
        self.learned = False
        self.level = 0.

    def reset(self, obs):
        self.states = [sensor(obs)]
        self.requests = []
        self.events = []
        self.predictions = []

    def actuate(self, desired):
        modes = (self.identity, self.fast.transform, self.retained.transform)
        if self.mode == 0:
            return desired
        left, singular, right = np.linalg.svd(modes[self.mode])
        inverse = (right.T * (singular / (singular**2 + .057))) @ left.T
        blend = np.clip(self.level, .18, 1.)
        drive = (1. - blend) * desired + blend * (inverse @ desired)
        return np.clip(drive, -1., 1.)

    def observe(self, command, obs, pupil):
        self.states.append(sensor(obs))
        self.requests.append(command)
        ticks = len(self.requests)
        if ticks < max(TAPS) or ticks % STRIDE:
            return
        effect, fitted_target = np.split(self.observer.estimate(self.states), 2)
        action = np.sum(self.requests[ticks - DELAY - WINDOW:ticks - DELAY], axis=0)
        maps = np.asarray((self.identity, self.fast.transform, self.retained.transform))
        prediction = maps @ action
        errors = np.mean(np.log1p(((prediction - effect) / self.scale)**2), axis=1)
        errors += (.0, .17, .10)
        evidence = np.clip(np.linalg.norm(action) / .17, .2, 1.)
        self.loss += (.38 * evidence) * (errors - self.loss)
        winner = int(np.argmin(self.loss))
        if self.loss[winner] + .035 < self.loss[self.mode]:
            self.mode = winner
            self.events.append([ticks, winner])
            self.learned |= winner > 0
        self.level = min(1., max(0., 2.7 * float(self.loss[0] - self.loss[self.mode])))
        was_surprised = bool(errors[0] > .95 or self.loss[0] > .80)
        self.fast.update(action, effect, evidence)
        if was_surprised or self.mode:
            self.retained.update(action, effect, evidence)
        if self.learned and self.mode:
            recent = np.sum(self.requests[-DELAY:], axis=0)
            estimate = INC * (fitted_target + maps[self.mode] @ recent)
            pupil.issued += .19 * np.clip(estimate - pupil.issued, -.039, .039)
        self.predictions.append([ticks, self.mode, *errors.tolist(), self.level])


def shifted(drop, tick):
    return (drop == 2 and tick >= 24) or 3 <= drop <= 10 or (drop == 11 and tick < 24)


def replay(arena, index, row):
    arena.reset(seed=row['seed'], split='held', render=False)
    original = arena._drag_force_scale
    previous = None
    for tick, action in enumerate(row['actions']):
        is_changed = shifted(index, tick)
        if is_changed != previous:
            set_load(arena, is_changed, original)
            previous = is_changed
        _, _, done = arena.step(action, render=False)
        if done and tick + 1 != len(row['actions']):
            raise ValueError('physical replay ended before the stored tape')
    metrics = arena.judge_metrics()
    return metrics


def run_game(arena, checkpoint, seed, drop, wiring, adapter):
    obs = arena.reset(seed=seed, split='held')
    pupil = WristPupil.load(checkpoint)
    adapter.reset(obs)
    original = arena._drag_force_scale
    previous = None
    rewards, actions = [], []
    done = False
    while not done:
        tick = len(actions)
        is_changed = shifted(drop, tick)
        if is_changed != previous:
            set_load(arena, is_changed, original)
            previous = is_changed
        nominal = pupil.command(obs)
        command = adapter.actuate(nominal)
        physical = wiring @ command if is_changed else command
        obs, reward, done = arena.step(physical)
        if reward:
            rewards.append(tick)
        adapter.observe(command, obs, pupil)
        actions.append(physical.tolist())
    metrics = arena.judge_metrics()
    row = dict(seed=seed, run=metrics['longest_consecutive_rebounds'], score=metrics['rebounds'],
               ticks=len(actions), adaptation=adapter.events, reward_frames=rewards,
               actions=actions, proof=adapter.predictions)
    if replay(arena, drop, row) != metrics:
        raise ValueError('new adapter physical replay mismatch')
    return row


def summarize(rows):
    periods = dict(stock=(0, 2), first_switch=(2, 3), early_changed=(3, 6), settled_changed=(6, 11),
                   restored_inflight=(11, 12), restored=(12, 15))
    score = {key: sum(row['score'] for row in rows[a:b]) for key, (a, b) in periods.items()}
    score['first_switch_post24'] = sum(tick >= 24 for tick in rows[2]['reward_frames'])
    score['restored_inflight_post24'] = sum(tick >= 24 for tick in rows[11]['reward_frames'])
    return score


def run(arena, checkpoint, stock, baseline, output, *, fresh=False):
    observer, scale, audit = calibrate(stock)
    print('encoder action observation errors', audit, 'noise scales', scale.tolist(), flush=True)
    if fresh:
        rng = np.random.default_rng(26082113)
        seeds = rng.integers(1, 2**31 - 1, size=15).tolist()
        wiring = transmission(rng)
        controls = evaluate(arena, checkpoint, None, None, seeds, wiring, 'frozen')
        reference = dict(streams=[dict(wiring=wiring.tolist(), modes={'frozen': controls})])
    else:
        with gzip.open(baseline, 'rt') as stream:
            reference = json.load(stream)
    results = []
    for s, group in enumerate(reference['streams']):
        wiring = np.asarray(group['wiring'])
        incumbent = group['modes']
        for name, rows in incumbent.items():
            for index, row in enumerate(rows):
                metrics = replay(arena, index, row)
                if metrics['rebounds'] != row['score'] or metrics['longest_consecutive_rebounds'] != row['run']:
                    raise ValueError(f'archived control replay mismatch for stream {s} {name} drop {index}')
        adapter = MotorRecall(observer, scale)
        rows = []
        for i, untouched in enumerate(incumbent['frozen']):
            row = run_game(arena, checkpoint, untouched['seed'], i, wiring, adapter)
            print('motor recall', s, i, row['score'], row['run'], row['adaptation'], flush=True)
            rows.append(row)
        modes = incumbent | dict(recall=rows)
        summaries = {key: summarize(value) for key, value in modes.items()}
        results.append(dict(modes=modes, summary=summaries, wiring=group['wiring']))
        print(s, summaries, flush=True)
        with gzip.open(output, 'wt') as stream:
            json.dump(dict(schema='tt24-motor-recall-1', streams=results,
                           source_baselines=None if fresh else str(baseline), fresh=fresh,
                           verified_physical_replays=True, calibration=audit, error_scale=scale.tolist(),
                           setup=dict(ball_free_steps=sum(row['frames'] for row in audit[:len(BODY_SEEDS)]),
                                      stock_steps=sum(row['frames'] for row in audit[len(BODY_SEEDS):]))),
                      stream, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock', type=Path, default=STOCK)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    environment = BounceArena(view='wrist', width=192, height=144, max_steps=360)
    try:
        if not args.stock.exists():
            collect(environment, args.checkpoint, args.stock)
        output = args.output or (FRESH if args.fresh else OUTPUT)
        run(environment, args.checkpoint, args.stock, args.baseline, output, fresh=args.fresh)
    finally:
        environment.close()


if __name__ == '__main__':
    main()
