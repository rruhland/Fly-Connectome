"""Identify body motion on ball-free data and retrain only in the cheap worlds."""

import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt24_world_prior import CHECKPOINT, LIMITS, WEIGHTS, WorldPrior
from .tt24_world_prior_train import STATISTICS, Worlds, data, fit, outcome


ROOT = CHECKPOINT.parent
PLAY = ROOT / 'world-arm-public-play.npz'
OUTPUT = ROOT / 'world-prior-body.npz'
MODEL = ROOT / 'world-prior-body-model.json'
RECEIPT = ROOT / 'world-prior-body-source.json'
HELD_SEED = 22780619531
STRESS_SEED = 58507934077
SCALE = LIMITS[[0, 0, 1, 2, 2]]
RIDGES = (.0001, .001, .01, .10, .5)
DELAYS = tuple(range(7))
HORIZONS = (1, 6, 18)
WIDTH = 96


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        commands = np.asarray(archive['goal'], dtype=float)
        states = np.asarray(archive['measured'], dtype=float)
        lengths = archive['lengths'] if 'lengths' in archive else np.full(len(commands), commands.shape[1])
        dt = archive['dt'].item() if 'dt' in archive else 1. / 60
    if commands.ndim != 3 or states.ndim != 3 or commands.shape[2] != 5 or states.shape[2] != 5:
        raise ValueError('expected goal and measured arrays with shape (R,T,5)')
    if len(commands) < 6 or len(commands) != len(states) or len(lengths) != len(commands):
        raise ValueError('need at least six complete runs with aligned commanded and measured state')
    if not np.isclose(dt, 1. / 60.) or states.shape[1] not in (commands.shape[1], commands.shape[1] + 1):
        raise ValueError('use 60 Hz sequences aligned before each command')
    rows = []
    for command, state, raw_count in zip(commands, states, lengths, strict=True):
        count = int(raw_count)
        state_count = count + (len(state) > len(command) and count == len(command))
        if not 48 <= count <= len(command):
            raise ValueError('each complete movement run needs at least 48 aligned frames')
        cmd, sampled = command[:count], state[:state_count]
        if not np.isfinite(np.r_[cmd, sampled]).all():
            raise ValueError('recorded active frames must be finite')
        if np.max(np.abs(sampled[:, :3])) > .34 or np.max(np.linalg.norm(sampled[:, 3:], axis=1)) > 1.001:
            raise ValueError('measured geometry is outside expected metre or normal units')
        rows.append((cmd / SCALE, sampled / SCALE))
    return rows


def design(run, delay):
    goal, pose = run
    times = np.arange(max(1, delay), min(len(goal), len(pose) - 1))
    at = pose[times]
    delta = at - pose[times - 1]
    error = goal[times - delay] - at
    return np.column_stack((np.ones(len(times)), at, delta, error)), pose[times + 1] - at


def fit_response(runs, delay, ridge):
    states, results = (np.concatenate(rows) for rows in zip(*(design(run, delay) for run in runs), strict=True))
    values = np.linalg.solve(states.T @ states + ridge * np.diag(np.r_[.12, np.full(5, 4.), np.ones(10)]),
                             states.T @ results)
    change = (values[1:6] - values[11:16]).T
    drag = values[6:11].T
    system = np.block([[np.eye(5) + change, drag], [change, drag]])
    radius = float(np.max(np.abs(np.linalg.eigvals(system))))
    observed = np.concatenate([row[1] for row in runs])
    rates = np.maximum(.01, 1.25 * np.quantile(np.abs(results), .997, axis=0))
    positions = np.maximum(.006 / SCALE, 1.23 * np.quantile(np.abs(observed), .999, axis=0))
    return dict(coefficient=values, rate=rates, position=positions, radius=radius)


def estimate(model, goals, poses, delay, horizon, *, baseline=False):
    start = np.arange(max(1, delay), len(poses) - horizon)
    current = poses[start].copy()
    prior = poses[start - 1].copy()
    closure = np.array((.43, .43, .43, .36, .36))
    cap = np.array((.65, .65, .68, 3.1, 3.1)) / 60. / SCALE
    for step in range(horizon):
        demand = goals[start + step - delay]
        if baseline:
            following = current + np.clip((demand - current) * closure, -cap, cap)
        else:
            features = np.column_stack((np.ones(len(start)), current, current - prior, demand - current))
            output = np.clip(features @ model['coefficient'], -model['rate'], model['rate'])
            following = np.clip(current + output, -model['position'], model['position'])
        current, prior = following, current
    return current - poses[start + horizon]


def errors(model, runs, delay, *, baseline=False):
    results = {}
    for horizon in HORIZONS:
        rows = np.concatenate([estimate(model, goals, positions, delay, horizon, baseline=baseline)
                               for goals, positions in runs])
        rms = np.sqrt(np.mean(rows ** 2, axis=0))
        results[str(horizon)] = dict(normalized=float(np.sqrt(np.mean(rows ** 2))), per_axis=(rms * SCALE).tolist(),
                                    windows=len(rows))
    return results


def select(runs):
    """Select from complete-run cross-validation without touching held runs."""
    candidates = []
    for delay in DELAYS:
        for ridge in RIDGES:
            values = []
            for i in range(len(runs)):
                held = runs[i:i+1]
                model = fit_response(runs[:i] + runs[i+1:], delay, ridge)
                if model['radius'] >= 1.001:
                    break
                values.append(errors(model, held, delay))
            if len(values) != len(runs):
                continue
            criterion = float(np.mean([score[str(h)]['normalized'] for score in values for h in HORIZONS]))
            candidates.append(dict(delay=delay, ridge=ridge, criterion=criterion))
    if not candidates:
        raise ValueError('unable to find a stable linear movement response under whole-run validation')
    chosen = min(candidates, key=lambda row: row['criterion'])
    return chosen, candidates


def pack(model, initial, delay):
    return np.r_[model['coefficient'].ravel(), model['rate'], initial * SCALE, model['position'] * SCALE, delay]


def identify(path, output):
    runs = load(path)
    training, held = runs[:-2], runs[-2:]
    chosen, candidates = select(training)
    delay, ridge = chosen['delay'], chosen['ridge']
    full = fit_response(training, delay, ridge)
    if full['radius'] >= 1.001:
        raise ValueError('whole-training-run fit was unstable')
    bank = [pack(full, row[1][0], delay) for row in training]
    for idx in range(len(training)):
        potential = fit_response(training[:idx] + training[idx+1:], delay, ridge)
        if potential['radius'] < 1.001:
            bank.append(pack(potential, training[idx][1][0], delay))
    report = dict(schema='tt24-body-metrology-1', data=str(path), data_sha256=sha256(path.read_bytes()).hexdigest(),
                  used_fields=['goal', 'measured'], training_runs=list(range(len(training))),
                  withheld_runs=list(range(len(training), len(runs))), fit_frames=sum(len(row[1]) for row in training),
                  held_frames=sum(len(row[1]) for row in held), scale=SCALE.tolist(), selected=chosen,
                  variants=candidates, radius=full['radius'], held_model=errors(full, held, delay),
                  held_nominal=errors(None, held, 1, baseline=True), body_bank=np.asarray(bank).tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, separators=(',', ':')) + '\n')
    print('identified', chosen, 'held model', {h: round(s['normalized'], 4) for h, s in report['held_model'].items()},
          'held nominal', {h: round(s['normalized'], 4) for h, s in report['held_nominal'].items()}, flush=True)
    return np.asarray(bank), report


class AlignedWorlds(Worlds):
    def __init__(self, body, workers=2, frames=300):
        super().__init__(workers, frames)
        self.body = np.ascontiguousarray(body, dtype=np.float64)
        if self.body.ndim != 2 or self.body.shape[1] != WIDTH or not len(self.body):
            raise ValueError('invalid continuous body bank')
        self.aligned = self.native.tt24_world_aligned_rollouts
        p, i = ctypes.c_void_p, ctypes.c_int
        self.aligned.argtypes = [p, i, p, i, i, i, i, p, i, p]
        self.aligned.restype = None

    def play(self, population, instances, *, profile=0):
        policy = np.ascontiguousarray(np.atleast_2d(population), dtype=np.float64)
        launches = np.ascontiguousarray(instances, dtype=np.uint64)
        if policy.shape[1:] != (WEIGHTS,) or launches.ndim != 1 or not len(launches) or not len(policy):
            raise ValueError('invalid fitted-body rollout input')
        output = np.empty((len(policy), len(launches), len(STATISTICS)), dtype=np.float64)
        self.aligned(policy.ctypes.data, len(policy), launches.ctypes.data, len(launches), profile, self.frames,
                     self.workers, self.body.ctypes.data, len(self.body), output.ctypes.data)
        return output


def evaluate(world, old, new, seeds, profile):
    result = world.play((old, new, np.zeros(WEIGHTS)), seeds, profile=profile)
    outcomes = {name: outcome(rows) for name, rows in zip(('original', 'body_aligned', 'idle'), result, strict=True)}
    return dict(outcomes=outcomes, launches=seeds.tolist(), evaluation_episodes=int(np.prod(result.shape[:2])),
                evaluation_frames=int(result[..., 1].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--movement', type=Path, default=PLAY)
    parser.add_argument('--resume', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--receipt', type=Path, default=RECEIPT)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--seed', type=int, default=31105728)
    parser.add_argument('--iterations', type=int, default=184)
    parser.add_argument('--directions', type=int, default=42)
    parser.add_argument('--elite', type=int, default=17)
    parser.add_argument('--worlds', type=int, default=40)
    parser.add_argument('--development', type=int, default=160)
    parser.add_argument('--check-every', type=int, default=8)
    parser.add_argument('--held', type=int, default=500)
    parser.add_argument('--rate', type=float, default=.078)
    parser.add_argument('--sigma', type=float, default=.135)
    parser.add_argument('--stress-fraction', type=float, default=.25)
    parser.add_argument('--identify-only', action='store_true')
    args = parser.parse_args()
    if args.resume.resolve() == args.output.resolve():
        parser.error('retain the original checkpoint at a separate path')
    body, metrology = identify(args.movement, args.model)
    if args.identify_only:
        return
    original = WorldPrior.load(args.resume).weights
    start = perf_counter()
    world = AlignedWorlds(body, args.workers, args.frames)
    try:
        aligned, _, receipt = fit(world, args)
        fitting = set(receipt['training_seeds'] + receipt['development_seeds'])
        held = data(HELD_SEED, args.held)
        stress = data(STRESS_SEED, args.held)
        if fitting & (set(held.tolist()) | set(stress.tolist())):
            raise ValueError('held evaluation and fitting overlap')
        receipt['held'] = evaluate(world, original, aligned, held, 0)
        receipt['stress'] = evaluate(world, original, aligned, stress, 2)
    finally:
        world.close()
    original_world = Worlds(args.workers, args.frames)
    try:
        receipt['source_retention'] = evaluate(original_world, original, aligned, data(40915212603, args.held), 0)
    finally:
        original_world.close()
    receipt |= dict(schema='tt24-world-aligned-1', body_model_sha256=sha256(args.model.read_bytes()).hexdigest(),
                    movement_sha256=metrology['data_sha256'], original_sha256=sha256(args.resume.read_bytes()).hexdigest(),
                    controller_sha256=sha256(args.output.read_bytes()).hexdigest(), target_reward_exposures=0,
                    elapsed_s=perf_counter() - start,
                    settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    args.receipt.write_text(json.dumps(receipt, separators=(',', ':')) + '\n')
    for phase in ('held', 'stress', 'source_retention'):
        print(phase, {name: round(row['rebounds'], 3) for name, row in receipt[phase]['outcomes'].items()}, flush=True)


if __name__ == '__main__':
    main()
