"""Train the 3D source-world feedback prior from scored perturbation rollouts."""

import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .tt24_world_prior import CHECKPOINT, SCHEMA, WEIGHTS, WorldPrior


SOURCE = Path(__file__).with_name('tt24_world_prior_native.cpp')
RECEIPT = CHECKPOINT.with_name('world-prior-source.json')
STATISTICS = ('rebounds', 'frames', 'consecutive', 'contacts', 'mean_rise', 'powered')
DEVELOPMENT_SEED = 6230961151
HELD_SEED = 9425180937
STRESS_SEED = 11219596817
WIDE_DEVELOPMENT_SEED = 2477407201


class Worlds:
    """Run parallel source environments with identical physics for every treatment."""

    def __init__(self, workers=2, frames=300):
        self.workspace = TemporaryDirectory(prefix='tt24-world-')
        library = Path(self.workspace.name) / 'worlds.so'
        subprocess.run(['c++', '-std=c++17', '-O3', '-ffast-math', '-fopenmp', '-shared', '-fPIC',
                        str(SOURCE), '-o', str(library)], check=True)
        code = ctypes.CDLL(str(library))
        self.native = code
        pointer = ctypes.c_void_p
        self.evaluate = code.tt24_world_rollouts
        self.evaluate.argtypes = [pointer, ctypes.c_int, pointer, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, pointer]
        self.evaluate.restype = None
        self._target = code.tt24_world_target
        self._target.argtypes = [pointer, pointer, pointer]
        self._target.restype = None
        self.frames, self.workers = frames, workers

    def close(self):
        self.workspace.cleanup()

    def play(self, population, instances, *, profile=0):
        population = np.ascontiguousarray(np.atleast_2d(population), dtype=np.float64)
        instances = np.ascontiguousarray(instances, dtype=np.uint64)
        if population.ndim != 2 or population.shape[1] != WEIGHTS or instances.ndim != 1:
            raise ValueError('invalid rollout batch dimensions')
        if not len(instances) or not len(population):
            raise ValueError('need nonempty policy and world batches')
        result = np.empty((len(population), len(instances), len(STATISTICS)), dtype=np.float64)
        self.evaluate(population.ctypes.data, len(population), instances.ctypes.data, len(instances),
                      profile, self.frames, self.workers, result.ctypes.data)
        return result

    def target(self, weights, state):
        w = np.ascontiguousarray(weights, dtype=np.float64)
        x = np.ascontiguousarray(state, dtype=np.float64)
        if w.shape != (WEIGHTS,) or x.shape != (11,):
            raise ValueError('expected one policy and one state')
        result = np.empty(5, dtype=np.float64)
        self._target(w.ctypes.data, x.ctypes.data, result.ctypes.data)
        return result


def data(seed, count):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 2**63, size=count, dtype=np.uint64)


def utility(result, frames):
    score = result[..., 0] + .19 * result[..., 2] + .52 * result[..., 1] / frames
    return score.mean(axis=-1)


def outcome(result):
    stats = {label: float(average) for label, average in zip(STATISTICS, result.mean(axis=0), strict=True)}
    counts = result[:, 0]
    return stats | dict(maximum=int(counts.max()), median=float(np.median(counts)),
                        at_least_five=int(np.count_nonzero(counts >= 5)), at_least_ten=int(np.count_nonzero(counts >= 10)),
                        at_least_fifteen=int(np.count_nonzero(counts >= 15)), drops=len(counts),
                        per_drop=result.tolist())


def save(path, weights):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        np.savez_compressed(temporary, schema=np.array(SCHEMA), weights=np.array(weights, dtype=np.float64))
        temporary.replace(path)


def qualify(world, weights, frozen, seed, count, profile):
    launches = data(seed, count)
    runs = world.play((weights, np.zeros(WEIGHTS), frozen), launches, profile=profile)
    checked = {name: outcome(rows) for name, rows in zip(('learned', 'idle', 'frozen_random'), runs, strict=True)}
    return dict(seed=seed, launches=launches.tolist(), outcomes=checked,
                episodes=int(np.prod(runs.shape[:2])), frames=int(runs[..., 1].sum()))


def fit(world, args, *, progress=True):
    rng = np.random.default_rng(args.seed)
    development = data(DEVELOPMENT_SEED, args.development)
    wide = data(WIDE_DEVELOPMENT_SEED, args.development) if args.stress_fraction else np.array([], dtype=np.uint64)
    frozen_random = np.random.default_rng(args.seed ^ 214118524).normal(0., .25, WEIGHTS)
    weights = np.zeros(WEIGHTS) if args.resume is None else WorldPrior.load(args.resume).weights.copy()
    def checkpoints(w):
        nominal = world.play(w, development)[0]
        if not len(wide):
            return nominal, nominal, 0.
        varied = world.play(w, wide, profile=1)[0]
        return nominal, np.concatenate((nominal, varied)), float(varied[:, 0].mean())

    _, calibration, _ = checkpoints(weights)
    best = weights.copy()
    priority = float(utility(calibration, world.frames))
    used = dict(training_episodes=0, training_frames=0, development_episodes=len(calibration),
                development_frames=int(calibration[:, 1].sum()))
    pairs = []
    train_seeds = []
    started = perf_counter()
    coefficient_scale = np.r_[np.full(10, 1.25), np.full(16, .67), np.full(16, 1.)]
    for iteration in range(1, args.iterations + 1):
        fraction = (iteration - 1) / max(1, args.iterations - 1)
        spread = args.sigma * (1. - .65 * fraction)
        direction = rng.normal(size=(args.directions, WEIGHTS)) * coefficient_scale
        population = weights + spread * np.concatenate((direction, -direction))
        worlds = rng.integers(1, 2**63, size=args.worlds, dtype=np.uint64)
        train_seeds += worlds.tolist()
        varied = round(args.worlds * args.stress_fraction)
        if varied:
            baseline = world.play(population, worlds[:-varied])
            result = np.concatenate((baseline, world.play(population, worlds[-varied:], profile=1)), axis=1)
        else:
            result = world.play(population, worlds)
        positive, negative = utility(result, world.frames).reshape(2, args.directions)
        select = np.argsort(-np.maximum(positive, negative), kind='stable')[:args.elite]
        normal = np.std(np.r_[positive[select], negative[select]])
        advantage = (positive[select] - negative[select]) / max(.15, normal)
        change = np.mean(advantage[:, None] * direction[select], axis=0)
        step = args.rate * (1. - .59 * fraction)
        weights = np.clip(weights + step * change, -5.5, 5.5)
        used['training_episodes'] += int(np.prod(result.shape[:2]))
        used['training_frames'] += int(result[..., 1].sum())
        if iteration == 1 or iteration % args.check_every == 0 or iteration == args.iterations:
            nominal, tested, wide_mean = checkpoints(weights)
            used['development_episodes'] += len(tested)
            used['development_frames'] += int(tested[:, 1].sum())
            criterion = float(utility(tested, world.frames))
            if criterion > priority:
                best = weights.copy()
                priority = criterion
                save(args.output, best)
            pairs.append(dict(iteration=iteration, development_mean=float(tested[:, 0].mean()),
                              nominal_development_mean=float(nominal[:, 0].mean()),
                              wide_development_mean=wide_mean, development_max=int(tested[:, 0].max()), criterion=criterion,
                              best_criterion=priority, elapsed_s=perf_counter() - started))
            if progress:
                print(iteration, 'dev rebounds', round(float(tested[:, 0].mean()), 3),
                      'max', int(tested[:, 0].max()), 'criterion', round(criterion, 3),
                      'best', round(priority, 3), 'elapsed', round(perf_counter() - started, 1), flush=True)
    development_seeds = development.tolist() + wide.tolist()
    if set(train_seeds) & set(development_seeds):
        raise ValueError('training and development seeds overlap')
    save(args.output, best)
    return best, frozen_random, dict(exposure=used, development_curve=pairs, development_seeds=development_seeds,
                                     training_seeds=train_seeds, elapsed_s=perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CHECKPOINT)
    parser.add_argument('--receipt', type=Path, default=RECEIPT)
    parser.add_argument('--seed', type=int, default=8924381)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--directions', type=int, default=40)
    parser.add_argument('--elite', type=int, default=14)
    parser.add_argument('--worlds', type=int, default=30)
    parser.add_argument('--development', type=int, default=120)
    parser.add_argument('--check-every', type=int, default=8)
    parser.add_argument('--held', type=int, default=384)
    parser.add_argument('--rate', type=float, default=.095)
    parser.add_argument('--sigma', type=float, default=.17)
    parser.add_argument('--stress-fraction', type=float, default=.26)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--development-only', action='store_true')
    args = parser.parse_args()
    if min(args.workers, args.frames, args.iterations, args.directions, args.elite, args.worlds,
           args.development, args.check_every, args.held) < 1 or args.elite > args.directions:
        parser.error('need positive rollout dimensions and no more elite directions than sampled')
    if not 0. <= args.stress_fraction < 1. or round(args.worlds * args.stress_fraction) >= args.worlds:
        parser.error('nominal worlds must remain in every training batch')
    initial_hash = sha256(args.resume.read_bytes()).hexdigest() if args.resume is not None else None
    world = Worlds(args.workers, args.frames)
    try:
        best, frozen, report = fit(world, args)
        report['settings'] = {key: str(x) if isinstance(x, Path) else x for key, x in vars(args).items()}
        report['schema'] = 'tt24-world-prior-training-1'
        report['policy_sha256'] = sha256(args.output.read_bytes()).hexdigest()
        report['physics_sha256'] = sha256(SOURCE.read_bytes()).hexdigest()
        report['frozen_random_weights'] = frozen.tolist()
        if args.resume is not None:
            report['initial_policy_sha256'] = initial_hash
        if not args.development_only:
            report['held'] = qualify(world, best, frozen, HELD_SEED, args.held, 0)
            report['stress'] = qualify(world, best, frozen, STRESS_SEED, args.held, 2)
            fitted = set(report['training_seeds'] + report['development_seeds'])
            for name in ('held', 'stress'):
                passed = report[name]
                if fitted & set(passed['launches']):
                    raise ValueError('held and fitting worlds must be disjoint')
                print(name, {mode: {k: stats[k] for k in ('rebounds', 'median', 'maximum', 'at_least_ten')}
                             for mode, stats in passed['outcomes'].items()}, flush=True)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, separators=(',', ':')) + '\n')
    finally:
        world.close()


if __name__ == '__main__':
    main()
