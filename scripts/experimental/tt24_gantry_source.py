"""Refine the analytic feedback prior on an independent MuJoCo Cartesian body."""

import argparse
import ctypes
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import mujoco
import numpy as np

from .tt24_world_prior import CHECKPOINT, WEIGHTS, WorldPrior
from .tt24_world_prior_train import RECEIPT, SOURCE as ANALYTIC_CODE, STATISTICS, Worlds, data, outcome, save, utility


NATIVE_CODE = Path(__file__).with_name('tt24_gantry_source_native.cpp')
DEV_SEED = 78135821
HELD_SEED = 275309014
STRESS_SEED = 673983235


class GantryWorlds:
    """Compile MuJoCo's tiny source scene and run outcomes in native C++."""

    def __init__(self, frames=300):
        self.frames = frames
        self.workspace = TemporaryDirectory(prefix='tt24-gantry-')
        library = Path(self.workspace.name) / 'gantry.so'
        package = Path(mujoco.__file__).parent
        engine = package / f'libmujoco.so.{mujoco.__version__}'
        if not engine.is_file():
            raise RuntimeError(f'expected the MuJoCo C runtime at {engine}')
        subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC', '-fopenmp',
                        '-I', str(package / 'include'), str(NATIVE_CODE), str(ANALYTIC_CODE), str(engine),
                        f'-Wl,-rpath,{package}', '-o', str(library)], check=True)
        self.native = ctypes.CDLL(str(library))
        pointer = ctypes.c_void_p
        self.native.tt24_gantry_open.argtypes = []
        self.native.tt24_gantry_open.restype = pointer
        self.native.tt24_gantry_close.argtypes = [pointer]
        self.native.tt24_gantry_close.restype = None
        self.native.tt24_gantry_rollouts.argtypes = [pointer, pointer, ctypes.c_int, pointer, ctypes.c_int,
                                                     ctypes.c_int, ctypes.c_int, pointer]
        self.native.tt24_gantry_rollouts.restype = None
        self.handle = self.native.tt24_gantry_open()
        if not self.handle:
            raise RuntimeError('the native Cartesian gantry model could not be initialized')

    def close(self):
        if self.handle:
            self.native.tt24_gantry_close(self.handle)
            self.handle = None
        self.workspace.cleanup()

    def play(self, population, launches, *, profile=0):
        weights = np.ascontiguousarray(np.atleast_2d(population), dtype=np.float64)
        seeds = np.ascontiguousarray(launches, dtype=np.uint64)
        if weights.ndim != 2 or weights.shape[1] != WEIGHTS or seeds.ndim != 1 or not len(weights) or not len(seeds):
            raise ValueError('expected a nonempty batch of 42-weight readouts and source seeds')
        result = np.empty((len(weights), len(seeds), len(STATISTICS)), dtype=np.float64)
        self.native.tt24_gantry_rollouts(self.handle, weights.ctypes.data, len(weights), seeds.ctypes.data,
                                        len(seeds), profile, self.frames, result.ctypes.data)
        return result


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _concise(summary):
    return {key: summary[key] for key in ('rebounds', 'consecutive', 'median', 'maximum', 'at_least_ten', 'drops')}


def train(gantry, initial, args):
    rng = np.random.default_rng(args.seed)
    development = data(DEV_SEED, args.development)
    weights = initial.copy()
    best = weights.copy()
    original = gantry.play(weights, development)[0]
    best_score = float(utility(original, gantry.frames))
    curve, train_seeds = [], []
    exposures, frames = 0, 0
    scales = np.r_[np.full(10, 1.25), np.full(16, .67), np.full(16, 1.)]
    start = perf_counter()
    for iteration in range(1, args.iterations + 1):
        fraction = (iteration - 1) / max(1, args.iterations - 1)
        directions = rng.normal(size=(args.directions, WEIGHTS)) * scales
        perturbation = args.sigma * (1. - .60 * fraction) * directions
        candidates = np.concatenate((weights + perturbation, weights - perturbation))
        instances = rng.integers(1, 2**63, size=args.worlds, dtype=np.uint64)
        train_seeds.extend(instances.tolist())
        standard_count = args.worlds - args.worlds // 4
        regular = gantry.play(candidates, instances[:standard_count])
        wider = gantry.play(candidates, instances[standard_count:], profile=1) if standard_count < args.worlds else None
        scores = regular if wider is None else np.concatenate((regular, wider), axis=1)
        exposures += int(np.prod(scores.shape[:2]))
        frames += int(scores[..., 1].sum())
        positive, negative = utility(scores, gantry.frames).reshape(2, args.directions)
        elite = np.argsort(-np.maximum(positive, negative), kind='stable')[:args.elite]
        spread = max(.15, float(np.std(np.r_[positive[elite], negative[elite]])))
        advantage = (positive[elite] - negative[elite]) / spread
        gradient = np.mean(advantage[:, None] * directions[elite], axis=0)
        weights = np.clip(weights + args.rate * (1. - .55 * fraction) * gradient, -5.5, 5.5)
        if iteration == 1 or iteration % 4 == 0 or iteration == args.iterations:
            checked = gantry.play(weights, development)[0]
            score = float(utility(checked, gantry.frames))
            if score > best_score:
                best_score = score
                best = weights.copy()
                save(args.out / 'gantry-source.npz', best)
            row = dict(iteration=iteration, details=_concise(outcome(checked)), utility=score, best=best_score,
                       elapsed_s=perf_counter() - start)
            curve.append(row)
            print('source training', row, flush=True)
    save(args.out / 'gantry-source.npz', best)
    training = dict(development_initial=outcome(original), source_training_episodes=exposures,
                    source_training_frames=frames, development_curve=curve, training_seeds=train_seeds,
                    development_seeds=development.tolist(), elapsed_s=perf_counter() - start)
    return best, training


def compare(world, prior, refined, seeds, *, profile=0):
    entries = world.play((prior, refined, np.zeros(WEIGHTS)), seeds, profile=profile)
    return dict(seeds=list(map(int, seeds)), outcomes={name: outcome(values) for name, values in zip(
        ('analytic_prior', 'gantry_refined', 'stationary'), entries, strict=True)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24'))
    parser.add_argument('--initial', type=Path, default=CHECKPOINT)
    parser.add_argument('--analytic-receipt', type=Path, default=RECEIPT)
    parser.add_argument('--seed', type=int, default=8539261)
    parser.add_argument('--iterations', type=int, default=24)
    parser.add_argument('--directions', type=int, default=12)
    parser.add_argument('--elite', type=int, default=5)
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=.21)
    parser.add_argument('--rate', type=float, default=.10)
    parser.add_argument('--development', type=int, default=32)
    parser.add_argument('--held', type=int, default=128)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if min(args.iterations, args.directions, args.elite, args.worlds, args.development, args.held) < 1 or args.elite > args.directions:
        parser.error('need positive training counts and no more elites than directions')
    args.out.mkdir(parents=True, exist_ok=True)
    prior = WorldPrior.load(args.initial).weights
    gantry = GantryWorlds()
    try:
        if args.probe:
            values = gantry.play((prior, np.zeros(WEIGHTS)), data(DEV_SEED, 24))
            for name, rows in zip(('analytic_prior', 'stationary'), values, strict=True):
                print(name, _concise(outcome(rows)), flush=True)
            return
        best, trace = train(gantry, prior, args)
        evaluated = dict(held=compare(gantry, prior, best, data(HELD_SEED, args.held)),
                         stress=compare(gantry, prior, best, data(STRESS_SEED, args.held), profile=1))
        original = json.loads(args.analytic_receipt.read_text())
        analytic = Worlds(workers=1)
        try:
            evaluated['original_analytic_held'] = compare(analytic, prior, best, original['held']['launches'])
            evaluated['original_analytic_stress'] = compare(analytic, prior, best, original['stress']['launches'], profile=2)
        finally:
            analytic.close()
        for name, series in evaluated.items():
            print(name, {kind: _concise(statistics) for kind, statistics in series['outcomes'].items()}, flush=True)
        source = dict(schema='tt24-cartesian-gantry-source-1', mujoco_version=mujoco.__version__,
                      source_observations='clean metric ball and paddle pose and ball motion',
                      training_feedback='source scores only; paired random coefficient search',
                      target_practice=False, target_motor_labels=False, initial_policy_sha256=_digest(args.initial),
                      refined_policy_sha256=_digest(args.out / 'gantry-source.npz'), native_scene_sha256=_digest(NATIVE_CODE),
                      settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                      training=trace, evaluations=evaluated)
        with gzip.open(args.out / 'gantry-source.json.gz', 'wt', encoding='utf-8') as file:
            json.dump(source, file, separators=(',', ':'))
    finally:
        gantry.close()


if __name__ == '__main__':
    main()
