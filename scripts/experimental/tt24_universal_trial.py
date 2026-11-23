"""Fit channel-neutral quadratic feedback on the three hinged donor bodies."""

import argparse
import ctypes
from dataclasses import asdict
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.linalg import qr

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena, mujoco
from .tt24_bounce_ident import CleanState
from .tt24_donor_trial import (DESIGNS, DEV_RNG, DONOR_HELD_RNG, NATIVE, ROOT, STOCK_HELD_RNG, TRAIN_RNG,
                               Native as OriginalNative, World, _evaluate, _live, _paired, _seeds, _summary, _utility)
from .tt24_public_body import FUSED_OUTPUT


CHANNELS = 11
QUADRATIC = tuple((i, j) for i in range(CHANNELS) for j in range(i, CHANNELS))
PAIRS = {pair: 1 + CHANNELS + i for i, pair in enumerate(QUADRATIC)}
FEATURES = 1 + CHANNELS + len(PAIRS)
SHAPE = (5, FEATURES)
FRAGMENT = Path(__file__).with_name('tt24_universal_native.cpp')
ORIGINAL_RECEIPT = ROOT / 'tt24_donor-refined.json.gz'
ORIGINAL_CHECKPOINT = ROOT / 'tt24_donor-refined.npz'
CHECKPOINT = ROOT / 'universal-donor.npz'
RECEIPT = ROOT / 'universal-donor.json.gz'


class UniversalNative:
    """Compile the unchanged donor plant/scorer with a different feature function."""

    def __init__(self):
        self.working = TemporaryDirectory(prefix='tt24_universal-')
        output = Path(self.working.name)
        model = NATIVE.read_text()
        first = model.index('void features(')
        last = model.index('void command(', first)
        model = model[:first] + FRAGMENT.read_text() + '\n' + model[last:]
        model = model.replace('constexpr int kFeatures = 20;', f'constexpr int kFeatures = {FEATURES};')
        model = model.replace('tt24_donor_', 'tt24_universal_')
        model += '''
extern "C" void tt24_universal_configuration(const double* center, const double* scale, const double* basis) {
    for (int i = 0; i < kChannels; ++i) {
        universal_center[i] = center[i];
        universal_scale[i] = 1. / scale[i];
    }
    for (int i = 0; i < kChannels*kChannels; ++i) universal_matrix[i] = basis[i];
}
'''
        source, library = output / 'universal.cpp', output / 'universal.so'
        source.write_text(model)
        package = Path(mujoco.__file__).parent
        engine = package / f'libmujoco.so.{mujoco.__version__}'
        subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared', '-fPIC', '-I',
                        str(package / 'include'), str(source), str(engine), f'-Wl,-rpath,{package}', '-o', str(library)],
                       check=True)
        self.so = ctypes.CDLL(str(library))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        signatures = {'open': ([pointer], pointer), 'close': ([pointer], None),
                      'evaluate': ([pointer, pointer, integer, pointer, pointer, integer, integer, pointer], None),
                      'command': ([pointer, pointer, pointer, pointer], None),
                      'configuration': ([pointer, pointer, pointer], None)}
        for name, (inputs, result) in signatures.items():
            interface = getattr(self.so, f'tt24_universal_{name}')
            interface.argtypes, interface.restype = inputs, result
            setattr(self, name, interface)

    def install(self, center, scale, basis):
        arrays = [np.ascontiguousarray(spec, dtype=np.float64) for spec in (center, scale, basis)]
        if ([array.shape for array in arrays] != [(CHANNELS,), (CHANNELS,), (CHANNELS, CHANNELS)]
                or not all(np.isfinite(array).all() for array in arrays) or np.any(arrays[1] <= 0.)):
            raise ValueError('expected a finite eleven-channel transform with positive scales')
        self.configuration(*(spec.ctypes.data for spec in arrays))

    def load_checkpoint(self, path):
        """Install the fitted feature transform and return all five readout rows."""
        with np.load(path, allow_pickle=False) as saved:
            if saved['schema'].item() != 'tt24-universal-feedback-1':
                raise ValueError('unsupported universal feedback checkpoint')
            self.install(saved['center'], saved['scale'], saved['basis'])
            weights = saved['weights']
        if weights.shape != SHAPE or not np.isfinite(weights).all():
            raise ValueError(f'expected finite readout weights of shape {SHAPE}')
        return weights


class UniversalWorld(World):
    def play(self, weights, batch, cap=300):
        states, physics = batch
        options = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1, *SHAPE)
        scores = np.empty((len(options), len(states), 4))
        self.native.evaluate(self.handle, options.ctypes.data, len(options), states.ctypes.data,
                             physics.ctypes.data, len(states), cap, scores.ctypes.data)
        return scores


def _read_sensors(arena, seed, split, action, frames, *, score=False):
    obs = arena.reset(seed=int(seed), split=split, regime='changed', render=False)
    observation = CleanState(arena)
    samples = [observation.observe(obs)]
    for _ in range(frames):
        obs, _, done = arena.step(action, render=False)
        samples.append(observation.observe(obs))
        if done:
            break
    value = None
    if score:
        judge = arena.judge_metrics()
        value = [judge['rebounds'], judge['longest_consecutive_rebounds'], judge['steps'], judge['paddle_contacts']]
    return np.asarray(samples), value


def calibrate(worlds, first, cap):
    probe_data, neutral_data, checks, endpoint = [], [], [], []
    for body, (world, starts) in enumerate(zip(worlds, first, strict=True)):
        trials = []
        for axis in range(5):
            pair = []
            for sign in (-1., 1.):
                action = np.eye(5)[axis] * .26 * sign
                samples, _ = _read_sensors(world.arena, 1916070932 + body, 'practice', action, 5)
                if len(samples) != 6:
                    raise ValueError('the independently timed motor probe ended early')
                probe_data.append(samples)
                pair.append(samples[-1, :])
            trials.append(pair)
        endpoint.append(trials)
        idle = [_read_sensors(world.arena, seed, 'held', np.zeros(5), cap, score=True) for seed in starts]
        neutral_data.extend(samples for samples, _ in idle)
        checks.append([val for _, val in idle])
    observed = np.concatenate(probe_data + neutral_data)
    center = np.median(observed, axis=0)
    scale = np.maximum(.027, np.quantile(abs(observed - center), .84, axis=0))

    def normalize(q):
        return np.clip(np.arcsinh((q - center) / scale), -3.4, 3.4)

    changes = np.concatenate([np.diff(normalize(row), axis=0) for row in probe_data + neutral_data])
    eigenvalues, vectors = np.linalg.eigh(changes.T @ changes / len(changes))
    order = np.argsort(-eigenvalues)
    psychophysics = normalize(np.asarray(endpoint))
    difference = (psychophysics[:, :, 1] - psychophysics[:, :, 0]) / 2.
    response = difference.mean(axis=0).T
    _, _, sensitive = qr(response.T, pivoting=True, mode='economic')
    inverse = np.linalg.solve(response.T @ response + .08**2 * np.eye(5), response.T)
    motor = inverse[:, sensitive[:5]]
    motor /= np.linalg.norm(motor, axis=0)
    basis = np.ascontiguousarray(vectors[:, order].T)
    controllability = np.linalg.norm(basis @ response, axis=1)
    return dict(center=center, scale=scale, basis=basis, motor=motor, shifts=eigenvalues[order],
                controllability=controllability, motor_channel_pivots=sensitive[:5]), np.asarray(checks, dtype=float)


def _candidate(rng, generation, index, stats, noise, stalled):
    sensory = np.zeros(FEATURES)
    temporal = np.maximum(0., stats['shifts'])**.38
    temporal += .025 * max(1.e-9, temporal.max())
    temporal /= temporal.sum()
    pair = rng.choice(CHANNELS, size=2, replace=False, p=temporal)
    a, b = int(pair[0]), int(pair[1])
    interaction = PAIRS[tuple(sorted((a, b)))]
    group = index % 6
    if group in (0, 1):
        deck = (0, 1 + a, 1 + b, PAIRS[a, a], PAIRS[b, b], interaction)
        selected = rng.choice(deck, size=3, replace=False)
        sensory[selected] = rng.normal(size=3)
    elif group == 2:
        c = int(rng.integers(CHANNELS))
        sensory[1 + c], sensory[PAIRS[tuple(sorted((a, c)))]], sensory[PAIRS[c, c]] = rng.normal(size=3)
    elif group == 3:
        actual = stats['controllability'] + .014
        pairs = rng.choice(CHANNELS, size=5, replace=False, p=actual / actual.sum())
        sensory[np.r_[0, pairs + 1]] = rng.normal(size=6)
    elif group == 4:
        feature = rng.choice((rng.integers(1 + CHANNELS), rng.integers(FEATURES)))
        sensory[feature] = rng.choice((-1., 1.))
    else:
        sensory[:] = rng.normal(size=FEATURES) * (.35 * np.sqrt(20. / FEATURES))
    axes = stats['motor'][:, (generation + index) % 5]
    change = np.outer(axes, sensory)
    change += rng.normal(size=SHAPE) * np.sqrt(20. / FEATURES) * (.12 if group < 4 else .35)
    spread = rng.choice((.24, .48, 1., 1.5)) * noise
    if stalled > 15 and group < 2:
        spread *= 1.5
    return change * spread / (1. + .013 * generation)


def _qualitative_utility(scores, cap):
    blend = scores.copy()
    blend[..., 0] = .55 * scores[..., 0] + .45 * scores[..., 1]
    return _utility(blend, cap)


def _store(path, weights, stats):
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, schema=np.array('tt24-universal-feedback-1'), weights=weights,
                        **{k: stats[k] for k in ('center', 'scale', 'basis')})
    tmp.replace(path)


def train(worlds, starts, first_scores, stats, args, receipt):
    rng = np.random.default_rng(TRAIN_RNG + 965252)
    best = np.zeros(SHAPE)
    dev = _seeds(DEV_RNG + 11117, len(worlds), 8)
    checked = [world.prepare(seeds, 'held') for world, seeds in zip(worlds, dev, strict=True)]

    def validate(options):
        return np.asarray([world.play(options, batch) for world, batch in zip(worlds, checked, strict=True)])

    neutral = validate(best)
    criterion = float(_qualitative_utility(neutral, 300)[0])
    exposures = dict(training_drops=0, training_frames=0, development_drops=int(np.prod(neutral.shape[:-1])),
                     development_frames=int(neutral[..., 2].sum()), motor_probes=30, motor_probe_frames=150)
    receipt.update(dev_seeds=dev, baseline_utility=criterion, updates=[], exposure=exposures)
    _store(args.checkpoint, best, stats)
    last_gain = 0
    begun = perf_counter()
    for generation, launches in enumerate(starts):
        directions = np.stack([_candidate(rng, generation, index, stats, args.noise, generation - last_gain)
                               for index in range(args.directions)])
        options = np.clip(np.concatenate((best[None], best[None] + directions, best[None] - directions)), -2., 2.)
        if generation == 0:
            shifted = _evaluate(worlds, options[1:], launches, split='held', cap=args.train_cap)
            training = np.concatenate((first_scores[:, None], shifted), axis=1)
        else:
            training = _evaluate(worlds, options, launches, split='held', cap=args.train_cap)
        rank = np.argsort(-_qualitative_utility(training, args.train_cap), kind='stable')
        leaders = np.asarray([index for index in rank if index != 0][:3])
        evaluated = validate(options[leaders])
        quality = _qualitative_utility(evaluated, 300)
        winner = int(np.argmax(quality))
        if quality[winner] > criterion:
            best, criterion = options[leaders[winner]].copy(), float(quality[winner])
            last_gain = generation
            _store(args.checkpoint, best, stats)
        receipt['updates'].append(dict(generation=generation + 1, training_seeds=launches,
                                       training_results=training.tolist(), development_results=evaluated.tolist(),
                                       development_selection_indices=leaders.tolist(), best_utility=criterion))
        for name, values in (('training', training), ('development', evaluated)):
            exposures[name + '_drops'] += int(np.prod(values.shape[:-1]))
            exposures[name + '_frames'] += int(values[..., 2].sum())
        if generation % 5 == 4 or last_gain == generation or generation == len(starts) - 1:
            print('universal donor', generation + 1, 'best', round(criterion, 4), 'body validation',
                  evaluated[:, winner, :, 1].mean(axis=-1).round(2).tolist(), 'seconds', round(perf_counter()-begun, 1), flush=True)
            _write(args.output, receipt)
    receipt['training_wall_seconds'] = perf_counter() - begun
    receipt['checkpoint_sha256'] = _hash(args.checkpoint)
    receipt['model_frozen_before_held'] = True
    _write(args.output, receipt)
    return best


def _physics_audit(native, worlds):
    weights = .05 * np.random.default_rng(362271).normal(size=SHAPE)
    seeds = _seeds(7163141, len(worlds), 2)
    truth = _evaluate(worlds, (np.zeros(SHAPE), weights), seeds, split='held')
    for i, (world, seeds_i) in enumerate(zip(worlds, seeds, strict=True)):
        for j, policy in enumerate((np.zeros(SHAPE), weights)):
            for k, seed in enumerate(seeds_i):
                judge = _live(native, world.arena, policy, seed, 'changed', 'clean')['metrics']
                stats = [judge[key] for key in ('rebounds', 'longest_consecutive_rebounds', 'steps', 'paddle_contacts')]
                np.testing.assert_array_equal(truth[i, j, k], stats)
    return int(np.prod(truth.shape[:-1]))


def _test(native, worlds, current, args, receipt):
    held = _seeds(DONOR_HELD_RNG + 77791, len(worlds), 8)
    prior_receipt = _read(args.original_receipt)
    original = OriginalNative()
    try:
        with np.load(args.original_checkpoint, allow_pickle=False) as saved:
            older = saved['weights']
        if 'frozen_sha256' in prior_receipt:
            assert prior_receipt['frozen_sha256'] == _hash(args.original_checkpoint)
        receipt['original_selected_sha256'] = _hash(args.original_checkpoint)
        receipt['original_completed_rounds'] = len(prior_receipt['updates'])
        standard, special = [], []
        for world, seeds in zip(worlds, held, strict=True):
            standard.append(_paired(world.arena, native, current, seeds, ('clean', 'hold'), 'changed'))
            special.append(_paired(world.arena, original, older, seeds, ('clean',), 'changed'))
        generic_summary = [_summary(case) for case in standard]
        original_summary = [_summary(case) for case in special]
        gate = (np.mean([row['clean']['mean'] for row in generic_summary]) >= 4.
                and all(row['clean']['mean'] > row['hold']['mean'] + .40 for row in generic_summary)
                and sum(row['clean']['at_least_eight'] for row in generic_summary) >= 3)
        receipt.update(held_donor_seeds=held, held_donor=dict(universal=standard, original=special),
                       donor_summary=dict(universal=generic_summary, original=original_summary), donor_gate_passed=bool(gate))
        _write(args.output, receipt)
        print('universal vs original held donors', receipt['donor_summary'], 'gate', gate, flush=True)
        if gate:
            target = BounceArena(view='wrist', width=192, height=144)
            seeds = _seeds(STOCK_HELD_RNG, 1, 16)[0]
            try:
                learned = _paired(target, native, current, seeds, ('public', 'clean', 'hold'), 'standard')
                comparison = _paired(target, original, older, seeds, ('public', 'clean'), 'standard')
            finally:
                target.close()
            assert receipt['checkpoint_sha256'] == _hash(args.checkpoint)
            receipt['held_stock'] = dict(seeds=seeds, observer_sha256=_hash(FUSED_OUTPUT),
                                         universal=learned, original=comparison)
            receipt['stock_summary'] = dict(universal=_summary(learned), original=_summary(comparison))
            print('universal vs original held wrist', receipt['stock_summary'], flush=True)
        _write(args.output, receipt)
    finally:
        original.working.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--original-receipt', type=Path, default=ORIGINAL_RECEIPT)
    parser.add_argument('--original-checkpoint', type=Path, default=ORIGINAL_CHECKPOINT)
    parser.add_argument('--rounds', type=int, default=120)
    parser.add_argument('--directions', type=int, default=12)
    parser.add_argument('--examples', type=int, default=2)
    parser.add_argument('--train-cap', type=int, default=260)
    parser.add_argument('--noise', type=float, default=.18)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.rounds, args.directions, args.examples) < 1 or not 1 <= args.train_cap <= 300:
        parser.error('expected positive search counts and training cap up to 300')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    original = _read(args.original_receipt)
    same_budget = (args.rounds <= len(original['updates']) and all(
        original['settings'].get(key) == getattr(args, key) for key in ('directions', 'examples', 'train_cap')))
    if same_budget:
        launches = [round_['training_seeds'] for round_ in original['updates'][:args.rounds]]
    else:
        rng = np.random.default_rng(TRAIN_RNG + 19044)
        launches = rng.integers(1, 2**31 - 1, (args.rounds, len(DESIGNS), args.examples)).tolist()
    native = UniversalNative()
    worlds = [UniversalWorld(native, design) for design in DESIGNS]
    try:
        stats, neutral = calibrate(worlds, launches[0], args.train_cap)
        native.install(stats['center'], stats['scale'], stats['basis'])
        audited = _physics_audit(native, worlds)
        print('universal witness audit', audited, 'native/Python matches; unlabelled temporal eigenvalues',
              stats['shifts'].round(5).tolist(), flush=True)
        if args.smoke:
            return
        receipt = dict(schema='tt24-universal-donor-1', source_features=FEATURES, body=tuple(map(asdict, DESIGNS)),
                       sensory_channels=CHANNELS,
                       semantic_extraction_assumption='clean metric ball/paddle observation on donors',
                       feature_fit='generic asinh scaling and temporal PCA, followed by all second degree products',
                       calibration={key: value.tolist() for key, value in stats.items()},
                       normalizer_neutral_practice_seeds=launches[0], normalizer_neutral_duplicates=0,
                       matched_original_training_seed_roster=same_budget, native_physics_hash=_hash(NATIVE),
                       universal_fragment_hash=_hash(FRAGMENT), verified_native_judge_drops=audited,
                       target_training=False, teacher_actions=False)
        weights = train(worlds, launches, neutral, stats, args, receipt)
        _test(native, worlds, weights, args, receipt)
    finally:
        for world in worlds:
            world.close()
        native.working.cleanup()


if __name__ == '__main__':
    main()
