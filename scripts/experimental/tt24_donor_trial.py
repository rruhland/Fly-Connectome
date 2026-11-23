"""Reward-only physical hinged donor training and frozen stock-arm deployment."""

import argparse
import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena, mujoco
from .tt24_bounce_ident import CleanState
from .tt24_public_body import FUSED_OUTPUT, FUSED_RECEIPT, load_public


ROOT = Path('artifacts/tt24')
NATIVE = Path(__file__).with_name('tt24_donor_native.cpp')
RECEIPT = ROOT / 'tt24_donor.json.gz'
CHECKPOINT = ROOT / 'tt24_donor-frozen.npz'
FEATURES = 20
SHAPE = (5, FEATURES)
TRAIN_RNG = 1755261883
DEV_RNG = 1091881107
DONOR_HELD_RNG = 859030807
STOCK_HELD_RNG = 1618784699
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True)
class Design:
    elbow_axis_xy: tuple[float, float]
    upper_mass: float
    camera_mass: float
    drive: float
    force: float


DESIGNS = (
    Design((.075, -.055), .91, .030, 1.08, 1.04),
    Design((.100, -.080), 1.04, .042, .99, .96),
    Design((.132, -.105), 1.15, .052, .93, .91),
)


class DonorBounce(BounceArena):
    """Three physically different hinged plants, all outside the stock target."""

    def __init__(self, design, *, max_steps=300, view='none', width=96, height=72):
        self.design = design
        self._changed_drive = BounceArena._changed_drive * np.r_[[design.drive] * 5, 1.]
        self._changed_force = BounceArena._changed_force * np.r_[[design.force] * 5, 1.]
        super().__init__(arm='alternate', view=view, width=width, height=height, max_steps=max_steps)

    def _document(self, width, height, detail, arm):
        doc = super()._document(width, height, detail, arm)
        upper = doc.find(".//body[@name='learner_upper_arm']")
        joint = upper.find("body[@name='learner_elbow']/joint")
        x, y = self.design.elbow_axis_xy
        joint.set('axis', f'{x} {y} {np.sqrt(1. - x*x - y*y)}')
        inertia = upper.find('inertial')
        inertia.set('mass', str(float(inertia.get('mass')) * self.design.upper_mass))
        values = np.fromstring(inertia.get('fullinertia'), sep=' ') * self.design.upper_mass
        inertia.set('fullinertia', ' '.join(map(str, values)))
        doc.find(".//geom[@name='tt24_payload_housing']").set('mass', str(self.design.camera_mass))
        return doc


class Native:
    def __init__(self):
        self.working = TemporaryDirectory(prefix='tt24_donor-')
        library = Path(self.working.name) / 'donor.so'
        package = Path(mujoco.__file__).parent
        engine = package / f'libmujoco.so.{mujoco.__version__}'
        subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared', '-fPIC', '-I',
                        str(package / 'include'), str(NATIVE), str(engine), f'-Wl,-rpath,{package}',
                        '-o', str(library)], check=True)
        self.so = ctypes.CDLL(str(library))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        signatures = {
            'open': ([pointer], pointer), 'close': ([pointer], None),
            'evaluate': ([pointer, pointer, integer, pointer, pointer, integer, integer, pointer], None),
            'command': ([pointer, pointer, pointer, pointer], None),
        }
        for name, (inputs, output) in signatures.items():
            interface = getattr(self.so, f'tt24_donor_{name}')
            interface.argtypes, interface.restype = inputs, output
            setattr(self, name, interface)


class World:
    def __init__(self, native, design):
        self.native = native
        self.arena = DonorBounce(design)
        self.handle = native.open(self.arena._model._address)

    def prepare(self, seeds, split):
        model = self.arena._model
        states = np.empty((len(seeds), mujoco.mj_stateSize(model, STATE)))
        physics = np.empty((len(seeds), 2))
        for index, seed in enumerate(seeds):
            self.arena.reset(seed=int(seed), split=split, regime='changed', render=False)
            mujoco.mj_getState(model, self.arena._data, states[index], STATE)
            physics[index] = self.arena._drag_force_scale, self.arena._drop.damping
        return states, physics

    def play(self, weights, batch, cap=300):
        states, physics = batch
        weights = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1, *SHAPE)
        result = np.empty((len(weights), len(states), 4))
        self.native.evaluate(self.handle, weights.ctypes.data, len(weights), states.ctypes.data,
                             physics.ctypes.data, len(states), cap, result.ctypes.data)
        return result

    def close(self):
        self.native.close(self.handle)
        self.arena.close()


def _seeds(source, rows, n):
    return np.random.default_rng(source).integers(1, 2**31 - 1, (rows, n)).tolist()


def _utility(results, cap):
    body_returns = np.mean(results[..., 0] + .25 * results[..., 2] / cap, axis=-1)
    return .8 * body_returns.mean(axis=0) + .2 * body_returns.min(axis=0)


def _evaluate(worlds, options, launches, *, split, cap=300):
    return np.asarray([world.play(options, world.prepare(seeds, split), cap=cap)
                       for world, seeds in zip(worlds, launches, strict=True)])


def _checkpoint(path, weights):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(pending, weights=weights)
    pending.replace(path)


def _train(worlds, receipt, args):
    rng = np.random.default_rng(TRAIN_RNG)
    zero = np.zeros(SHAPE)
    best, score = zero.copy(), None
    dev = _seeds(DEV_RNG, len(worlds), 5)
    development = [world.prepare(seeds, 'practice') for world, seeds in zip(worlds, dev, strict=True)]
    score_rows = lambda opts: np.asarray([w.play(opts, batch) for w, batch in zip(worlds, development, strict=True)])
    neutral = score_rows(zero)
    score = float(_utility(neutral, 300)[0])
    receipt['dev_seeds'] = dev
    receipt['development_is_only_on_donors'] = True
    receipt['neutral_development'] = neutral[:, 0].tolist()
    _write(args.output, receipt)
    begin = perf_counter()
    centre = best.copy()
    scales = np.array((.70, 1., 1., 1.4, 1., 1., 1.4, .70, .70, .82,
                       .77, .77, 1.08, .95, .82, .84, .84, .74, .74, 1.15))
    for generation in range(args.rounds):
        fraction = generation / max(1, args.rounds - 1)
        step = args.noise * (1. - .65 * fraction)
        directions = rng.normal(size=(args.directions, *SHAPE)) * scales
        # Rank-one edits explore coupled motor effects both in the retina and
        # across joints. Full-rank changes remain available in each update.
        for idx in range(1, args.directions, 2):
            directions[idx] = rng.normal(size=(5, 1)) * rng.normal(size=(1, FEATURES)) * scales
        shifted = step * directions
        options = np.concatenate((centre[None], centre[None] + shifted, centre[None] - shifted))
        launches = rng.integers(1, 2**31 - 1, (len(worlds), args.examples)).tolist()
        training = _evaluate(worlds, options, launches, split='practice', cap=args.train_cap)
        fitness = _utility(training, args.train_cap)
        elite = np.argsort(-fitness, kind='stable')[:3]
        positive, negative = fitness[1:1+args.directions], fitness[1+args.directions:]
        ordering = np.argsort(-np.maximum(positive, negative), kind='stable')[:max(2, args.directions // 2)]
        precision = max(.20, np.std(np.r_[positive[ordering], negative[ordering]]))
        gradient = np.mean(((positive - negative)[ordering] / precision)[:, None, None] * directions[ordering], axis=0)
        proposal = np.clip(.42 * centre + .58 * np.mean(options[elite], axis=0) + .60 * step * gradient, -2., 2.)
        tested = np.stack((options[elite[0]], proposal))
        evaluated = score_rows(tested)
        dev_scores = _utility(evaluated, 300)
        winner = int(np.argmax(dev_scores))
        if dev_scores[winner] > score:
            best, score = tested[winner].copy(), float(dev_scores[winner])
            _checkpoint(args.checkpoint, best)
        centre = best.copy() if generation % 4 == 3 else tested[winner].copy()
        update = dict(generation=generation + 1, training_seeds=launches, weights=options.tolist(),
                      training_results=training.tolist(), development_weights=tested.tolist(),
                      development=evaluated.tolist(), best_donor_utility=score)
        receipt['updates'].append(update)
        elapsed = perf_counter() - begin
        print(f'donor {generation+1} elapsed={elapsed:.1f}s best={score:.3f}',
              np.round(evaluated[:, winner, :, 1].mean(axis=-1), 2).tolist(), flush=True)
        _write(args.output, receipt)
    _seal(receipt, args, best, neutral, begin)
    return best


def _seal(receipt, args, best, neutral, begin):
    _checkpoint(args.checkpoint, best)
    receipt['training_wall_seconds'] = perf_counter() - begin
    receipt['exposure'] = dict(training_drops=sum(np.prod(np.shape(row['training_results'])[:-1]).item()
                                                   for row in receipt['updates']),
                               training_frames=sum(int(np.asarray(row['training_results'])[..., 2].sum())
                                                   for row in receipt['updates']),
                               donor_validation_drops=int(np.prod(neutral.shape[:-1])) +
                                   sum(int(np.prod(np.shape(row['development'])[:-1])) for row in receipt['updates']),
                               donor_validation_frames=int(neutral[..., 2].sum()) +
                                   sum(int(np.asarray(row['development'])[..., 2].sum()) for row in receipt['updates']))
    receipt['frozen_sha256'] = _hash(args.checkpoint)
    receipt['model_selected_before_held'] = True
    _write(args.output, receipt)


def _exploration(worlds):
    effects, history = [], []
    scales = np.array((.04, .04, .04, .14, .14))
    for number, world in enumerate(worlds):
        arena = world.arena
        probes = []
        for axis in range(5):
            outcomes = []
            for sign in (-1., 1.):
                seed = 1916070932 + number
                obs = arena.reset(seed=seed, regime='changed', render=False)
                read = CleanState(arena)
                origin = read.observe(obs)[6:]
                action = np.eye(5)[axis] * sign * .26
                for _ in range(5):
                    obs, _, done = arena.step(action, render=False)
                    if done:
                        raise ValueError('the independently timed donor motor probe ended early')
                move = (read.observe(obs)[6:] - origin) / scales
                outcomes.append(move)
                probes.append(dict(seed=seed, action=action.tolist(), frames=5, normalized_change=move.tolist()))
            effects.append((outcomes[1] - outcomes[0]) / 2.)
        history.append(probes)
    response = np.asarray(effects).reshape(len(worlds), 5, 5).mean(axis=0).T
    regularization = .08
    basis = np.linalg.solve(response.T @ response + regularization**2 * np.eye(5), response.T)
    basis /= np.linalg.norm(basis, axis=0)
    return basis, history


def _refine(worlds, receipt, args):
    rng = np.random.default_rng(TRAIN_RNG + 19044)
    zero = np.zeros(SHAPE)
    dev = _seeds(DEV_RNG + 11117, len(worlds), 8)
    development = [world.prepare(seeds, 'held') for world, seeds in zip(worlds, dev, strict=True)]

    def check(weights):
        return np.asarray([w.play(weights, batch) for w, batch in zip(worlds, development, strict=True)])

    def utility(scores, cap):
        # Favor repeating qualifying impacts rather than separated contacts.
        blended = scores.copy()
        blended[..., 0] = .55 * scores[..., 0] + .45 * scores[..., 1]
        return _utility(blended, cap)

    neutral = check(zero)
    score = float(utility(neutral, 300)[0])
    best = zero.copy()
    basis, probe_rows = _exploration(worlds)
    receipt['dev_seeds'] = dev
    receipt['development_is_only_on_donors'] = True
    receipt['neutral_development'] = neutral[:, 0].tolist()
    receipt['non_task_donor_motor_probes'] = probe_rows
    receipt['exploration_basis'] = basis.tolist()
    begin = perf_counter()
    last_gain = 0
    vertical = [3, 6, 9, 12, 13, 14, 19]
    horizontal = [1, 2, 4, 5, 7, 8, 15, 16, 17, 18]
    pose = [0, 7, 8, 9, 10, 11]
    for generation in range(args.rounds):
        directions = np.empty((args.directions, *SHAPE))
        cooling = 1. / (1. + .013 * generation)
        for index in range(args.directions):
            group = index % 6
            axes = basis[:, (generation + index) % 5]
            sensory = np.zeros(FEATURES)
            if group in (0, 1):
                sensory[rng.choice(vertical, size=3, replace=False)] = rng.normal(size=3)
            elif group == 2:
                sensory[rng.choice(horizontal, size=3, replace=False)] = rng.normal(size=3)
            elif group == 3:
                sensory[pose] = rng.normal(size=len(pose))
            elif group == 4:
                sensory[rng.integers(FEATURES)] = rng.choice((-1., 1.))
            else:
                sensory[:] = rng.normal(size=FEATURES) * .35
            direction = np.outer(axes, sensory)
            direction += rng.normal(size=SHAPE) * (.12 if group < 4 else .35)
            spread = rng.choice((.24, .48, 1., 1.5)) * args.noise
            if generation - last_gain > 15 and group < 2:
                spread *= 1.5
            directions[index] = direction * spread * cooling
        options = np.clip(np.concatenate((best[None], best[None] + directions, best[None] - directions)), -2., 2.)
        launches = rng.integers(1, 2**31 - 1, (len(worlds), args.examples)).tolist()
        training = _evaluate(worlds, options, launches, split='held', cap=args.train_cap)
        rank = np.argsort(-utility(training, args.train_cap), kind='stable')
        # Keep demonstrating held-distribution competence on a separate donor
        # validation roster while limiting expensive close comparisons.
        leaders = [idx for idx in rank if idx != 0][:3]
        validated = options[leaders]
        assessed = check(validated)
        performance = utility(assessed, 300)
        winner = int(np.argmax(performance))
        if performance[winner] > score:
            best, score = validated[winner].copy(), float(performance[winner])
            last_gain = generation
            _checkpoint(args.checkpoint, best)
        receipt['updates'].append(dict(generation=generation + 1, training_seeds=launches, weights=options.tolist(),
                                       training_results=training.tolist(), development_weights=validated.tolist(),
                                       development=assessed.tolist(), best_donor_utility=score))
        if last_gain == generation or generation % 5 == 4 or generation + 1 == args.rounds:
            print(f'donor refine {generation+1} elapsed={perf_counter()-begin:.1f}s best={score:.3f}',
                  np.round(assessed[:, winner, :, 1].mean(axis=-1), 2).tolist(), flush=True)
        if generation % 5 == 4:
            _write(args.output, receipt)
    _seal(receipt, args, best, neutral, begin)
    return best


def _live(native, arena, weights, seed, regime, sense, *, tape=None):
    obs = arena.reset(seed=seed, split='held', regime=regime, render=sense == 'public')
    observer = load_public(FUSED_OUTPUT) if sense == 'public' else CleanState(arena) if sense == 'clean' else None
    weights = np.ascontiguousarray(weights, dtype=float)
    issued = np.zeros(5)
    action = np.zeros(5)
    actions = []
    for tick in range(arena.max_steps):
        if observer is not None:
            state = np.ascontiguousarray(observer.observe(obs), dtype=float)
            native.command(state.ctypes.data, weights.ctypes.data, issued.ctypes.data, action.ctypes.data)
        elif tape is not None:
            action = np.asarray(tape[tick])
        actions.append(action.tolist())
        obs, _, done = arena.step(action, render=sense == 'public')
        if tape is not None and bool(done) != (tick + 1 == len(tape)):
            raise ValueError(f'action replay changed the terminal tick on {seed}')
        if done:
            break
    return dict(metrics=arena.judge_metrics(), actions=actions)


def _paired(arena, native, weights, seeds, modes, regime):
    result = []
    for seed in seeds:
        items = {}
        for mode in modes:
            current = _live(native, arena, weights, seed, regime, mode)
            replay = _live(native, arena, weights, seed, regime, 'replay', tape=current['actions'])
            if digest_json(current) != digest_json(replay):
                raise ValueError(f'the verified {mode} tape changed actual collisions on seed {seed}')
            items[mode] = current
        result.append(dict(seed=seed, conditions=items))
    return result


def _summary(cases):
    outcomes = {}
    for mode in cases[0]['conditions']:
        rows = [row['conditions'][mode] for row in cases]
        streaks = np.array([row['metrics']['longest_consecutive_rebounds'] for row in rows])
        moves = np.concatenate([row['actions'] for row in rows])
        outcomes[mode] = dict(drops=len(rows), mean=float(streaks.mean()), consecutive=streaks.tolist(),
                              maximum=int(streaks.max()), at_least_eight=int((streaks >= 8).sum()),
                              at_least_ten=int((streaks >= 10).sum()),
                              active_frames_by_joint=np.sum(abs(moves) > 1e-4, axis=0).tolist())
    return outcomes


def _native_audit(native, worlds):
    rng = np.random.default_rng(968324194)
    options = np.stack((np.zeros(SHAPE), .10 * rng.normal(size=SHAPE)))
    launches = _seeds(516733408, len(worlds), 2)
    checked = _evaluate(worlds, options, launches, split='held')
    for arm, (world, seeds) in enumerate(zip(worlds, launches, strict=True)):
        for ix, weights in enumerate(options):
            for j, seed in enumerate(seeds):
                outcome = _live(native, world.arena, weights, seed, 'changed', 'clean')['metrics']
                comparable = [outcome[key] for key in ('rebounds', 'longest_consecutive_rebounds', 'steps', 'paddle_contacts')]
                if not np.array_equal(checked[arm, ix, j], comparable):
                    raise ValueError(f'native/Python physics judge mismatch: {checked[arm, ix, j]} vs {comparable}')
    return dict(drops=int(np.prod(checked.shape[:-1])), launch_seeds=launches, comparisons=checked.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--rounds', type=int, default=32)
    parser.add_argument('--directions', type=int, default=6)
    parser.add_argument('--examples', type=int, default=1)
    parser.add_argument('--noise', type=float, default=.18)
    parser.add_argument('--train-cap', type=int, default=220)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--donor-only', action='store_true')
    parser.add_argument('--refine', action='store_true', help='use new mentor-free measured motor exploration')
    args = parser.parse_args()
    if min(args.rounds, args.directions, args.examples) < 1 or not 1 <= args.train_cap <= 300:
        parser.error('expected positive search budget and training cutoff in [1,300]')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native = Native()
    worlds = [World(native, design) for design in DESIGNS]
    try:
        audit = _native_audit(native, worlds)
        print('native/Python audited drops', audit['drops'], flush=True)
        if args.smoke:
            return
        meta = dict(schema='tt24-native-hinged-donors-1', donors=[asdict(design) for design in DESIGNS],
                    train_rng=TRAIN_RNG, settings={k: v for k, v in vars(args).items() if not isinstance(v, Path)},
                    initial_weights='all zero', observations='privileged donor metric ball, blade pose, and ball motion',
                    training_feedback='qualified rebound count, with a 0.25-frame-fraction survival tiebreak',
                    common_action='five direct normalized joint rates, all coefficients learned from donor reward',
                    controller_features=FEATURES, native_code_sha256=_hash(NATIVE), stock_motor_practice_drops=0,
                    model_action_labels=0, stock_observer_training='eight prior stereo RGB/encoder calibration drops',
                    native_judge_equivalence=audit, updates=[])
        weights = (_refine if args.refine else _train)(worlds, meta, args)
        held = _seeds(DONOR_HELD_RNG + int(args.refine) * 77791, len(worlds), 8)
        groups = [_paired(world.arena, native, weights, launches, ('clean', 'hold'), 'changed')
                  for world, launches in zip(worlds, held, strict=True)]
        meta['held_donor'] = groups
        donor = [_summary(rows) for rows in groups]
        meta['held_donor_summary'] = donor
        gate = (np.mean([row['clean']['mean'] for row in donor]) >= 4. and
                all(row['clean']['mean'] > row['hold']['mean'] + .40 for row in donor) and
                sum(row['clean']['at_least_eight'] for row in donor) >= 3)
        meta['donor_gate_passed'] = bool(gate)
        meta['verified_tapes'] = sum(2 * len(group) for group in groups)
        _write(args.output, meta)
        print('held donor', donor, 'gate', gate, flush=True)
        if gate and not args.donor_only:
            arena = BounceArena(view='wrist', width=192, height=144)
            try:
                launches = _seeds(STOCK_HELD_RNG, 1, 16)[0]
                meta['stock_observer_sha256'] = _hash(FUSED_OUTPUT)
                meta['stock_observer_receipt_sha256'] = _hash(FUSED_RECEIPT)
                meta['held_stock'] = _paired(arena, native, weights, launches, ('public', 'clean', 'hold'), 'standard')
                meta['held_stock_summary'] = _summary(meta['held_stock'])
                meta['verified_tapes'] += 3 * len(launches)
            finally:
                arena.close()
            if _hash(args.checkpoint) != meta['frozen_sha256']:
                raise ValueError('checking the stock target changed the donor readout')
            _write(args.output, meta)
            print('held stock', meta['held_stock_summary'], flush=True)
    finally:
        for world in worlds:
            world.close()
        native.working.cleanup()


if __name__ == '__main__':
    main()
