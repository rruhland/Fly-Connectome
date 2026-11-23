"""Learn a single-flight receiving goal on donors, then lock it for stock bouncing."""

import argparse
import ctypes
from dataclasses import asdict
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena, mujoco
from .tt24_donor_trial import (DESIGNS, ROOT, SHAPE, STATE, World, _checkpoint, _exploration, _live, _seeds,
                              _summary)
from .tt24_public_body import FUSED_OUTPUT


NATIVE = Path(__file__).with_name('tt24_volley_source_native.cpp')
OUTPUT = ROOT / 'tt24_volley_source.json.gz'
CHECKPOINT = ROOT / 'tt24_volley_source-frozen.npz'
REPEATED = ROOT / 'tt24_donor-refined.npz'
CONTROL_LIMIT = 66
STATS = ('reward', 'gate', 'powered_gate', 'decisions', 'physical_steps', 'encounters', 'ending',
         'max_height_from_home', 'outgoing_vx', 'paddle_upward_speed', 'legal_departure', 'x', 'y')
DEV_RNG = 1428117562
SOURCE_RNG = 763930688
STOCK_RNG = 1669117219
OPTIMIZER_RNG = 817345389


class VolleyNative:
    def __init__(self, catchable=False, two_receivers=False):
        self.catchable = catchable
        self.workspace = TemporaryDirectory(prefix='tt24_volley_source-')
        path = Path(self.workspace.name) / 'volley.so'
        package = Path(mujoco.__file__).parent
        runtime = package / f'libmujoco.so.{mujoco.__version__}'
        flags = ['-DTT24_CATCHABLE'] if catchable else []
        if two_receivers:
            flags.append('-DTT24_TWO_RECEIVERS')
        subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared', '-fPIC', *flags, '-I',
                        str(package / 'include'), str(NATIVE), str(runtime), f'-Wl,-rpath,{package}',
                        '-o', str(path)], check=True)
        self.so = ctypes.CDLL(str(path))
        p, i = ctypes.c_void_p, ctypes.c_int
        spec = {'open': ('tt24_donor_open', [p], p), 'close': ('tt24_donor_close', [p], None),
                'volley': ('tt24_volley_evaluate', [p,p,i,p,p,i,i,p], None),
                'command': ('tt24_donor_command', [p,p,p,p], None)}
        for alias, (name, args, result) in spec.items():
            function = getattr(self.so, name)
            function.argtypes, function.restype = args, result
            setattr(self, alias, function)


def _feed(seed, catchable=False):
    random = np.random.default_rng(seed ^ 9736152)
    side = random.choice((-1., 1.))
    start_x = side * random.uniform(.065, .125)
    start_y = random.uniform(-.082, .082)
    height = random.uniform(.23, .39)
    vertical = random.uniform(-.58, .24)
    if catchable:
        start_x = side * random.uniform(.040, .093)
        if random.uniform() < .66:
            height = random.uniform(.075, .180)
            vertical = random.uniform(.55, 1.58)
    arrival = (vertical + np.sqrt(vertical * vertical + 2. * 9.81 * (height - .028))) / 9.81
    aim_x = side * random.uniform(-.021, .020)
    aim_y = random.uniform(-.028, .028)
    vx = (aim_x - start_x) / arrival + random.uniform(-.045, .045)
    vy = (aim_y - start_y) / arrival + random.uniform(-.050, .050)
    return np.array((start_x, start_y, height)), np.array((vx, vy, vertical))


class VolleyWorld(World):
    def prepare(self, seeds, split):
        arena = self.arena
        model = arena._model
        states = np.empty((len(seeds), mujoco.mj_stateSize(model, STATE)))
        settings = np.empty((len(seeds), 2))
        for index, seed in enumerate(seeds):
            arena.reset(seed=int(seed), split=split, regime='changed', render=False)
            offset, velocity = _feed(int(seed), self.native.catchable)
            arena._ball_position[:] = arena._data.geom_xpos[arena._blade] + offset
            arena._ball_velocity[:] = velocity
            mujoco.mj_forward(model, arena._data)
            mujoco.mj_getState(model, arena._data, states[index], STATE)
            settings[index] = arena._drag_force_scale, arena._drop.damping
        return states, settings

    def play(self, weights, batch, cap=CONTROL_LIMIT):
        states, settings = batch
        weights = np.ascontiguousarray(weights, dtype=float).reshape(-1, *SHAPE)
        evidence = np.empty((len(weights), len(states), len(STATS)))
        self.native.volley(self.handle, weights.ctypes.data, len(weights), states.ctypes.data,
                           settings.ctypes.data, len(states), cap, evidence.ctypes.data)
        if np.any(evidence[..., 5] > 1) or np.any(evidence[..., 3] > evidence[..., 4]):
            raise ValueError('a single-flight trajectory crossed its decision or encounter boundary')
        return evidence


def _prepare(worlds, roster):
    return [world.prepare(seeds, split='held') for world, seeds in zip(worlds, roster, strict=True)]


def _source(worlds, weights, states):
    return np.stack([world.play(weights, batch) for world, batch in zip(worlds, states, strict=True)])


def _quality(result):
    returns = result[..., 0].mean(axis=-1)
    return .84 * returns.mean(axis=0) + .16 * returns.min(axis=0)


def _summary_source(result):
    result = result.reshape(-1, len(STATS))
    routes = result[:, 1] > 0
    return dict(drops=len(result), successful_routes=int(routes.sum()), powered_routes=int(result[:, 2].sum()),
                legal_departures=int(result[:, 10].sum()), mean_reward=float(result[:, 0].mean()),
                mean_inference_steps=float(result[:, 3].mean()), maximum_encounters=int(result[:, 5].max()),
                early_recontact_faults=int((result[:, 6] == 2).sum()),
                median_gate_surface_upward_speed_m_s=float(np.median(result[routes, 9])) if routes.any() else None)


def _candidates(rng, best, basis, rounds, magnitude, last_gain, count):
    groups = ([3, 6, 9, 12, 13, 14, 19], [1, 4, 7, 10, 15, 17], [2, 5, 8, 11, 16, 18], [0, 7, 8, 9, 10, 11])
    directions = []
    scaling = 1. / (1. + .011 * rounds)
    for index in range(count):
        group = index % 6
        axes = basis[:, (rounds + index) % 5]
        local = np.zeros(SHAPE[1])
        if group < len(groups):
            selected = rng.choice(groups[group], size=3, replace=False)
            local[selected] = rng.normal(size=3)
        elif group == 4:
            local[rng.integers(SHAPE[1])] = 1.
        else:
            local = rng.normal(size=SHAPE[1]) * .42
        update = np.outer(axes, local) + rng.normal(size=SHAPE) * (.11 if group < 4 else .32)
        step = rng.choice((.28, .55, 1., 1.6)) * scaling * magnitude
        if rounds - last_gain > 12:
            step *= 1.35
        directions.append(step * update)
    proposals = np.stack(directions)
    return np.clip(np.concatenate((best[None], best[None]+proposals, best[None]-proposals)), -2., 2.)


def _train(worlds, basis, args, report):
    rng = np.random.default_rng(OPTIMIZER_RNG + 1397 * args.catchable + 8337 * args.two_receivers)
    neutral = np.zeros(SHAPE)
    best, latest_gain = neutral.copy(), 0
    development = _seeds(DEV_RNG + 488 * args.catchable + 6113 * args.two_receivers, len(worlds), 10)
    frozen_starts = _prepare(worlds, development)
    zero_rows = _source(worlds, neutral, frozen_starts)
    high = float(_quality(zero_rows)[0])
    report['development_seeds'] = development
    report['stationary_development'] = zero_rows[:, 0].tolist()
    begin = perf_counter()
    for round_id in range(args.rounds):
        candidates = _candidates(rng, best, basis, round_id, args.noise, latest_gain, args.directions)
        starts = rng.integers(1, 2**31 - 1, (len(worlds), args.examples)).tolist()
        training = _source(worlds, candidates, _prepare(worlds, starts))
        rank = np.argsort(-_quality(training), kind='stable')
        shortlist = candidates[[i for i in rank if i != 0][:3]]
        validation = _source(worlds, shortlist, frozen_starts)
        scores = _quality(validation)
        winner = int(np.argmax(scores))
        if scores[winner] > high:
            best, high = shortlist[winner].copy(), float(scores[winner])
            latest_gain = round_id
            _checkpoint(args.checkpoint, best)
        report['updates'].append(dict(round=round_id+1, training_seeds=starts, training_weights=candidates.tolist(),
                                      training=training.tolist(), validation_weights=shortlist.tolist(),
                                      validation=validation.tolist(), best_development=high))
        if round_id % 5 == 4 or latest_gain == round_id:
            score = _summary_source(validation[:, winner])
            print(f'volley {round_id+1} {perf_counter()-begin:.1f}s best={high:.3f}', score, flush=True)
        if round_id % 10 == 9:
            _write(args.output, report)
    _checkpoint(args.checkpoint, best)
    exposure = dict(training_drops=0, training_decisions=0, training_physics_frames=0,
                    donor_validation_drops=int(np.prod(zero_rows.shape[:-1])),
                    donor_validation_decisions=int(zero_rows[..., 3].sum()),
                    donor_validation_physics_frames=int(zero_rows[..., 4].sum()))
    for result in report['updates']:
        for key, label in [('training', 'training'), ('validation', 'donor_validation')]:
            data = np.asarray(result[key])
            exposure[label+'_drops'] += int(np.prod(data.shape[:-1]))
            exposure[label+'_decisions'] += int(data[..., 3].sum())
            exposure[label+'_physics_frames'] += int(data[..., 4].sum())
    report |= dict(exposure=exposure, frozen_weights_sha256=_hash(args.checkpoint),
                   selected_on_donor_only=True, wall_seconds=perf_counter()-begin)
    _write(args.output, report)
    return best


def _target(native, learned, reference, launches):
    arena = BounceArena(view='wrist', width=192, height=144)
    comparisons = dict(public_volley=(learned, 'public'), public_repeat=(reference, 'public'),
                       clean_volley=(learned, 'clean'), clean_repeat=(reference, 'clean'),
                       hold=(np.zeros_like(learned), 'hold'))
    tapes = []
    try:
        for seed in launches:
            conditions = {}
            for mode, (weights, observer) in comparisons.items():
                live = _live(native, arena, weights, seed, 'standard', observer)
                replay = _live(native, arena, weights, seed, 'standard', 'replay', tape=live['actions'])
                if digest_json(live) != digest_json(replay):
                    raise ValueError(f'the transfer tape changed the physical result on {seed}')
                conditions[mode] = live
            tapes.append(dict(seed=seed, conditions=conditions))
            print('stock bounce', seed, {name: row['metrics']['longest_consecutive_rebounds']
                                        for name, row in conditions.items()}, flush=True)
    finally:
        arena.close()
    return tapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--repeat', type=Path, default=REPEATED)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--directions', type=int, default=12)
    parser.add_argument('--examples', type=int, default=2)
    parser.add_argument('--noise', type=float, default=.22)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--catchable', action='store_true', help='also receive rising feeds and limit outbound speed')
    parser.add_argument('--two-receivers', action='store_true', help='let a catchable flight reach either receiving region')
    args = parser.parse_args()
    if min(args.rounds, args.examples) < 1 or args.directions < 2:
        parser.error('need positive rounds and examples with at least two directions')
    if args.two_receivers and not args.catchable:
        parser.error('the two-receiver variant requires catchable flight speed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native = VolleyNative(args.catchable, args.two_receivers)
    worlds = [VolleyWorld(native, design) for design in DESIGNS]
    try:
        if args.preflight:
            launches = _seeds(DEV_RNG + 488 * args.catchable + 6113 * args.two_receivers, len(worlds), 10)
            rows = _source(worlds, np.zeros(SHAPE), _prepare(worlds, launches))
            print('stationary feasibility', [_summary_source(arm[0]) for arm in rows], flush=True)
            return
        basis, probes = _exploration(worlds)
        report = dict(schema='tt24-single-flight-donor-1', donors=[asdict(design) for design in DESIGNS],
                      settings={key: value for key, value in vars(args).items() if not isinstance(value, Path)},
                      initial_weights='zero', motor_teacher_examples=0, repeated_ball_source_episodes=0,
                      source_game=('receive one lateral incoming ball, route toward either upward receiver'
                                   if args.two_receivers else
                                   'receive one lateral incoming ball, redirect upward toward entry side'),
                      gate=dict(height_m=.118 if args.catchable else .165,
                                signed_lateral_centre_m=.057 if args.catchable else .080,
                                lateral_half_width_m=.043 if args.catchable else .055,
                                transversal_half_width_m=.070, minimum_receiving_direction_speed_m_s=.045,
                                upward_crossing_speed_m_s=(.16, 1.45 if args.two_receivers else 1.18)
                                if args.catchable else None,
                                powered_paddle_point_upward_speed_m_s=.055),
                      source_feedback='first outbound gate acceptance with flight height, direction, proximity, and lift score',
                      source_observer='clean simulator metric ball and paddle geometry/motion',
                      acting_boundary='no new pupil decisions after first paddle encounter; referee only until gate or apex',
                      stats=STATS, model_code_sha256=_hash(NATIVE), posture_exploration_probes=probes,
                      constant_joint_exploration_basis=basis.tolist(), updates=[])
        _write(args.output, report)
        model = _train(worlds, basis, args, report)
        held = _seeds(SOURCE_RNG + 2791 * args.catchable + 14111 * args.two_receivers, len(worlds), 24)
        outcomes = _source(worlds, np.stack((model, np.zeros_like(model))), _prepare(worlds, held))
        course = {name: [_summary_source(arm[index]) for arm in outcomes]
                  for index, name in enumerate(('live', 'hold'))}
        report['source_held'] = dict(seeds=held, outcomes=outcomes.tolist(), summary=course)
        gate_measure = 'powered_routes' if args.two_receivers else 'successful_routes'
        pass_gate = (sum(row['powered_routes'] for row in course['live']) >= 12 and
                     all(active[gate_measure] >= idle[gate_measure] + 2
                         for active, idle in zip(course['live'], course['hold'], strict=True)))
        report['source_gate_passed'] = pass_gate
        _write(args.output, report)
        print('locked source held', course, 'gate', pass_gate, flush=True)
        if pass_gate:
            with np.load(args.repeat) as archive:
                repeated = archive['weights'].copy()
            stock = _seeds(STOCK_RNG + 8671 * args.catchable + 53033 * args.two_receivers, 1, 16)[0]
            cases = _target(native, model, repeated, stock)
            report['stock'] = dict(seeds=stock, cases=cases, summary=_summary(cases), motor_practice_drops=0,
                                   observer_sha256=_hash(FUSED_OUTPUT), repeated_policy_sha256=_hash(args.repeat),
                                   verified_tapes=len(cases)*len(cases[0]['conditions']))
            if _hash(args.checkpoint) != report['frozen_weights_sha256']:
                raise ValueError('the stock assessment changed the single-flight readout')
            _write(args.output, report)
            print('locked stock', report['stock']['summary'], flush=True)
    finally:
        for world in worlds:
            world.close()
        native.workspace.cleanup()


if __name__ == '__main__':
    main()
