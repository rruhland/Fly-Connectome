"""Use learned native outcome credit to screen continuous arm policies."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from multiprocessing import get_context
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena, mujoco
from .tt24_donor_trial import DESIGNS, NATIVE, ROOT, SHAPE, Native, World, _exploration, _live
from .tt24_joint_pong import DONOR, WARM, WIDTH, arcade
from .tt24_public_body import FUSED_OUTPUT


TAG = 'tt24_outcome_pong'
PLAN = ROOT / f'{TAG}-plan.json'
FIT = ROOT / f'{TAG}-fit.json.gz'
OUTPUT = ROOT / f'{TAG}.json.gz'
MODES = ('native', 'shuffle', 'geometry', 'reward')
GRAVITY = (.76, 1.23, .88)
PRACTICE_CAP = 47
HELD_CAP = 240
PROPOSALS = 19
SOURCE_STARTS = (48, 32)
SOURCE_ACTORS = 16


def register():
    rng = np.random.default_rng(81318645)
    schedules = []
    for grav in GRAVITY:
        schedules.append(dict(gravity=grav, rng=int(rng.integers(2**31)),
                              practice=rng.integers(1, 2**31-1, (3, 2)).tolist(),
                              held_donor=rng.integers(1, 2**31-1, (3, 5)).tolist(),
                              held_wrist=rng.integers(1, 2**31-1, 4).tolist()))
    plan = dict(schema='tt24-outcome-pong-1', modes=list(MODES), donor_sha256=_hash(DONOR),
                observer_sha256=_hash(FUSED_OUTPUT), native_parent_sha256=_hash(WARM), donor_native_sha256=_hash(NATIVE),
                source_starts=[rng.integers(1, 2**31-1, n).tolist() for n in SOURCE_STARTS],
                source_actor_count=SOURCE_ACTORS, schedules=schedules, practice_horizon=PRACTICE_CAP,
                held_horizon=HELD_CAP, proposals=PROPOSALS, bonus=.85, opening_range_blade_radii=1.55,
                primary='mean independent wrist longest consecutive physical rebounds at equal practice exposure')
    if PLAN.exists():
        if _read(PLAN) != plan:
            raise ValueError('the locked outcome-credit design changed')
    else:
        _write(PLAN, plan)
    return plan


class Witness:
    def __init__(self, seeds):
        self.seeds = seeds
        self.armed = np.zeros(SOURCE_ACTORS * len(seeds), dtype=bool)
        self.frame = np.zeros((len(self.armed), 4))
        self.events = []

    def sample(self, paddle, paddle_speed, ball, velocity, observed):
        depth = (ball[:, 0] - 1.5 / WIDTH) / .10
        ready = observed & (depth > .45) & (depth <= 1.55) & (velocity[:, 0] < -.14) & ~self.armed
        gap = (ball[ready, 1] - paddle[ready]) / .10
        drift = (velocity[ready, 1] - paddle_speed[ready]) / .55
        self.frame[ready] = np.column_stack((gap**2, drift**2, gap * drift, -velocity[ready, 0] / .55))
        self.armed[ready] = True

    def outcome(self, contacts, score, active):
        resolved = np.flatnonzero(self.armed & active & ((contacts > 0) | (score < 0)))
        for index in resolved:
            self.events.append([int(index // len(self.seeds)), self.seeds[index % len(self.seeds)],
                                float(contacts[index] > 0), *self.frame[index].tolist()])
        self.armed[resolved] = False


def feature(raw):
    gap, drift, derivative, normal = np.clip(np.asarray(raw), [-0.01, -.01, -8., 0.], [16., 9., 8., 3.]).T
    return np.column_stack((gap, drift, derivative, normal, gap * normal, derivative * normal, gap**2))


def rank_auc(values, success):
    wins = success.astype(bool)
    count = wins.sum()
    return float((rankdata(values)[wins].sum() - count * (count + 1) / 2) / (count * (~wins).sum()))


def learn(plan):
    if FIT.exists():
        result = _read(FIT)
        if result['plan'] != _hash(PLAN):
            raise ValueError('cached outcome fit differs')
        return result
    rng = np.random.default_rng(62170943)
    learned = np.array(_read(WARM)['weights'])
    fixed_actors = rng.normal(0., .54, (SOURCE_ACTORS, len(learned)))
    fixed_actors[::2] += learned
    fixed_actors[:2] = [np.zeros_like(learned), learned]
    tapes = []
    point_frames = []
    for starts in plan['source_starts']:
        witness = Witness(starts)
        outcome = arcade(fixed_actors, starts, witness=witness)
        tapes.append(np.array(witness.events))
        point_frames.append(int(outcome[..., 2].sum()))
    train, held = tapes
    x, v = feature(train[:, 3:]), feature(held[:, 3:])
    rank, reference = train[:, 2], held[:, 2]
    center, spread = x.mean(axis=0), np.maximum(.1, x.std(axis=0))
    y = np.c_[np.ones(len(x)), (x - center) / spread]
    q = np.c_[np.ones(len(v)), (v - center) / spread]
    shuffled = rank.copy()
    for actor in np.unique(train[:, 0]):
        group = train[:, 0] == actor
        shuffled[group] = rng.permutation(rank[group])
    models = {}
    audit = dict(geometry=rank_auc(-held[:, 3], reference))
    for mode, reward in (('native', rank), ('shuffle', shuffled)):
        w = np.zeros(y.shape[1])
        ridge = np.eye(y.shape[1]) * 5.
        for _ in range(12):
            fitted = expit(y @ w)
            gradient = y.T @ (reward - fitted) - ridge @ w
            response = y.T @ (y * (fitted * (1. - fitted))[:, None]) + ridge
            w += np.linalg.solve(response, gradient)
        models[mode] = w.tolist()
        audit[mode] = rank_auc(q @ w, reference)
    output = dict(plan=_hash(PLAN), center=center.tolist(), scale=spread.tolist(), policy=models,
                  held_auc=audit, tapes=[a.tolist() for a in tapes], source_frames=point_frames,
                  played_actors=fixed_actors.tolist(), counts=[[len(a), int(a[:, 2].sum())] for a in tapes],
                  gate=audit['native'] >= .68 and audit['native'] >= audit['shuffle'] + .03)
    _write(FIT, output)
    print('actual native outcome cue', output['counts'], audit, 'gate', output['gate'], flush=True)
    return output


class WitnessKernel(Native):
    """Keep the physical native judge and record pre-impact features alongside it."""

    def __init__(self):
        self.working = TemporaryDirectory(prefix='tt24-outcome-')
        path = Path(self.working.name)
        package = Path(mujoco.__file__).parent
        engine = package / f'libmujoco.so.{mujoco.__version__}'
        code = NATIVE.read_text()
        code = code.replace('Judge judge;', 'Judge judge; double w[5] = {}; double previous[2] = {};')
        marker = '            command(state, weights, issued, action);'
        extra = '''            if (!w[4] && tick && state[2] - state[8] <= 1.55*.076 && state[5] < -.27) {
                const double x = (state[0]-state[6])/.076, y = (state[1]-state[7])/.076;
                const double u = (state[3]-(state[6]-previous[0])*60.)/1.5;
                const double v = (state[4]-(state[7]-previous[1])*60.)/1.5;
                w[0] = x*x+y*y; w[1] = u*u+v*v; w[2] = x*u+y*v; w[3] = -state[5]/1.5; w[4] = 1.;
            }
            previous[0] = state[6]; previous[1] = state[7];
'''
        if code.count(marker) != 1:
            raise ValueError('physical witness insertion is ambiguous')
        code = code.replace(marker, extra + marker)
        code = code.replace('result[3] = judge.contacts;', 'result[3] = judge.contacts; std::copy(w, w+5, result+4);')
        stride = 'outputs + 4 * (policy * starts + start)'
        if code.count(stride) != 1:
            raise ValueError('physical witness scorer dimensions changed')
        code = code.replace(stride, 'outputs + 9 * (policy * starts + start)')
        source, library = path / 'world.cpp', path / 'world.so'
        source.write_text(code)
        subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared', '-fPIC', '-I',
                        str(package / 'include'), str(source), str(engine), f'-Wl,-rpath,{package}', '-o', str(library)], check=True)
        self.so = ctypes.CDLL(str(library))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        for name, (inputs, output) in {'open': ([ptr], ptr), 'close': ([ptr], None),
                                       'evaluate': ([ptr, ptr, integer, ptr, ptr, integer, integer, ptr], None),
                                       'command': ([ptr, ptr, ptr, ptr], None)}.items():
            fn = getattr(self.so, f'tt24_donor_{name}')
            fn.argtypes, fn.restype = inputs, output
            setattr(self, name, fn)


class WitnessWorld(World):
    def play(self, weights, batch, cap=300):
        states, physics = batch
        weights = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1, *SHAPE)
        result = np.empty((len(weights), len(states), 9))
        self.native.evaluate(self.handle, weights.ctypes.data, len(weights), states.ctypes.data,
                             physics.ctypes.data, len(states), cap, result.ctypes.data)
        return result


def courses(kernel, gravity):
    worlds = [WitnessWorld(kernel, design) for design in DESIGNS]
    for world in worlds:
        world.arena._model.opt.gravity[:] *= gravity
    return worlds


def selector(fit, result):
    witness = result[..., 4:8].reshape(-1, 4)
    x = np.c_[np.ones(len(witness)), (feature(witness)-fit['center']) / fit['scale']]
    shape = result.shape[:-1]
    valid = result[..., 8] > 0
    bonus = {mode: np.where(valid, expit(x @ fit['policy'][mode]).reshape(shape), 0.).mean(axis=(0, 2))
             for mode in ('native', 'shuffle')}
    bonus['geometry'] = np.where(valid, expit(2.5 * (1. - np.sqrt(result[..., 4]))), 0.).mean(axis=(0, 2))
    target = result[..., 1].mean(axis=(0, 2))
    paid = {mode: target + .85 * (rankdata(values) - 1.) / (len(values) - 1.) for mode, values in bonus.items()}
    paid['reward'] = target
    return {mode: int(np.argmax(values)) for mode, values in paid.items()}, bonus, target


def tested(task):
    roster, fit = task
    native = WitnessKernel()
    worlds = courses(native, roster['gravity'])
    wrist = BounceArena(view='wrist', width=192, height=144, max_steps=HELD_CAP)
    wrist._model.opt.gravity[:] *= roster['gravity']
    try:
        with np.load(DONOR, allow_pickle=False) as file:
            baseline = file['weights']
        basis, probes = _exploration(worlds)
        rng = np.random.default_rng(roster['rng'])
        shifts = []
        for index in range((PROPOSALS - 1) // 2):
            basis_vector = basis[:, index % basis.shape[1]]
            sensory = rng.normal(size=SHAPE[1])
            J = np.outer(basis_vector, sensory) + .19 * rng.normal(size=SHAPE)
            shifts.append(.43 * J / np.linalg.norm(J))
        proposals = np.clip(np.r_[baseline[None], baseline + shifts, baseline - np.asarray(shifts)], -2., 2.)
        proposals = proposals[rng.permutation(len(proposals))]

        def physical(weights, seeds, cap):
            return np.array([w.play(weights, w.prepare(s, 'held'), cap) for w, s in zip(worlds, seeds, strict=True)])

        practice = physical(proposals, roster['practice'], PRACTICE_CAP)
        selected, bonus, direct = selector(fit, practice)
        policies = np.array([proposals[selected[mode]] for mode in MODES])
        results = physical(policies, roster['held_donor'], HELD_CAP)
        wrist_runs = {}
        for mode, policy in zip(MODES, policies, strict=True):
            rows = []
            for seed in roster['held_wrist']:
                played = _live(native, wrist, policy, seed, 'changed', 'public')
                replay = _live(native, wrist, policy, seed, 'changed', 'replay', tape=played['actions'])
                if played != replay:
                    raise ValueError('original wrist judge changed in a witness-score test')
                rows.append(played)
            wrist_runs[mode] = rows
        print('outcome screen', roster['gravity'], selected, 'held wrist',
              {mode: [a['metrics']['longest_consecutive_rebounds'] for a in rows] for mode, rows in wrist_runs.items()}, flush=True)
        return dict(gravity=roster['gravity'], chosen=selected, probes=probes, practice=practice.tolist(),
                    bonuses={n: a.tolist() for n, a in bonus.items()}, direct=direct.tolist(), policies=policies.tolist(),
                    held_donor=results.tolist(), held_wrist=wrist_runs)
    finally:
        for world in worlds:
            world.close()
        wrist.close()
        native.working.cleanup()


def run(plan, fit, workers):
    receipt = (_read(OUTPUT) if OUTPUT.exists() else
               dict(plan=_hash(PLAN), fit=_hash(FIT), trials=[]))
    if receipt['plan'] != _hash(PLAN) or receipt['fit'] != _hash(FIT):
        raise ValueError('outcome-credit lineage failed')
    tasks = [(row, fit) for row in plan['schedules'][len(receipt['trials']):]]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for row in pool.map(tested, tasks):
            receipt['trials'].append(row)
            _write(OUTPUT, receipt)
    receipt['summary'] = {mode: dict(wrist=[sum(x['metrics']['longest_consecutive_rebounds'] for x in r['held_wrist'][mode])
                                           for r in receipt['trials']],
                                    donor=[float(np.asarray(r['held_donor'])[:, index, :, 1].mean()) for r in receipt['trials']],
                                    contacts=[sum(x['metrics']['paddle_contacts'] for x in r['held_wrist'][mode])
                                              for r in receipt['trials']],
                                    qualified=[sum(x['metrics']['rebounds'] for x in r['held_wrist'][mode])
                                               for r in receipt['trials']],
                                    wrist_eight=[sum(x['metrics']['longest_consecutive_rebounds'] >= 8
                                                     for x in r['held_wrist'][mode]) for r in receipt['trials']])
                          for index, mode in enumerate(MODES)}
    _write(OUTPUT, receipt)
    print('paired outcome-credit summary', receipt['summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = register()
    fit = learn(plan)
    if not args.source_only:
        run(plan, fit, args.workers)


if __name__ == '__main__':
    main()
