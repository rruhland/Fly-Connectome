"""Select a visual-motion modulation of the frozen wrist controller on donors."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import ctypes
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import NATIVE, Native, ROOT
from .tt24_posture import WidePosture
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_public_donor_select import EncoderRig, SOURCE_NAMES, body
from .tt24_ready import episode


TAG = 'tt24_motion_ready'
PLAN = ROOT / f'{TAG}-plan.json'
SOURCE = ROOT / f'{TAG}-source.json.gz'
LOCK = ROOT / f'{TAG}-lock.json'
HELD = ROOT / f'{TAG}.json.gz'
POSTURE = ROOT / 'posture-source.npz'
SEED = 1692978871
FAMILY = {'low_no_floor': (.14, 0.), 'low_with_floor': (.14, .28),
          'high_no_floor': (.34, 0.), 'high_with_floor': (.34, .28)}
FIXED = {'carried': 1., 'fixed_gain': .65}
SOURCE_CAP, HELD_CAP = 210, 300


class VisualNative(Native):
    """Multiply the unchanged visual joint target before the original servo."""

    def __init__(self):
        source = NATIVE.read_text()
        command = 'const double target = .245 * std::tanh(value);'
        if source.count(command) != 1:
            raise ValueError('the frozen original target path changed')
        source = 'static double visual_gain = 1.;\n' + source.replace(command,
                 'const double target = visual_gain * (.245 * std::tanh(value));')
        source += '\nextern "C" void tt24_donor_visual_gain(double gain) { visual_gain = gain; }\n'
        with TemporaryDirectory(prefix='tt24-motion-ready-') as directory:
            file = Path(directory) / NATIVE.name
            file.write_text(source)
            super().__init__(source=file)
        self.set_gain = self.so.tt24_donor_visual_gain
        self.set_gain.argtypes = [ctypes.c_double]
        self.set_gain.restype = None


class EncoderCamera(EncoderRig):
    def body(self, joints):
        self.set_pose(joints)
        return self.pose.copy()


class MotorGainObserver:
    """Use accepted optic motion with causal decay when sightings are absent."""

    def __init__(self, native, sensor, gain):
        self.native = native
        self.sensor = sensor
        self.decoder = sensor.decoder
        self.setting = gain
        self.history = deque(maxlen=3)
        self.memory = 0.
        self.gains = []
        self.sightings = 0

    @property
    def home(self):
        return self.sensor.home

    @home.setter
    def home(self, value):
        self.sensor.home = value

    def observe(self, obs):
        before = self.sensor.gaps
        observed = self.sensor.observe(obs)
        seen = before == self.sensor.gaps
        self.sightings += int(seen)
        self.memory *= .94
        if seen:
            self.history.append(observed[:3].copy())
            if len(self.history) == 3:
                first, middle, last = self.history
                displacement = np.linalg.norm(last - first)
                path = np.linalg.norm(middle - first) + np.linalg.norm(last - middle)
                if path > 1e-8:
                    self.memory = max(self.memory, 30. * displacement * displacement / path)
        else:
            self.history.clear()
        if isinstance(self.setting, (list, tuple)):
            scale, floor = self.setting
            r = min(1., self.memory / scale)
            gain = floor + (1. - floor) * r * r * (3. - 2. * r)
        else:
            gain = float(self.setting)
        self.native.set_gain(gain)
        self.gains.append(gain)
        return observed


def register():
    rng = np.random.default_rng(SEED)
    draws = iter((rng.choice(2**31 - 2, 26, replace=False) + 1).tolist())
    source = {domain: [next(draws) for _ in range(2)] for domain in SOURCE_NAMES}
    held = {domain: [next(draws) for _ in range(n)] for domain, n in
            (('stock', 8), ('alternate', 8), ('immediate_stock', 4))}
    ready = Path(__file__).with_name('tt24_ready.py')
    plan = dict(schema='tt24-motion-ready-1', seed=SEED, source=source, held=held,
                candidates={n: list(v) for n, v in FAMILY.items()}, reference=FIXED,
                inputs='accepted public wrist sightings after encoder-based egomotion compensation',
                controller='visual joint target scaled toward the same familiar joint equilibrium',
                selection='60% mean fraction reaching eight, 20% worst body fraction, 20% mean capped streak',
                original_hashes={file.name: _hash(file) for file in (ready, POSTURE, CHECKPOINT, FUSED_OUTPUT, NATIVE)},
                source_max_steps=SOURCE_CAP, held_max_steps=HELD_CAP, arm_target_practice=0)
    if PLAN.exists():
        if _read(PLAN) != plan:
            raise ValueError('the registered motion modulation course changed')
    else:
        _write(PLAN, plan)
    return plan


def fixture(domain, cap):
    if domain in SOURCE_NAMES:
        server = body(domain, cap)
        player = body(domain, cap)
        player.sample_drop = WidePosture.sample_drop
        return player, server, 'changed', None
    arm = 'alternate' if domain == 'alternate' else 'stock'
    player = WidePosture(view='wrist', width=192, height=144, max_steps=cap, arm=arm)
    server = BounceArena(view='none', arm=arm)
    return player, server, 'changed' if arm == 'alternate' else 'standard', 0 if domain == 'immediate_stock' else None


def play_mode(arena, server, native, weights, home, seed, domain, regime, lead, gain):
    def factory(obs):
        sensor = load_public(FUSED_OUTPUT)
        if domain in SOURCE_NAMES:
            sensor.decoder = EncoderCamera(sensor.decoder, arena)
        observers.append(MotorGainObserver(native, sensor, gain))
        return observers[-1]

    observers = []
    result = episode(arena, server, seed, native, weights, home, 'coherent', regime,
                     sensor_factory=factory, lead_override=lead)
    observer, = observers
    if len(observer.gains) != len(result['actions']):
        raise ValueError('a joint action lacked its current optical observation')
    return result | dict(gains=observer.gains, sightings=observer.sightings)


def evaluate(task):
    domain, seeds, cap, modes = task
    native = VisualNative()
    arena, server, regime, lead = fixture(domain, cap)
    try:
        with np.load(POSTURE, allow_pickle=False) as data:
            home = data['center']
        with np.load(CHECKPOINT, allow_pickle=False) as data:
            weights = np.ascontiguousarray(data['weights'], dtype=np.float64)
        rows = {mode: [play_mode(arena, server, native, weights, home, seed, domain, regime, lead, gain)
                       for seed in seeds] for mode, gain in modes.items()}
        print(domain, {m: [r['run'] for r in data] for m, data in rows.items()}, flush=True)
        return domain, rows
    finally:
        arena.close()
        server.close()
        native.working.cleanup()


def score(rows, mode):
    streak = np.asarray([[r['run'] for r in rows[domain][mode]] for domain in SOURCE_NAMES])
    qualified = (streak >= 8).mean(axis=1)
    return float(.60 * qualified.mean() + .20 * qualified.min() + .20 * (np.minimum(streak, 8) / 8.).mean())


def source_fit(plan, workers):
    receipt = _read(SOURCE) if SOURCE.exists() else dict(plan_sha256=_hash(PLAN), bodies={})
    if receipt['plan_sha256'] != _hash(PLAN):
        raise ValueError('the source gate has different inputs')
    modes = plan['candidates'] | plan['reference']
    tasks = [(domain, launches, SOURCE_CAP, modes) for domain, launches in plan['source'].items()
             if domain not in receipt['bodies']]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for domain, strokes in pool.map(evaluate, tasks):
            receipt['bodies'][domain] = strokes
            _write(SOURCE, receipt)
    scores = {name: score(receipt['bodies'], name) for name in modes}
    chosen = max(plan['candidates'], key=scores.get)
    result = dict(plan_sha256=_hash(PLAN), source_sha256=_hash(SOURCE), selected=chosen,
                  gain=plan['candidates'][chosen], scores=scores, target_practice=0)
    if LOCK.exists():
        if _read(LOCK) != result:
            raise ValueError('the frozen visual gate changed')
    else:
        _write(LOCK, result)
    print('LOCKED MOTION GATE', _hash(LOCK), result, flush=True)
    return result


def summarize(course):
    result = {}
    for name, rows in course.items():
        strokes = [row['run'] for row in rows]
        before = [g for row in rows for g in row['gains'][:row['preparation_ticks']]]
        later = [g for row in rows for g in row['gains'][row['preparation_ticks']:]]
        result[name] = dict(runs=strokes, mean=float(np.mean(strokes)), eight=sum(x >= 8 for x in strokes),
                            ten=sum(x >= 10 for x in strokes), sum_rebounds=sum(row['rebounds'] for row in rows),
                            before_gain=float(np.mean(before)) if before else None, later_gain=float(np.mean(later)))
    return result


def held_test(plan, lock, workers):
    report = _read(HELD) if HELD.exists() else dict(lock_sha256=_hash(LOCK), courses={})
    if report['lock_sha256'] != _hash(LOCK):
        raise ValueError('the held gate has a different lock')
    modes = {'learned': lock['gain']} | plan['reference']
    tasks = [(domain, starts, HELD_CAP, modes) for domain, starts in plan['held'].items()
             if domain not in report['courses']]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for domain, rows in pool.map(evaluate, tasks):
            report['courses'][domain] = rows
            report['summary'] = {name: summarize(value) for name, value in report['courses'].items()}
            _write(HELD, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    plan = register()
    lock = source_fit(plan, args.workers)
    if not args.source_only:
        held_test(plan, lock, args.workers)


if __name__ == '__main__':
    main()
