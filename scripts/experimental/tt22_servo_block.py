"""A prepared physical block followed by anonymously timed recurrent pushes."""

import argparse
import json

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_factory import FactoryModel, RATE, moving_joint_rate
from .tt19_flight import FlightLaw, optical_start
from .tt20_curriculum import CLAMP_SHIFT
from .tt20_fixture import TT20Arena
from .tt22_block_drill import BlockArena, BlockCoach, DEVELOP, Draws, first_return, prepare, reference_pulse
from .tt22_match import step_recorded
from .tt22_motion_object import MotionObject
from .tt22_rally_trial import OUTPUT, alternating
from .tt22_servo import relative_lateral
from .tt22_servo_trial import coefficients, metrics, scheduled


DONOR = OUTPUT/'tt22_block_drill-push.json.gz'
RESULT = OUTPUT/'servo-block.json.gz'
TRAIN = 348562726
HELD = 149137035
JITTER = .024
PREFLIGHT = 225223771
OPTIONS = ((2.2, .04), (3.4, .04), (2.2, .059), (3.4, .059))
DEPTH = -.863


class PerturbedDraws:
    def __init__(self, rng, opening, jitter):
        self.center = Draws(np.random.default_rng(np.random.SeedSequence((DEVELOP, 0, 4))), .12, opening)
        self.rng = rng
        self.jitter = jitter

    def uniform(self, low, high):
        center = self.center.uniform(low, high)
        return center + self.jitter*(self.rng.uniform(low, high) - .5*np.add(low, high))


class BlockCourse(BlockArena):
    def __init__(self, opening, jitter=JITTER):
        self.jitter = jitter
        super().__init__(opening)

    def _opening(self):
        rng = self._rng
        if self.server == -1:
            self._rng = PerturbedDraws(rng, self.opening, self.jitter)
        try:
            TT20Arena._opening(self)
        finally:
            self._rng = rng


class PushServo:
    def __init__(self, ready, weights, settings):
        self.ready = np.asarray(ready, dtype=float)
        self.joints = self.shifted = self.ready.copy()
        self.weights, (self.lead, pace) = weights, settings
        self.model = FactoryModel()
        self.model.base += CLAMP_SHIFT
        self.origin, self.normal = self.model.pose(self.ready)[:2]
        self.drive = moving_joint_rate(self.model, self.ready, pace)
        self.law = FlightLaw()
        self.history, self.pulses = [], []
        self.armed, self.launch = False, None
        self.shift, self.changed = 0., -4

    def command(self, tick, xyz):
        self.history.append((tick, xyz))
        estimate = optical_start(self.history, self.law.gravity)
        if estimate is not None:
            position, speed = estimate
            if position[1] > .065 and speed[1] > .022:
                self.armed, self.launch = True, None
            if self.armed and DEPTH < position[1] < .20 and speed[1] < -.037:
                time = (DEPTH-position[1])/speed[1]
                shift = np.clip(relative_lateral(self.origin[0], position, speed, time, self.weights), -.050, .050)
                if tick-self.changed >= 2 and abs(shift-self.shift) >= .008:
                    self.shifted = self.model.solve(self.origin + [shift, 0., 0.], self.normal, self.shifted)
                    self.shift, self.changed = float(shift), tick
                if time <= self.lead:
                    self.armed, self.launch = False, tick + time - self.lead
                    self.pulses.append(dict(tick=tick, phase_origin=float(self.launch), shift=float(self.shift),
                                             incoming=position.tolist(), velocity=speed.tolist()))
        age = 10. if self.launch is None else tick-self.launch
        push = min(3., age, max(0., 10.-age))
        goal = self.shifted + push*self.drive
        command = np.clip((goal-self.joints)/RATE[:5], -1., 1.)
        self.joints = self.joints + command*RATE[:5]
        return np.r_[command, 0.]


class Station:
    def __init__(self, reference):
        self.arena = BlockCourse(reference['opening'])
        self.ready = reference['joints']
        template = reference_pulse(self.ready, -.63, .04)
        self.partner = BlockCoach(self.arena, self.ready, pulse=template)
        self.weights = coefficients(loft=True)

    def close(self):
        self.partner.close()
        self.arena.close()

    def point(self, seed, mode, settings):
        arena = self.arena
        preparation = prepare(arena, seed, self.ready)
        view = MotionObject(arena.lens_intrinsics())
        pupil = PushServo(self.ready, self.weights[mode], settings)
        commands, opponent, optical = [], [], []
        for tick in range(arena.max_steps):
            seen = view.update(arena.render_cameras(), tick)
            command = pupil.command(tick, seen)
            commands.append(command.tolist())
            optical.append(None if seen is None else seen.tolist())
            reward, done, cap, far = step_recorded(arena, command)
            opponent.append(far)
            if done or cap:
                break
        return dict(seed=seed, mode=mode, settings=list(settings), preparation=preparation, actions=commands,
                    rival_actions=opponent, optical=optical, initialized=view.seed_score is not None, pulses=pupil.pulses,
                    longest=alternating(arena.events), reward=reward, outcome=arena.point_result(), events=arena.events.copy())


def verify(reference, rows):
    arena = BlockCourse(reference['opening'])
    try:
        for row in rows:
            preparation = prepare(arena, row['seed'], reference['joints'])
            if preparation != row['preparation']:
                raise ValueError('prepared direct posture changed')
            for command, far in zip(row['actions'], row['rival_actions'], strict=True):
                arena._aim = arena._command_to = lambda *args, rates=np.asarray(far): rates
                _, reward, _, _, _ = arena.step(command, render=False)
            if (row['reward'] != reward or row['outcome'] != arena.point_result()
                    or json.dumps(row['events']) != json.dumps(arena.events)):
                raise ValueError(f"physical block-servo replay changed on {row['seed']}")
    finally:
        arena.close()
    return len(rows)


def run(path):
    reference = _read(DONOR)['reference']
    bench = Station(reference)
    train = [seed for seed,_ in scheduled(TRAIN, [0.])]
    held = [seed for seed,_ in scheduled(HELD, [0.]*4)]
    if len(set(train+held)) != len(train)+len(held) or DEVELOP in train+held:
        raise ValueError('practice and held release jitter must be distinct')
    result = dict(schema='tt22-recurrent-block-servo-1', donor_sha256=_hash(DONOR), opening=reference['opening'],
                  posture=reference['joints'], jitter=JITTER, weights=bench.weights, practice=[], pairs=[],
                  practice_seed=TRAIN, held_seed=HELD)
    try:
        for settings in OPTIONS:
            row = bench.point(train[0], 'ignore', settings)
            result['practice'].append(row)
            print('practice', settings, row['longest'], len(row['pulses']), flush=True)
            _write(path, result)
        best = max(result['practice'], key=lambda row: (row['outcome']['legal_returns'], row['longest'], row['reward']))
        result['selected'] = best['settings']
        for i, seed in enumerate(held):
            order = ('follow', 'ignore') if i%2 == 0 else ('ignore', 'follow')
            pair = {mode: bench.point(seed, mode, result['selected']) for mode in order}
            result['pairs'].append(pair)
            print('held', seed, {mode: (row['longest'], len(row['pulses'])) for mode,row in pair.items()}, flush=True)
            _write(path, result)
    finally:
        bench.close()
    result['verified'] = verify(reference, result['practice'] + [row for pair in result['pairs'] for row in pair.values()])
    result['summary'] = {mode: metrics([pair[mode] for pair in result['pairs']]) for mode in ('follow','ignore')}
    _write(path, result)
    print(result['summary'], flush=True)


def preflight():
    reference = _read(DONOR)['reference']
    seeds = [seed for seed, _ in scheduled(PREFLIGHT, [0.]*8)]
    checks = []
    for jitter in (JITTER, JITTER/10):
        arena = BlockCourse(reference['opening'], jitter)
        try:
            rows = [first_return(arena, seed, reference['joints'], settled=True) for seed in seeds]
        finally:
            arena.close()
        checks.append(dict(jitter=jitter, first=sum(row['legal'] for row in rows), rows=rows))
    _write(OUTPUT/'servo-block-width-preflight.json.gz', dict(seed=PREFLIGHT, donor_sha256=_hash(DONOR), cases=checks))
    print([(row['jitter'], row['first']) for row in checks], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        run(RESULT)
