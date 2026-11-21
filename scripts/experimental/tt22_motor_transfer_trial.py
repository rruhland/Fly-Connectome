"""Paired Pong-preference transfer through an outcome-selected motor primitive."""

import argparse
import json
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt22_motor_transfer import MotorPrimitive
from .tt22_openloop import rally_count
from .tt22_rally_trial import CONFIG, GentleDraws, RallyArena
from .tt22_spatial import broad_hypotheses
from .tt22_spatial_trial import schedule


OUTPUT = ROOT/'artifacts/tt22'/'motor-transfer.json.gz'
MACRO = OUTPUT.with_name('openloop-practice.json')
PONG = ROOT/'artifacts/tt3'/'discovered-numeric-source.json'
PONG_QUALIFICATION = ROOT/'artifacts/tt3'/'discovered-source-refinement.json.gz'
DRAW = 378498210
PRACTICE = 1522482439
HELD = 907336941
PRACTICE_LANES = (-.65, -.45, -.25, .25, .45, .65)
HELD_LANES = (-.78, -.57, -.34, .34, .57, .78)


class LaneDraws(GentleDraws):
    def __init__(self, rng, lane):
        super().__init__(rng, .45, 1.)
        self.lane = lane

    def uniform(self, low, high):
        value = super().uniform(low, high)
        if np.ndim(value) == 0 and (low, high) in ((-.30, .30), (-.28, .28)):
            return high*self.lane
        return value


class MotorArena(RallyArena):
    """Keep the native serve mechanics and expose varied initial lateral lanes."""

    def __init__(self):
        self.lane = 0.
        super().__init__(.45, 1.)

    def _opening(self):
        rng = self._rng
        if self.server == -1:
            self._rng = LaneDraws(rng, self.lane)
        try:
            return TT20Arena._opening(self)
        finally:
            self._rng = rng


class ReturnChoice:
    """Update the option's value solely from the live referee's learner return count."""

    def __init__(self, common, source, *, carried):
        self.candidates = np.asarray(common if not carried else np.vstack((common, source['coefficients'])))
        self.prior = np.ones(len(self.candidates))
        self.initial = np.zeros(len(self.candidates))
        if carried:
            self.prior[len(common):] = 8.
            self.initial[len(common):] = 8.*np.divide(source['totals'], source['counts'])
        self.total = np.zeros(len(self.candidates))
        self.count = np.zeros(len(self.candidates), dtype=int)
        self.ties = np.random.default_rng(DRAW+1).random(len(self.candidates))

    def choose(self, explore=False):
        value = (self.initial+self.total)/(self.prior+self.count)
        if explore:
            value += .48*np.sqrt(np.log(2+self.count.sum())/(1+self.count))
        return int(np.argmax(value+1.e-8*self.ties))

    def record(self, option, returns):
        self.total[option] += returns
        self.count[option] += 1

    def receipt(self):
        selected = self.choose()
        return dict(count=self.count.tolist(), total=self.total.tolist(), coefficients=self.candidates.tolist(),
                    selected=selected, greedy=self.candidates[selected].tolist(), prior=self.prior.tolist(),
                    initial_total=self.initial.tolist())


def _observed(track):
    return None if track is None or not track.observed else np.asarray(track.position)


def point(arena, seed, lane, macro, coefficients, *, shared_prefix=None):
    arena.lane = lane
    arena.reset(seed=seed, server=-1, render=False)
    tracker = NaturalViewTransport('fixed', lens_fov=arena.lens_intrinsics(), flight=False, associate=True).tracker()
    controller = MotorPrimitive(macro, coefficients)
    commands, optics = [], []
    reuse = shared_prefix is not None and shared_prefix['decision'] is not None
    started = perf_counter()
    for tick in range(arena.max_steps):
        ball = blade = None
        if controller.watching(tick):
            if reuse:
                prefix = shared_prefix['optical']
                if tick >= len(prefix) or prefix[tick]['tick'] != tick:
                    raise ValueError('a paired policy requested RGB outside the common prefix')
                entry = prefix[tick]
                ball = None if entry['mover'] is None else np.asarray(entry['mover'])
                blade = None if entry['actor'] is None else np.asarray(entry['actor'])
                optics.append(entry)
            else:
                tracker.update(arena.render_cameras(), tick=tick)
                if tracker.sources.get('ball_range') == 'stereo':
                    ball = _observed(tracker.ball)
                if len(tracker.sources.get('blade', ())) == 2:
                    blade = _observed(tracker.actor)
                optics.append(dict(tick=tick, mover=None if ball is None else ball.tolist(),
                                   actor=None if blade is None else blade.tolist()))
        control = controller.command(tick, ball, blade)
        commands.append(control.tolist())
        _, reward, done, capped, _ = arena.step(control, render=False)
        if done or capped:
            break
    if reuse:
        cue = shared_prefix['decision']['tick']
        if (controller.decision is None or controller.decision['tick'] != cue
                or commands[:cue] != shared_prefix['actions'][:cue]):
            raise ValueError('controls diverged before the shared RGB cue')
    digest = shared_prefix['rgb_sha256'] if reuse else tracker.digest.hexdigest()
    return dict(seed=seed, lane=lane, coefficients=None if coefficients is None else list(map(float, coefficients)),
                actions=commands, optical=optics, rgb_sha256=digest, reused_rgb_prefix=reuse, decision=controller.decision,
                outcome=arena.point_result(), events=arena.events.copy(), reward=reward, legal_alternating=rally_count(arena.events),
                wall_seconds=perf_counter()-started)


def metrics(rows):
    return dict(points=len(rows), first_returns=sum(row['outcome']['legal_returns'] > 0 for row in rows),
                total_returns=sum(row['outcome']['legal_returns'] for row in rows),
                wins=sum(row['reward'] > 0 for row in rows), longest=max((row['legal_alternating'] for row in rows), default=0),
                computed_goals=sum(row['decision'] is not None for row in rows),
                optical_frames=sum(len(row['optical']) for row in rows),
                centered_max_abs_goal=max((abs(row['decision']['displacement_m']) for row in rows
                                           if row['decision'] is not None), default=0.))


def verify(arena, rows):
    for row in rows:
        arena.lane = row['lane']
        arena.reset(seed=row['seed'], server=-1, render=False)
        for command in row['actions']:
            _, reward, _, _, _ = arena.step(command, render=False)
        if (row['outcome'] != arena.point_result() or row['reward'] != reward
                or json.dumps(row['events']) != json.dumps(arena.events)):
            raise ValueError(f"physical replay differs for {row['seed']}")
    return len(rows)


def run():
    macro, source, scored = _read(MACRO), _read(PONG), _read(PONG_QUALIFICATION)
    if macro['spread'] != .45 or macro['opponent'] != 'base' or not macro['training']:
        raise ValueError('expected the scalar-score trained stock clock stroke')
    if source != scored['portable_memory'] or _hash(PONG) != scored['numeric_state_sha256']:
        raise ValueError('the Pong candidate state changed after qualification')
    common = broad_hypotheses(DRAW)
    branches = {name: ReturnChoice(common, source, carried=name == 'pong') for name in ('scratch', 'pong')}
    fresh = {name: brain.candidates[brain.choose()].copy() for name, brain in branches.items()}
    practice, held = schedule(PRACTICE, PRACTICE_LANES), schedule(HELD, HELD_LANES)
    if len({seed for seed, _ in practice+held}) != len(practice)+len(held):
        raise ValueError('independent lateral schedules overlap')
    data = dict(schema='tt22-motor-transfer-1', arena=CONFIG, spread=.45, pace=1., opponent='base',
                macro_sha256=_hash(MACRO), source_sha256=_hash(PONG), source_qualification_sha256=_hash(PONG_QUALIFICATION),
                macro=macro['parameters'], source=source, profiles={'practice':practice,'held':held},
                practice={name: [] for name in branches}, held={}, no_target_practice={n: v.tolist() for n,v in fresh.items()})
    arena = MotorArena()
    started = perf_counter()
    try:
        for index, (seed, lane) in enumerate(practice):
            order = ('scratch','pong') if index % 2 == 0 else ('pong','scratch')
            for name in order:
                brain = branches[name]
                option = brain.choose(explore=True)
                row = point(arena, seed, lane, macro['parameters'], brain.candidates[option])
                brain.record(option, row['outcome']['legal_returns'])
                row.update(option=option, learned=brain.receipt())
                data['practice'][name].append(row)
                print('practice', index+1, name, lane, option, row['outcome']['legal_returns'], row['reward'], flush=True)
            _write(OUTPUT, data)
        data['endpoint'] = {name: brain.receipt() for name, brain in branches.items()}
        controls = {'center':None, 'scratch_frozen':fresh['scratch'], 'pong_frozen':fresh['pong'],
                    'scratch':data['endpoint']['scratch']['greedy'], 'pong':data['endpoint']['pong']['greedy']}
        data['held'] = {name: [] for name in controls}
        for index, (seed, lane) in enumerate(held):
            cache = {}
            for name, coefficients in controls.items():
                key = None if coefficients is None else tuple(coefficients)
                if key not in cache:
                    cache[key] = point(arena, seed, lane, macro['parameters'], coefficients)
                data['held'][name].append(cache[key])
            print('held', index+1, lane, {name: rows[-1]['outcome']['legal_returns'] for name, rows in data['held'].items()}, flush=True)
            data['metrics'] = {stage: {name: metrics(rows) for name, rows in data[stage].items()}
                               for stage in ('practice','held')}
            _write(OUTPUT, data)
        data['verified'] = verify(arena, [row for stage in ('practice','held') for rows in data[stage].values() for row in rows])
        data['wall_seconds'] = perf_counter()-started
        _write(OUTPUT, data)
        print(data['metrics'], flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        saved = _read(OUTPUT)
        arena = MotorArena()
        try:
            print('replayed', verify(arena,[row for stage in ('practice','held') for rows in saved[stage].values() for row in rows]))
        finally:
            arena.close()
    else:
        run()
