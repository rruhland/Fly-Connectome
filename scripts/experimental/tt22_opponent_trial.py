"""Matched physical rival replies and uninterrupted native RGB rallies."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt19_view_full_game import digest_json
from .tt20_curriculum import ARENA, GAME_SEED, OUTPUT as TT20, apparatus
from .tt20_natural import NaturalViewTransport
from .tt21_coach import install_coach
from .tt21_coach_trial import first_contact, launches
from .tt22_opponent import RALLY_STEPS, RallyArena, install_rally_opponent
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import CONFIG, GentleDraws, RallyArena as FixedOpening, Station, alternating


OUTPUT = ROOT / 'artifacts' / 'tt22'
SOURCE = TT20 / f'curriculum-frozen-{GAME_SEED}.json.gz'
MODES = ('coach', 'anchor', 'steered', 'steered-varied')
LIVE_SEED = 1273025990
SPREAD, PACE = .30, .82


class SteeredOpening(RallyArena):
    def _opening(self):
        if self.server == 1:
            return super()._opening()
        original = self._rng
        self._rng = GentleDraws(original, SPREAD, PACE)
        try:
            super()._opening()
        finally:
            self._rng = original


def arena_for(mode, live):
    steered = mode.startswith('steered')
    if live:
        return SteeredOpening(**CONFIG) if steered else FixedOpening(SPREAD, PACE)
    if steered:
        return RallyArena(**dict(ARENA, max_steps=RALLY_STEPS))
    arena = apparatus(False)
    arena.max_steps = RALLY_STEPS
    return arena


def opponent(arena, mode):
    if mode == 'coach':
        return install_coach(arena)
    return install_rally_opponent(arena, lane='varied' if mode == 'steered-varied' else 'anchor')


class FarTape:
    """Record only the rates actually returned to the live physics step."""

    def __init__(self, arena):
        self.steps = {}
        aim, posture = arena._aim, arena._command_to

        def recording(function):
            def command(side, *args, **kwargs):
                output = function(side, *args, **kwargs)
                if side == -1:
                    self.steps[arena.steps] = output.tolist()
                return output
            return command

        arena._aim = recording(aim)
        arena._command_to = recording(posture)

    def tape(self, count):
        return [self.steps[i] for i in range(count)]


def archived(mode, starts):
    arena = arena_for(mode, live=False)
    rival = opponent(arena, mode)
    recorder = FarTape(arena)
    rows = []
    try:
        for index, original, strike in starts:
            arena.reset(seed=original['seed'], server=original['server'], render=False)
            recorder.steps.clear()
            actions, applied = [], []
            for tick in range(arena.max_steps):
                action = original['actions'][tick] if tick < strike['step'] else np.zeros(arena.action_size).tolist()
                actions.append(action)
                _, reward, done, capped, _ = arena.step(action, render=False)
                if tick + 1 == strike['step']:
                    actual = first_contact(arena.events, 1)
                    if (actual is None or actual['step'] != strike['step']
                            or not np.allclose(actual['incoming'], strike['incoming'], rtol=0, atol=1.e-7)
                            or not np.allclose(actual['location'], strike['location'], rtol=0, atol=1.e-7)):
                        raise ValueError('rival variation disturbed the archived opening contact')
                if arena.rival_returns:
                    applied.append([*arena.ball.tolist(), *arena.velocity.tolist()])
                if done or capped:
                    break
            rows.append(dict(held_point=index, seed=original['seed'], role=original['server'], actions=actions,
                             rival_actions=recorder.tape(len(actions)), outcome=arena.point_result(),
                             events=arena.events.copy(), reward=reward, receiving_path=applied,
                             decisions=rival.decisions.copy()))
            print('opening', mode, index, arena.rival_returns, arena.reason, flush=True)
    finally:
        arena.close()
    return rows


def native(mode, seeds):
    # Reuse the fixed optical learner; replace the arena and rival only.
    station = Station.__new__(Station)
    station.arena = arena_for(mode, live=True)
    station.rival = opponent(station.arena, mode)
    law = FlightLaw()
    station.player = RallyPlayer(law)
    station.optics = NaturalViewTransport('fixed', lens_fov=station.arena.lens_intrinsics(), law=law,
                                           flight=False, associate=True)
    recorder = FarTape(station.arena)
    rows = []
    try:
        for seed in seeds:
            recorder.steps.clear()
            row = station.point(seed)
            row['rival_actions'] = recorder.tape(len(row['actions']))
            rows.append(row)
            print('native', mode, seed, row['longest'], row['outcome']['contacts'],
                  row['outcome']['rival_contacts'], row['outcome']['reason'], flush=True)
    finally:
        station.close()
    return rows


def verify(rows, mode, live):
    arena = arena_for(mode, live)
    try:
        for row in rows:
            arena.reset(seed=row['seed'], server=row['role'], render=False)
            for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
                arena._rival_attempt = True
                arena._aim = lambda *args: np.asarray(far)
                _, reward, done, capped, _ = arena.step(near, render=False)
                if bool(done or capped) != (tick + 1 == len(row['actions'])):
                    raise ValueError('the two rate tapes changed point duration')
            if (arena.point_result() != row['outcome'] or digest_json(arena.events) != digest_json(row['events'])
                    or row['reward'] != reward):
                raise ValueError('the independent two-arm referee replay changed')
    finally:
        arena.close()
    return len(rows)


def metrics(rows):
    verdict = [row['outcome'] for row in rows]
    durations = np.asarray([len(row['actions']) for row in rows])
    exchange = [alternating(row['events']) for row in rows]
    contacts = [outcome['contacts'] + outcome['rival_contacts'] for outcome in verdict]
    plane = []
    for row in rows:
        window = np.asarray(row.get('receiving_path', []))
        if len(window):
            suitable = (window[:, 1] < -.78) & (window[:, 1] > -1.02) & (window[:, 4] < -.3)
            if np.any(suitable):
                good = window[suitable]
                plane.append(good[np.argmin(abs(good[:, 1] + .91))].tolist())
    return dict(points=len(rows), legal_replies=sum(o['rival_returns'] for o in verdict),
                points_with_legal_rival=sum(o['rival_returns'] > 0 for o in verdict),
                learner_legal_returns=sum(o['legal_returns'] for o in verdict),
                longest_alternating=exchange, contacts=contacts, frames=durations.tolist(), strike_plane=plane)


def run(path, suite, modes, count):
    source = _read(SOURCE)
    starts = launches(source)
    seeds = np.random.default_rng(LIVE_SEED).integers(2**31, size=count).tolist()
    if set(seeds).intersection(row['seed'] for row in source['points']):
        raise ValueError('native starts overlap the archived opening course')
    data = dict(schema='tt22-physical-rival-1', suite=suite, source=SOURCE.name, source_sha256=_hash(SOURCE),
                learner='native stereo fixed rally player', control_cap=RALLY_STEPS, modes=modes,
                opening=dict(spread=SPREAD, pace=PACE), live_seeds=seeds, variants={})
    if path.exists():
        prior = _read(path)
        if any(prior.get(key) != data[key] for key in ('suite', 'live_seeds', 'source_sha256')):
            raise ValueError('existing receipt belongs to a different course')
        data['variants'] = prior['variants']
    began = perf_counter()
    path.parent.mkdir(exist_ok=True, parents=True)
    for mode in modes:
        if mode in data['variants']:
            continue
        live = suite == 'native'
        rows = native(mode, seeds) if live else archived(mode, starts)
        data['variants'][mode] = dict(rows=rows, summary=metrics(rows), replayed=verify(rows, mode, live))
        data['wall_seconds'] = perf_counter() - began
        _write(path, data)
    print({key: value['summary'] for key, value in data['variants'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('archived', 'native'), default='native')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=('steered', 'anchor', 'coach'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.output or OUTPUT / f'tt22_opponent-{args.suite}.json.gz'
    run(target, args.suite, args.modes, args.count)
