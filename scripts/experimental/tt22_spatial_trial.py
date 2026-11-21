"""Score online Pong-derived lateral gains on paired native SO-101 starts."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .tt3_pong import state_sha256
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt22_rally_trial import CONFIG, GentleDraws, RallyArena, alternating, opponent, rates
from .tt22_spatial import MAX_CORRECTION, SpatialChoice, SpatialPlayer, broad_hypotheses, portable_contact


OUTPUT = ROOT/'artifacts/tt22'
SOURCE = ROOT/'artifacts/tt3/discovered-numeric-source.json'
SOURCE_RECEIPT = ROOT/'artifacts/tt3/discovered-source-search.json.gz'
SOURCE_REFINEMENT = ROOT/'artifacts/tt3/discovered-source-refinement.json.gz'
PRACTICE = 1101874493
HELD = 1526715082
DRAW = 62426019
SPREAD = .40
PACE = .82
PRACTICE_LANES = (-.56, -.43, -.32, -.21, .21, .32, .43, .56)
HELD_LANES = (-.96, -.87, -.77, -.68, .68, .77, .87, .96)
DIRECT_PRACTICE = 1027839803
DIRECT_HELD = 1693417351
DIRECT_PRACTICE_LANES = (-.50, -.22, .22, .50)
DIRECT_HELD_LANES = (-.67, -.59, .59, .67)
POLICIES = ('pong', 'scratch', 'score_permuted')


class LateralDraws(GentleDraws):
    def __init__(self, rng, lane):
        super().__init__(rng, SPREAD, PACE)
        self.lane = lane

    def uniform(self, low, high):
        result = super().uniform(low, high)
        if np.ndim(result) == 0:
            if (low, high) == (-.30, .30):
                return .30*self.lane
            if (low, high) == (-.28, .28):
                return .28*self.lane
        return result


class SpatialArena(RallyArena):
    """Set the lateral family of the native opponent launch."""

    def __init__(self):
        self.lane = 0.
        super().__init__(SPREAD, PACE)

    def _opening(self):
        rng = self._rng
        if self.server == -1:
            self._rng = LateralDraws(rng, self.lane)
        try:
            return TT20Arena._opening(self)
        finally:
            self._rng = rng


class Station:
    def __init__(self, *, direct=False):
        self.arena = SpatialArena()
        self.rival = opponent(self.arena, 'varied')
        self.law = FlightLaw()
        self.player = SpatialPlayer(self.law, direct=direct)
        self.lenses = self.arena.lens_intrinsics()

    def point(self, seed, lane, coefficients, *, flat=False, geometry=False):
        arena, player = self.arena, self.player
        arena.lane = lane
        arena.reset(seed=seed, server=-1, render=False)
        player.reset()
        player.coefficients, player.flat_center = coefficients, flat
        player.geometric_override = geometry
        tracker = NaturalViewTransport('fixed', lens_fov=self.lenses, law=self.law,
                                       flight=False, associate=True).tracker()
        actions, optical = [], []
        for tick in range(arena.max_steps):
            tracker.update(arena.render_cameras(), tick=tick)
            blade, ball = tracker.actor, tracker.ball
            actor = (np.asarray(blade.position) if blade is not None and blade.observed
                     and len(tracker.sources.get('blade', ())) == 2 else None)
            position = (np.asarray(ball.position) if ball is not None and ball.observed
                        and tracker.sources.get('ball_range') == 'stereo' else None)
            player.observe_actor(tick, actor)
            action = player.command(tick, position)
            actions.append(action.tolist())
            optical.append(dict(tick=tick, actor=None if actor is None else actor.tolist(),
                                mover=None if position is None else position.tolist()))
            _, reward, done, capped, _ = arena.step(action, render=False)
            if done or capped:
                break
        return dict(seed=seed, role=-1, lane=lane, coefficients=list(map(float, coefficients)), flat_center=flat,
                    direct=player.direct, geometric_override=geometry, neutral_fallbacks=player.neutral_fallbacks,
                    actions=actions, optical=optical, rgb=tracker.digest.hexdigest(), shots=player.shots,
                    outcome=arena.point_result(), events=arena.events.copy(), reward=reward,
                    longest=alternating(arena.events), changed_proposals=player.applied,
                    max_correction_m=player.max_correction, rival=self.rival.decisions.copy())

    def close(self):
        self.player.close()
        self.arena.close()


def schedule(seed, lanes):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(2**31, size=len(lanes)).tolist()
    order = rng.permutation(len(lanes))
    return [[case, lanes[i]] for case, i in zip(seeds, order, strict=True)]


def verify(rows):
    arena = SpatialArena()
    opponent(arena, 'varied')
    try:
        for row in rows:
            arena.lane = row['lane']
            arena.reset(seed=row['seed'], server=-1, render=False)
            for action in row['actions']:
                _, reward, done, capped, _ = arena.step(action, render=False)
            if (not (done or capped) or reward != row['reward'] or arena.point_result() != row['outcome'] or
                    state_sha256(arena.events) != state_sha256(row['events'])):
                raise ValueError(f"the physical lateral test changed at {row['seed']}")
    finally:
        arena.close()
    return len(rows)


def metrics(rows):
    return dict(rates(rows), first_returns=sum(row['outcome']['legal_returns'] > 0 for row in rows),
                second_returns=sum(row['outcome']['legal_returns'] > 1 for row in rows),
                contact_points=sum(row['outcome']['contacts'] > 0 for row in rows))


def point_key(seed, lane, coefficients, flat, geometry=False):
    return seed, lane, tuple(map(float, coefficients)), flat, geometry


def paired_point(station, cache, name, seed, lane, coefficients, *, flat=False, geometry=False):
    key = point_key(seed, lane, coefficients, flat, geometry)
    if key not in cache:
        row = station.point(seed, lane, coefficients, flat=flat, geometry=geometry)
        row['action_sha256'] = state_sha256(row['actions'])
        cache[key] = name, deepcopy(row)
        return row
    origin, recorded = cache[key]
    row = deepcopy(recorded)
    row['cached_from'] = origin
    return row


def annotate(data):
    """Restore public camera availability and the chosen lateral target per shot."""
    for phase in ('practice', 'held'):
        for rows in data[phase].values():
            for row in rows:
                optical = row['optical']
                row['ticks'] = len(row['actions'])
                row['control_seconds'] = row['ticks'] / 30.
                row['action_sha256'] = state_sha256(row['actions'])
                row['pose_visibility'] = dict(actor=sum(r['actor'] is not None for r in optical),
                                              mover=sum(r['mover'] is not None for r in optical))
                for shot in row['shots']:
                    tick = shot['tick']
                    actor = next((r['actor'] for r in reversed(optical[max(0, tick-4):tick+1])
                                  if r['actor'] is not None), None)
                    mover = optical[tick]['mover']
                    nominal = shot['anchor'][0]
                    target = nominal
                    if row.get('geometric_override', False):
                        target = nominal
                    elif row['flat_center'] or (row.get('direct', False) and actor is None):
                        target = 0.
                    elif actor is not None and mover is not None:
                        proposed = portable_contact(actor[0], mover[0], nominal, row['coefficients'])
                        target = float(proposed) if row.get('direct', False) else nominal + float(
                            np.clip(proposed-nominal, -MAX_CORRECTION, MAX_CORRECTION))
                    shot['lateral_target'] = dict(nominal_x_m=nominal, applied_x_m=target,
                                                 actor_x_m=None if actor is None else actor[0],
                                                 mover_x_m=None if mover is None else mover[0])
    data['source_refinement_sha256'] = _hash(SOURCE_REFINEMENT)
    data.setdefault('control_mode', 'bounded_residual')
    return data


def run(path, *, resume=False, direct=False):
    source = _read(SOURCE)
    refinement = _read(SOURCE_REFINEMENT)
    if (source != refinement['portable_memory'] or _hash(SOURCE) != refinement['numeric_state_sha256']):
        raise ValueError('the numeric source differs from its native Pong qualification')
    common = broad_hypotheses(DRAW)
    ties = np.random.default_rng(DRAW+1).random(len(common)+len(source['coefficients']))
    policies = {name: SpatialChoice(common, source, name, ties) for name in POLICIES}
    practice_seed, held_seed = (DIRECT_PRACTICE, DIRECT_HELD) if direct else (PRACTICE, HELD)
    practice_lanes, held_lanes = (DIRECT_PRACTICE_LANES, DIRECT_HELD_LANES) if direct else (PRACTICE_LANES, HELD_LANES)
    practice, held = schedule(practice_seed, practice_lanes), schedule(held_seed, held_lanes)
    prior_family = ((PRACTICE, PRACTICE_LANES), (HELD, HELD_LANES)) if direct else (
        (DIRECT_PRACTICE, DIRECT_PRACTICE_LANES), (DIRECT_HELD, DIRECT_HELD_LANES))
    existing = {seed for generator, lanes in prior_family for seed, _ in schedule(generator, lanes)}
    used = {seed for seed, _ in practice+held}
    if len(used) != len(practice)+len(held) or used & existing:
        raise ValueError('independent physical schedules overlap')
    controls = ('geometry',) if direct else ('geometry', 'flat')
    data = dict(schema='tt22-lateral-native-transfer-1', arena=CONFIG, rival='varied', source_sha256=_hash(SOURCE),
                control_mode='direct' if direct else 'bounded_residual',
                source_receipt_sha256=_hash(SOURCE_RECEIPT), source_refinement_sha256=_hash(SOURCE_REFINEMENT),
                source=source, common=common.tolist(),
                practice_schedule=practice, held_schedule=held, draw_seed=DRAW, practice_seed=practice_seed, held_seed=held_seed,
                practice={name: [] for name in POLICIES}, held={name: [] for name in (*POLICIES, *controls)})
    if resume and path.exists():
        previous = _read(path)
        if any(previous.get(key, value) != value for key, value in data.items() if key not in ('practice', 'held')):
            raise ValueError('the interrupted run used different hypotheses or native starts')
        data['practice'], data['held'] = previous['practice'], previous['held']
    cache = {}
    for phase in ('practice', 'held'):
        for name, rows in data[phase].items():
            for row in rows:
                key = point_key(row['seed'], row['lane'], row['coefficients'], row['flat_center'],
                                row.get('geometric_override', False))
                if key in cache and (cache[key][1]['actions'] != row['actions'] or cache[key][1]['rgb'] != row['rgb']):
                    raise ValueError('the same optical point and policy produced different tapes')
                cache.setdefault(key, (name, row))
    station = Station(direct=direct)
    try:
        for i, (seed, lane) in enumerate(practice):
            offset = i % len(POLICIES)
            for name in POLICIES[offset:]+POLICIES[:offset]:
                learner = policies[name]
                selected = learner.choose(explore=True)
                rows = data['practice'][name]
                if len(rows) > i:
                    row = rows[i]
                    learner.update(selected, row['reward'], row['outcome']['legal_returns'])
                    if (row['seed'], row['lane'], row['selected'], row['checkpoint']) != (seed, lane, selected, learner.state()):
                        raise ValueError('replaying the interrupted practice changed its learned state')
                    continue
                row = paired_point(station, cache, name, seed, lane, learner.coefficients[selected])
                learner.update(selected, row['reward'], row['outcome']['legal_returns'])
                row.update(selected=selected, checkpoint=learner.state())
                data['practice'][name].append(row)
                _write(path, data)
                print('practice', i+1, name, seed, lane, selected, row['reward'], row['outcome']['legal_returns'],
                      row['changed_proposals'], flush=True)
        data['final'] = {name: learner.state() for name, learner in policies.items()}
        data['curve'] = {str(i): {name: metrics(rows[:i]) for name, rows in data['practice'].items()}
                         for i in range(1, len(practice)+1)}
        _write(path, data)
        for i, (seed, lane) in enumerate(held):
            active = tuple(data['held']) if i < 4 else (*POLICIES, 'geometry')
            offset = i % len(active)
            for name in active[offset:]+active[:offset]:
                if len(data['held'][name]) > i:
                    if (data['held'][name][i]['seed'], data['held'][name][i]['lane']) != (seed, lane):
                        raise ValueError('the interrupted evaluation changed its held starts')
                    continue
                coefficients = data['final'][name]['greedy_coefficients'] if name in policies else (0., 1., 1.)
                row = paired_point(station, cache, name, seed, lane, coefficients, flat=name == 'flat',
                                   geometry=direct and name == 'geometry')
                data['held'][name].append(row)
                _write(path, data)
                print('held', i+1, name, seed, lane, row['reward'], row['outcome']['legal_returns'],
                      row['changed_proposals'], flush=True)
    finally:
        station.close()
    data['scores'] = {name: metrics(rows) for name, rows in data['held'].items()}
    data['replayed'] = verify([row for phase in ('practice', 'held') for rows in data[phase].values() for row in rows])
    _write(path, annotate(data))
    print('held final', data['scores'], 'replayed', data['replayed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--annotate', action='store_true')
    parser.add_argument('--direct', action='store_true')
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    receipt = args.folder/('spatial-direct.json.gz' if args.direct else 'spatial-score.json.gz')
    if args.annotate:
        _write(receipt, annotate(_read(receipt)))
    elif args.verify:
        saved = _read(receipt)
        print('replayed', verify([row for phase in ('practice', 'held') for rows in saved[phase].values() for row in rows]))
    else:
        run(receipt, resume=args.resume, direct=args.direct)
