"""Carry native Pong's opening displacement into a full stereo SO-101 point."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt19_view_full_game import digest_json
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt22_duration_transfer import CHECKPOINT as PONG_CHECKPOINT
from .tt22_duration_transfer import PONG, RECEIPT, STREAMS, priors
from .tt22_flight_inverse_trial import InversePlayer
from .tt22_joint_practice import READS, draw
from .tt22_match import CHECKPOINT as OPTICAL_CHECKPOINT
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer import MotorPrimitive, lateral_goal
from .tt22_motor_transfer_trial import MACRO, LaneDraws
from .tt22_opponent import RallyArena, install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt22_rally import ReturnMemory
from .tt22_rally_trial import CONFIG, OUTPUT, alternating


OPPONENT = OUTPUT / 'tt22_opponent-learned.json.gz'
ARCHIVES = (RECEIPT, OUTPUT / 'population-transfer.json.gz', OUTPUT / 'contextual-motor.json.gz',
            OUTPUT / 'paired-course.json.gz')
DESTINATION = OUTPUT.parent / 'tt23' / 'tt23_sequence.json.gz'
COHORTS = {'a': 17841592, 'b': 1954271632}
REPLICATION = {'c': 1750364263, 'd': 194462034}
STARTS_PER_STRATUM = 6
MODES = ('duration', 'neutral', 'permuted')
HANDOVER = 25


class SequenceArena(RallyArena):
    """Use MotorArena's release distribution with the five-axis live far arm."""

    lane = 0.

    def _opening(self):
        rng = self._rng
        if self.server == -1:
            self._rng = LaneDraws(rng, self.lane)
        try:
            return TT20Arena._opening(self)
        finally:
            self._rng = rng


class OpeningEnded(Exception):
    """The strict referee ended a point before the common observation ended."""


class LivePoint:
    """Let the existing anonymous prefix reader see the same live RGB as the continuation."""

    def __init__(self, arena, optics, player, far):
        self.arena, self.player, self.far = arena, player, far
        self.tracker = optics.tracker()
        self.actions, self.sightings = [], []
        self.terminal = False
        self.reward = 0

    def lens_intrinsics(self):
        return self.arena.lens_intrinsics()

    def reset(self, **settings):
        self.arena.lane = self.lane
        self.far.steps.clear()
        self.player.reset()
        self.arena.reset(**settings)

    def render_cameras(self):
        tick = len(self.actions)
        frames = self.arena.render_cameras()
        self.tracker.update(frames, tick=tick)
        ball = self.tracker.ball
        position = None
        if ball is not None and ball.observed and self.tracker.sources.get('ball_range') == 'stereo':
            position = np.asarray(ball.position)
            self.sightings.append((tick, position.tolist()))
        self.position = position
        return frames

    def step(self, action, *, render=False):
        if len(self.actions) < HANDOVER:
            self.player.history.append((len(self.actions), self.position))
            self.player.sim.twin.advance(action)
        self.actions.append(action.tolist())
        _, self.reward, done, cap, _ = self.arena.step(action, render=render)
        self.terminal = bool(done or cap)
        if self.terminal and len(self.actions) < READS:
            raise OpeningEnded


class SequenceStation:
    def __init__(self, macro, evidence, profile):
        self.arena = SequenceArena(**CONFIG)
        self.rival = install_rally_opponent(self.arena, target_profile=profile)
        self.far = FarTape(self.arena)
        law = FlightLaw()
        self.optics = NaturalViewTransport('fixed', lens_fov=self.arena.lens_intrinsics(), law=law,
                                           flight=False, associate=True)
        self.player = InversePlayer(law)
        self.player.memory = ReturnMemory(evidence)
        self.macro = macro

    def play(self, seed, lane, theta):
        live = LivePoint(self.arena, self.optics, self.player, self.far)
        try:
            sight = prefix(live, seed, lane)
        except OpeningEnded:
            sight = None
        primitive = MotorPrimitive(self.macro, None)
        goal = None
        if sight is not None and sight['mover'] is not None:
            ready = primitive.ready_world
            mover, velocity = np.asarray(sight['mover'])
            goal, _ = lateral_goal(ready, mover, velocity, np.r_[0., theta[:2]])
            primitive._translate(goal, ready)
        for tick in range(len(live.actions), self.arena.max_steps):
            if live.terminal:
                break
            live.render_cameras()
            action = primitive.command(tick) if tick < HANDOVER else self.player.command(tick, live.position)
            live.step(action)
        ticks = len(live.actions)
        return dict(seed=seed, lane=lane, reward=live.reward, longest=alternating(self.arena.events),
                    outcome=self.arena.point_result(), events=self.arena.events.copy(),
                    handover=HANDOVER if ticks > HANDOVER else None, opening=sight, goal=goal,
                    actions=live.actions, rival_actions=self.far.tape(ticks), optical=live.sightings,
                    rgb=live.tracker.digest.hexdigest(), shots=self.player.shots, rival=self.rival.decisions.copy())

    def close(self):
        self.player.close()
        self.arena.close()


def buried_seeds(value):
    if isinstance(value, dict):
        found = {value['seed']} if isinstance(value.get('seed'), int) else set()
        for key, child in value.items():
            if key == 'training_seeds':
                found.update(child)
            else:
                found.update(buried_seeds(child))
        return found
    if isinstance(value, list):
        return set().union(*(buried_seeds(item) for item in value if isinstance(item, (dict, list))))
    return set()


def course(cohorts):
    banks = []
    for cohort, seed in cohorts.items():
        rng = np.random.default_rng(seed)
        draws = {name: draw(rng, STARTS_PER_STRATUM, span) for name, span in STREAMS.items()}
        bank = []
        for visits in zip(*draws.values(), strict=True):
            for stratum, (point, lane) in zip(draws, visits, strict=True):
                bank.append(dict(cohort=cohort, stratum=stratum, seed=point, lane=lane))
        banks.append(bank)
    return [entry for visits in zip(*banks, strict=True) for entry in visits]


def setup(cohorts=COHORTS):
    coefficients = priors()
    coefficients['permuted'] = np.r_[_read(PONG)['frozen']['permuted'][1:], np.zeros(4)]
    macro = _read(MACRO)
    checkpoint = _read(OPTICAL_CHECKPOINT)
    opponent = _read(OPPONENT)
    starts = course(cohorts)
    known = set().union(*(buried_seeds(_read(path)) for path in ARCHIVES), buried_seeds(checkpoint),
                        buried_seeds(opponent), buried_seeds(macro), *(row['seeds'] for row in macro['training']))
    if cohorts == REPLICATION:
        known.update(case['seed'] for case in course(COHORTS))
    seeds = {case['seed'] for case in starts}
    if len(seeds) != len(starts) or known & seeds:
        raise ValueError('target launches must be fresh and unique')
    if (len(checkpoint['evidence']) != 7 or macro['spread'] != .45 or macro['opponent'] != 'base'
            or not macro['training'] or macro['parameters'][10] + macro['parameters'][11] + 5 > HANDOVER):
        raise ValueError('expected seven optical examples and a completed native clock macro')
    return dict(schema='tt23-pong-openers-physical-sequence-1', handover=HANDOVER, prefix_frames=READS,
                modes=list(MODES), starts=starts, bank_seeds=cohorts, arena=CONFIG,
                gains={mode: coefficients[mode].tolist() for mode in MODES},
                source_sha256={path.name: _hash(path) for path in (PONG, PONG_CHECKPOINT, MACRO, OPTICAL_CHECKPOINT, OPPONENT)},
                macro=macro['parameters'], evidence=checkpoint['evidence'], far_profile=opponent['fit']['profile'])


def milestones(row):
    opening_contact = opening_return = False
    last_contact = None
    previous, consecutive = None, 0
    thresholds = {}
    for event in row['events']:
        kind, side, tick = event['kind'], event.get('side'), event['step']
        if kind == 'contact' and side == 1:
            last_contact = tick
            opening_contact |= tick <= HANDOVER
        if kind == 'legal_return':
            if side == 1 and last_contact is not None and last_contact <= HANDOVER:
                opening_return = True
            consecutive = consecutive + 1 if side != previous else 1
            previous = side
            if consecutive in (10, 20):
                thresholds.setdefault(str(consecutive), tick)
    return dict(opening_contact=opening_contact, opening_return=opening_return, milestones=thresholds)


def summarize(rows):
    counts = dict(points=len(rows), opening_contacts=0, opening_returns=0, learner_returns=0, far_returns=0,
                  wins=0, ten_or_more=0, twenty_or_more=0, first_ten=None, lengths=[], control_frames=0)
    for index, row in enumerate(rows, 1):
        counts['opening_contacts'] += row['opening_contact']
        counts['opening_returns'] += row['opening_return']
        counts['learner_returns'] += row['outcome']['legal_returns']
        counts['far_returns'] += row['outcome']['rival_returns']
        counts['wins'] += row['reward'] > 0
        counts['ten_or_more'] += row['longest'] >= 10
        counts['twenty_or_more'] += row['longest'] >= 20
        tick = row['milestones'].get('10')
        if tick is not None and counts['first_ten'] is None:
            total = counts['control_frames'] + tick
            counts['first_ten'] = dict(point=index, frame_in_point=tick, total_frames=total,
                                       simulated_seconds=total / 30.)
        counts['lengths'].append(row['longest'])
        counts['control_frames'] += len(row['actions'])
    counts['longest'] = max(counts['lengths'], default=0)
    return counts


def verify(cases):
    arena = SequenceArena(**CONFIG)
    checked = 0
    try:
        for case in cases:
            for row in case['results'].values():
                arena.lane = row['lane']
                arena.reset(seed=row['seed'], server=-1, render=False)
                for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True), 1):
                    rates = np.asarray(far)
                    arena._aim = arena._command_to = lambda *args, rates=rates: rates
                    _, reward, done, cap, _ = arena.step(near, render=False)
                    if bool(done or cap) != (tick == len(row['actions'])):
                        raise ValueError(f'point duration diverged at seed {row["seed"]}')
                if (reward != row['reward'] or arena.point_result() != row['outcome']
                        or digest_json(arena.events) != digest_json(row['events'])
                        or alternating(arena.events) != row['longest']):
                    raise ValueError(f'physical command replay diverged at seed {row["seed"]}')
                checked += 1
    finally:
        arena.close()
    return checked


def run(path, start_index, limit, cohorts=COHORTS):
    meta = setup(cohorts)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read(path) if path.exists() else dict(meta, cases=[], verified=0)
    if any(digest_json(data[key]) != digest_json(value) for key, value in meta.items()):
        raise ValueError('the saved course has different evaluation inputs')
    by_seed = {case['seed']: case for case in data['cases']}
    _write(path, data)
    station = SequenceStation(data['macro'], data['evidence'], data['far_profile'])
    try:
        for index, start in enumerate(meta['starts'][start_index:limit], start_index):
            case = by_seed.setdefault(start['seed'], dict(start, results={}))
            if any(case[key] != value for key, value in start.items()):
                raise ValueError('the saved course has a different launch order')
            for offset in range(len(MODES)):
                mode = MODES[(index + offset) % len(MODES)]
                if mode in case['results']:
                    continue
                row = station.play(case['seed'], case['lane'], np.asarray(data['gains'][mode]))
                opening = row['opening']
                if any(row['opening'] != prior['opening'] or row['actions'][:READS] != prior['actions'][:READS]
                       for prior in case['results'].values()):
                    raise ValueError('paired policies had different initial RGB or commands')
                case['results'][mode] = dict(row, **milestones(row))
                data['cases'] = [by_seed[launch['seed']] for launch in meta['starts'] if launch['seed'] in by_seed]
                _write(path, data)
                print(index + 1, case['cohort'], case['stratum'], mode, 'sight', opening is not None
                      and opening['mover'] is not None, 'length', row['longest'], 'steps', len(row['actions']), flush=True)
    finally:
        station.close()
    complete = [case for case in data['cases'] if set(case['results']) == set(MODES)]
    groups = dict(combined=complete, **{cohort: [case for case in complete if case['cohort'] == cohort]
                                       for cohort in cohorts})
    data['summary'] = {group: {mode: summarize([case['results'][mode] for case in cases]) for mode in MODES}
                       for group, cases in groups.items()}
    data['verified'] = verify(complete)
    _write(path, data)
    print(data['summary']['combined'], 'verified', data['verified'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--limit', type=int, default=2 * STARTS_PER_STRATUM * len(COHORTS))
    parser.add_argument('--replication', action='store_true')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    cohorts = REPLICATION if args.replication else COHORTS
    path = args.output or (DESTINATION.with_name('tt23_sequence_replication.json.gz') if args.replication else DESTINATION)
    if args.plan:
        if path.exists():
            parser.error('the requested plan path already exists')
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, dict(setup(cohorts), cases=[], verified=0))
    elif args.verify:
        data = _read(path)
        print('verified', verify([case for case in data['cases'] if set(case['results']) == set(MODES)]))
    else:
        if not 0 <= args.start_index < args.limit <= 2 * STARTS_PER_STRATUM * len(COHORTS):
            parser.error('the requested index span must be within the predefined course')
        run(path, args.start_index, args.limit, cohorts)
