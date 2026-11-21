"""Train optical recovery timing on real replies, then score strict SO-101 matches."""

import argparse
from copy import copy
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt20_camera_carry import SETTINGS, summarize
from .tt20_curriculum import OUTPUT as SOURCE, receiver
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt20_opponent import install_opponent
from .tt21_recovery import (NEAR_WINDOW, PROGRAMS, ContextTimingMemory, NearSideRecovery, RallyContext,
                            RecoveryRollout, TimingMemory, time_to_reach, visual_merit)


OUTPUT = ROOT / 'artifacts/tt21'
ARENA = dict(SETTINGS, view_mode='fixed', width=320, height=240)
SOURCES = ('camera-carry-left-game-68239503-1-strong-adaptive.json.gz',
           'camera-carry-wrist-game-68239503-2-strong-adaptive.json.gz',
           'curriculum-natural-2082473549.json.gz')
GAME = 517132601
HOLD = 334951920
REPLICATION = 2026316919
REPLICATION_POINTS = 32
NEAR_SIDE = 1747263091
NEAR_SIDE_POINTS = 24
OPENING = 1
PREFIX = 30


def practice_starts():
    """Bank previous physical openings, completing each new exchange live."""
    seen, starts = set(), []
    for name in SOURCES:
        for row in _read(SOURCE/name)['points']:
            result = row.get('outcome', row.get('result'))
            role = row.get('role', row.get('server'))
            if row['seed'] in seen or result['rival_returns'] == 0 or len(row['actions']) < PREFIX:
                continue
            seen.add(row['seed'])
            starts.append(dict(seed=row['seed'], role=role, prefix=row['actions'][:PREFIX], origin=name))
    return starts


def observed(track):
    return None if track is None or not track.observed else np.asarray(track.position, dtype=float)


def factory_shadow(model):
    """Copy the receiver's point state at the optical choice, sharing dry planning."""
    if model.phase != 'receive' or model.role != -1:
        raise ValueError('the paired receiver shadow requires an opponent service')
    shadow = copy(model)
    shadow.history = model.history.copy()
    shadow.player = copy(model.player)
    shadow.player.arm = 0
    shadow.player.movement = model.player.movement.copy()
    shadow.player.history = model.player.history.copy()
    shadow.player.plans = model.player.plans.copy()
    return shadow


class Station:
    def __init__(self, receiver_type=RecoveryRollout):
        self.arena = TT20Arena(**ARENA)
        self.rival = install_opponent(self.arena)
        self.model = receiver(False, SOURCE)
        self.model.player = receiver_type(self.model.player)
        self.lenses = self.arena.lens_intrinsics()
        self.scene = self.new_scene()

    def new_scene(self):
        return NaturalViewTransport('fixed', lens_fov=self.lenses, flight=False)

    def close(self):
        self.model.player.close()
        self.arena.close()

    def point(self, seed, role, arm, prefix=(), *, fresh_optics=False, compare_factory=False):
        context = RallyContext(role)
        self.arena.reset(seed=seed, server=role, render=False)
        self.model.reset(role)
        self.model.serves.force = OPENING
        stroke = self.model.player
        stroke.arm, stroke.context = int(arm) if not callable(arm) else 0, context
        tracker = (self.new_scene() if fresh_optics else self.scene).tracker()
        actions, optics = [], []
        shadow, difference = None, None
        for tick in range(self.arena.max_steps):
            ball = blade = None
            if not prefix or tick >= 23 or tick in (0, 3):
                tracker.update(self.arena.render_cameras(), tick=tick)
                ball, blade = observed(tracker.ball), observed(tracker.actor)
                optics.append((tick, None if ball is None else ball.tolist(),
                              None if blade is None else blade.tolist()))
            recorded_opening = bool(prefix) and tick >= 22
            opening = recorded_opening or (role == 1 and self.model.actor is not None) or bool(stroke.plans)
            context.observe(tick, ball, blade, opening, stroke.law)
            if callable(arm) and context.approach == tick:
                stroke.arm = arm(context.eta)
                if compare_factory and stroke.arm != 0:
                    shadow = factory_shadow(self.model)
            if tick < len(prefix):
                action = np.asarray(prefix[tick])
            else:
                if prefix and tick == len(prefix):
                    self.model.phase = 'receive'
                    stroke.movement = np.sum(np.asarray(prefix), axis=0)[:5]
                    stroke.history.extend(context.history[:-1])
                plans = len(stroke.plans)
                action = self.model.command(tick, ball)
                if shadow is not None and not np.array_equal(action, shadow.command(tick, ball)):
                    difference = tick
                    shadow = None
                if context.approach is not None and context.launch is None and len(stroke.plans) > plans:
                    context.launch = tick
            actions.append(action.tolist())
            _, reward, done, truncated, _ = self.arena.step(action, render=False)
            if done or truncated:
                break
        return dict(seed=seed, role=role, arm=stroke.arm, reward=reward, actions=actions, optical=optics,
                    outgoing=context.outgoing_tick, approach=context.approach, eta=context.eta,
                    launch=context.launch, plans=stroke.plans, factory_action_difference=difference,
                    optical_merit=visual_merit(context.history, context.launch, stroke.law),
                    outcome=self.arena.point_result(), events=self.arena.events.copy(), rgb=tracker.digest.hexdigest())


def report(rows):
    result = summarize(rows)
    rival = [row for row in rows if row['outcome']['rival_returns']]
    opportunities = sum(row['outcome']['rival_returns'] for row in rows)
    countermoves = 0
    for row in rival:
        incoming = False
        for event in row['events']:
            if event['kind'] != 'legal_return':
                continue
            if event['side'] == -1:
                incoming = True
            elif incoming:
                countermoves += 1
                incoming = False
    result.update(rival_returns=opportunities, subsequent_returns=countermoves,
                  optical_approaches=sum(row['approach'] is not None for row in rows))
    return result


def verify(rows):
    arena = TT20Arena(**ARENA)
    install_opponent(arena)
    try:
        for row in rows:
            arena.reset(seed=row['seed'], server=row['role'], render=False)
            for action in row['actions']:
                _, reward, _, _, _ = arena.step(action, render=False)
            if (arena.point_result() != row['outcome'] or reward != row['reward'] or
                    json.dumps(arena.events) != json.dumps(row['events'])):
                raise ValueError(f"physical replay mismatch at {row['seed']}")
    finally:
        arena.close()
    return len(rows)


def train(path):
    memory = TimingMemory()
    contextual = ContextTimingMemory()
    station = Station()
    starts = practice_starts()
    result = dict(schema='tt21-recovery-practice-v1', arena=ARENA, programs=PROGRAMS,
                  sources={name: _hash(SOURCE/name) for name in SOURCES}, training=[],
                  opening_control=OPENING, prefix_frames=PREFIX)
    try:
        for start in starts:
            choice = memory.choose(explore=True)
            row = station.point(start['seed'], start['role'], choice, start['prefix'])
            if row['approach'] is not None:
                memory.record(choice, row['reward'], row['optical_merit'])
                contextual.record(row['eta'], choice, row['reward'], row['optical_merit'])
            row.update(origin=start['origin'], memory=memory.state(), context_memory=contextual.state())
            result['training'].append(row)
            result.update(state=memory.state(), contextual=contextual.state(), frozen=TimingMemory().state(),
                          summary=report(result['training']), training_frames=sum(len(p['actions']) for p in result['training']))
            _write(path, result)
            print('practice', start['seed'], choice, row['approach'], row['launch'], row['reward'],
                  row['outcome']['legal_returns'], row['optical_merit'], memory.state(), flush=True)
    finally:
        station.close()
    result['replayed'] = verify(result['training'])
    _write(path, result)


def fit_contexts(saved, law):
    contextual = ContextTimingMemory()
    for row in saved['training']:
        if row['approach'] is None:
            continue
        eta = row.get('eta')
        if eta is None:
            eta = time_to_reach([(t, ball) for t, ball, _ in row['optical'] if t <= row['approach']], law)
        if eta is not None:
            contextual.record(eta, row['arm'], row['reward'], row['optical_merit'])
    return contextual


def evaluate(path, practice, count):
    saved = _read(practice)
    station = Station()
    try:
        contextual = fit_contexts(saved, station.model.player.law)
    finally:
        station.close()
    timing = contextual.state()
    noop = timing['fast'] == timing['slow'] == 0
    train_seeds = {row['seed'] for row in saved['training']}
    held = np.random.default_rng(HOLD).integers(2**31, size=count).tolist()
    game = np.random.default_rng(GAME).integers(2**31, size=80).tolist()
    if train_seeds.intersection(held + game) or set(held).intersection(game):
        raise ValueError('training, held starts, and full game must use disjoint launches')
    result = dict(schema='tt21-recovery-held-v1', arena=ARENA, practice_sha256=_hash(practice), held_seed=HOLD,
                  game_seed=GAME, training_frames_per_condition=saved['training_frames'],
                  learned_timing=timing, conditions={})
    print('held timing', timing, 'global', saved['state'], flush=True)
    for name, choice in (('learned', contextual.choose), ('frozen', 0)):
        branch = result['conditions'][name] = dict(timing=timing if name == 'learned' else {'fast': 0, 'slow': 0},
                                                 held=[], game=[])
        if name == 'frozen' and noop:
            # Both branches execute the same deterministic optical policy.
            branch.update(result['conditions']['learned'], equivalent_to='learned')
            _write(path, result)
            break
        station = Station()
        try:
            for index, seed in enumerate(held):
                branch['held'].append(station.point(seed, (-1, 1)[index % 2], choice, fresh_optics=True))
            branch['held_summary'] = report(branch['held'])
            _write(path, result)
            print('held', name, branch['held_summary'], flush=True)
            score = ServiceScore(first_server=-1)
            for seed in game:
                row = station.point(seed, score.server, choice, fresh_optics=True)
                branch['game'].append(row)
                score.record(row['outcome'])
                if score.finished:
                    break
            if not score.finished:
                raise ValueError('the strict game did not finish')
            branch.update(score=list(score.score), game_summary=report(branch['game']))
            _write(path, result)
            print('game', name, score.score, branch['game_summary'], flush=True)
        finally:
            station.close()
    learned, frozen = result['conditions']['learned'], result['conditions']['frozen']
    result['shared_first_actions'] = {}
    for section in ('held', 'game'):
        matched = 0
        for a, b in zip(learned[section], frozen[section]):
            if (a['seed'], a['role']) != (b['seed'], b['role']):
                break
            cutoff = min(len(a['actions']), len(b['actions']), a['outgoing'] or ARENA['max_steps'],
                         b['outgoing'] or ARENA['max_steps'])
            if a['actions'][:cutoff] != b['actions'][:cutoff]:
                raise ValueError(f"first-shot controls diverged before optical recovery at {a['seed']}")
            matched += 1
        result['shared_first_actions'][section] = matched
    result['replayed'] = verify([row for branch in result['conditions'].values() for section in ('held', 'game')
                                 for row in branch[section]])
    _write(path, result)


def replicate(path, practice, *, near_side=False):
    """Freeze the learned rule and match all unselected rival service starts."""
    saved = _read(practice)
    cohort_seed, count = (NEAR_SIDE, NEAR_SIDE_POINTS) if near_side else (REPLICATION, REPLICATION_POINTS)
    seeds = np.random.default_rng(cohort_seed).integers(2**31, size=count).tolist()
    forbidden = {row['seed'] for row in saved['training']}
    forbidden.update(np.random.default_rng(HOLD).integers(2**31, size=80).tolist())
    forbidden.update(np.random.default_rng(GAME).integers(2**31, size=80).tolist())
    other_seed, other_count = (REPLICATION, REPLICATION_POINTS) if near_side else (NEAR_SIDE, NEAR_SIDE_POINTS)
    forbidden.update(np.random.default_rng(other_seed).integers(2**31, size=other_count).tolist())
    if len(set(seeds)) != count or forbidden.intersection(seeds):
        raise ValueError('the independent launches overlap')
    result = dict(schema='tt21-recovery-replication-v1', arena=ARENA, seed=cohort_seed, seeds=seeds,
                  practice_sha256=_hash(practice), present=[], training_frames_per_condition=saved['training_frames'])
    if near_side:
        result['extra_window'] = list(NEAR_WINDOW)
    station = Station(NearSideRecovery if near_side else RecoveryRollout)
    try:
        timing = fit_contexts(saved, station.model.player.law)
        result['learned_timing'] = timing.state()
        if path.exists():
            previous = _read(path)
            if any(previous[key] != result[key] for key in result if key != 'present'):
                raise ValueError('replication receipt differs from the frozen policy or starts')
            result['present'] = previous['present']
        for seed in seeds[len(result['present']):]:
            learned = station.point(seed, -1, timing.choose, fresh_optics=True, compare_factory=True)
            # These copied controller states receive the same live observations
            # while they produce identical physical actions. Only a divergent
            # command needs a separately rendered point.
            cutoff = learned['factory_action_difference']
            if cutoff is None:
                frozen = None
            else:
                frozen = station.point(seed, -1, 0, fresh_optics=True)
                if (len(frozen['actions']) <= cutoff or learned['actions'][:cutoff] != frozen['actions'][:cutoff] or
                        learned['actions'][cutoff] == frozen['actions'][cutoff]):
                    raise ValueError(f'live factory comparison differs from the command audit on {seed}')
            result['present'].append(dict(seed=seed, learned=learned, frozen=frozen,
                                          frozen_equivalent=frozen is None))
            _write(path, result)
            comparison = learned if frozen is None else frozen
            print('replication', len(result['present']), seed, learned['arm'], learned['reward'], comparison['reward'],
                  learned['outcome']['legal_returns'], comparison['outcome']['legal_returns'], flush=True)
    finally:
        station.close()
    learned = [row['learned'] for row in result['present']]
    frozen = [row['frozen'] or row['learned'] for row in result['present']]
    result.update(learned=report(learned), frozen=report(frozen), replayed=verify(learned + frozen))
    _write(path, result)
    print('replication final', result['learned'], result['frozen'], 'replayed', result['replayed'], flush=True)


def verify_receipts(folder):
    rows = list(_read(folder/'recovery-practice.json.gz')['training'])
    held = _read(folder/'recovery-held.json.gz')['conditions']
    rows.extend(row for branch in held.values() for section in ('held', 'game') for row in branch[section])
    for name in ('recovery-replication.json.gz', 'recovery-near-side.json.gz'):
        path = folder/name
        if path.exists():
            saved = _read(path)
            if [row['seed'] for row in saved['present']] != saved['seeds']:
                raise ValueError(f'incomplete cohort in {name}')
            for row in saved['present']:
                rows.extend((row['learned'], row['frozen'] or row['learned']))
    return verify(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--phase', choices=('practice', 'held', 'all', 'verify', 'replicate', 'near-side'), default='all')
    parser.add_argument('--held', type=int, default=12)
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    first = args.folder/'recovery-practice.json.gz'
    second = args.folder/'recovery-held.json.gz'
    if args.phase in ('practice', 'all'):
        train(first)
    if args.phase in ('held', 'all'):
        evaluate(second, first, args.held)
    if args.phase == 'replicate':
        replicate(args.folder/'recovery-replication.json.gz', first)
    if args.phase == 'near-side':
        replicate(args.folder/'recovery-near-side.json.gz', first, near_side=True)
    if args.phase == 'verify':
        print('replayed', verify_receipts(args.folder))
