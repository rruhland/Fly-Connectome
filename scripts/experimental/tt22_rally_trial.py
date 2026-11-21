"""Exercise the optical TT22 receiver against real, cooperative joint-rate rivals."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt19_model_rollout import BallObservation
from .tt20_camera_carry import SETTINGS
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt20_opponent import install_opponent
from .tt21_coach import install_coach
from .tt22_rally import RallyPlayer, ReturnMemory


OUTPUT = ROOT/'artifacts/tt22'
PILOT = 612423769
PRACTICE = 1082201781
HELD = 1505337903
FRAME = np.array((0., 0., TABLE_HEIGHT))
CONFIG = dict(SETTINGS, view_mode='fixed', max_steps=1400)


class GentleDraws:
    def __init__(self, rng, spread, pace):
        self.rng, self.spread, self.pace = rng, spread, pace

    def uniform(self, low, high):
        middle = .5*np.add(low, high)
        sample = middle + self.spread * (self.rng.uniform(low, high) - middle)
        if np.ndim(sample) == 0 and float(low) == 2.58 and float(high) == 2.88:
            sample *= self.pace
        return sample


class RallyArena(TT20Arena):
    """Keep the referee and all colliders. Select the initial release family only."""

    def __init__(self, spread, pace):
        self.spread, self.pace = spread, pace
        super().__init__(**CONFIG)

    def _opening(self):
        if self.server == 1:
            return super()._opening()
        rng = self._rng
        self._rng = GentleDraws(rng, self.spread, self.pace)
        try:
            super()._opening()
        finally:
            self._rng = rng


def opponent(arena, mode):
    if mode == 'coach':
        return install_coach(arena)
    if mode == 'stock':
        return install_opponent(arena)
    from .tt22_opponent import install_rally_opponent

    return install_rally_opponent(arena, lane=mode)


def teacher_observation(arena):
    """Evaluator ablation with a deliberately privileged observation."""
    return BallObservation(arena.ball.copy()-FRAME, arena.velocity.copy(), arena.spin.copy(), arena._received_bounce)


def alternating(events):
    longest = current = 0
    previous = None
    for event in events:
        if event['kind'] != 'legal_return':
            continue
        side = event['side']
        current = current+1 if previous is not None and side == -previous else 1
        previous, longest = side, max(longest, current)
    return longest


class Station:
    def __init__(self, spread, pace, rival, *, arena=None):
        self.arena = RallyArena(spread, pace) if arena is None else arena
        self.rival = opponent(self.arena, rival)
        law = FlightLaw()
        self.player = RallyPlayer(law)
        self.optics = NaturalViewTransport('fixed', lens_fov=self.arena.lens_intrinsics(), law=law,
                                           flight=False, associate=True)

    def point(self, seed, *, teacher=False, memory=None, update_memory=False):
        arena, player = self.arena, self.player
        arena.reset(seed=seed, server=-1, render=False)
        player.reset()
        player.memory, player.update_memory = memory, update_memory
        before = None if memory is None else dict(count=len(memory.evidence), bias=memory.bias().tolist())
        tracker = self.optics.tracker()
        actions, optical = [], []
        for tick in range(arena.max_steps):
            tracker.update(arena.render_cameras(), tick=tick)
            ball = tracker.ball
            position = None if ball is None or not ball.observed or tracker.sources.get('ball_range') != 'stereo' else np.asarray(ball.position)
            if position is not None:
                optical.append((tick, position.tolist()))
            command = player.command(tick, position, teacher=teacher_observation(arena) if teacher else None)
            actions.append(command.tolist())
            _, reward, done, cap, _ = arena.step(command, render=False)
            if done or cap:
                break
        after = None if memory is None else dict(count=len(memory.evidence), bias=memory.bias().tolist())
        return dict(seed=seed, role=-1, reward=reward, actions=actions, optical=optical, before=before, after=after,
                    longest=alternating(arena.events), shots=player.shots, events=arena.events.copy(),
                    rival=self.rival.decisions.copy(), outcome=arena.point_result(), rgb=tracker.digest.hexdigest())

    def close(self):
        self.player.close()
        self.arena.close()


def replay(rows, spread, pace, mode):
    arena = RallyArena(spread, pace)
    opponent(arena, mode)
    try:
        for row in rows:
            arena.reset(seed=row['seed'], server=-1, render=False)
            for command in row['actions']:
                _, reward, _, _, _ = arena.step(command, render=False)
            if row['outcome'] != arena.point_result() or row['reward'] != reward or json.dumps(row['events']) != json.dumps(arena.events):
                raise ValueError(f"physical rally replay changed at {row['seed']}")
    finally:
        arena.close()
    return len(rows)


def qualify(args):
    checkpoint = None if args.checkpoint is None else _read(args.checkpoint)
    suffix = '-teacher' if args.teacher else '-carried' if checkpoint is not None else '-zero-guard' if args.zero_guard else ''
    if args.refine:
        suffix += '-refined'
    path = args.folder/f'rally-{args.rival}-{args.seed}{suffix}.json.gz'
    station = Station(args.spread, args.pace, args.rival)
    station.player.late_refine = args.refine
    report = dict(schema='tt22-physical-rally-v1', seed=args.seed, spread=args.spread, pace=args.pace, rival=args.rival,
                  camera='native stereo', privileged=args.teacher, arena=CONFIG, rows=[],
                  checkpoint_sha256=None if args.checkpoint is None else _hash(args.checkpoint),
                  zero_guard=args.zero_guard, late_refine=args.refine)
    seeds = np.random.default_rng(args.seed).integers(2**31, size=args.count).tolist()
    memory = ReturnMemory(guard_empty=True) if args.zero_guard else None
    try:
        if checkpoint is not None:
            if (args.rival, args.spread, args.pace) != (checkpoint['rival'], checkpoint['spread'], checkpoint['pace']):
                raise ValueError('the benchmark differs from the frozen practice fixture')
            if set(seeds) & {row['seed'] for row in checkpoint['practice']}:
                raise ValueError('the benchmark points overlap checkpoint practice')
            memory = ReturnMemory(checkpoint['evidence'])
        for seed in seeds:
            row = station.point(seed, teacher=args.teacher, memory=memory)
            report['rows'].append(row)
            print('rally', seed, row['longest'], row['outcome'], 'plans', len(row['shots']), flush=True)
            _write(path, report)
    finally:
        station.close()
    report['verified'] = replay(report['rows'], args.spread, args.pace, args.rival)
    _write(path, report)


def rates(rows):
    return dict(points=len(rows), at_least_ten=sum(row['longest'] >= 10 for row in rows),
                longest=max(row['longest'] for row in rows), alternating=[row['longest'] for row in rows],
                legal_returns=sum(row['outcome']['legal_returns'] for row in rows),
                rival_returns=sum(row['outcome']['rival_returns'] for row in rows),
                wins=sum(row['reward'] > 0 for row in rows), frames=sum(len(row['actions']) for row in rows))


def experiment(args):
    path = args.folder/f'rally-learning-{args.rival}.json.gz'
    practice = np.random.default_rng(PRACTICE).integers(2**31, size=args.practice).tolist()
    assess = np.random.default_rng(HELD).integers(2**31, size=args.count).tolist()
    pilot = set(np.random.default_rng(PILOT).integers(2**31, size=20).tolist())
    if len(set(practice+assess)) != len(practice)+len(assess) or set(practice+assess) & pilot:
        raise ValueError('pilot, practice, and held points must be distinct')
    memory = ReturnMemory()
    station = Station(args.spread, args.pace, args.rival)
    report = dict(schema='tt22-rally-learning-v1', spread=args.spread, pace=args.pace, rival=args.rival,
                  camera='native stereo', privileged=False, arena=CONFIG, practice_seed=PRACTICE, held_seed=HELD,
                  practice=[], pairs=[], checkpoint=[], running_evidence=[])
    try:
        for seed in practice:
            row = station.point(seed, memory=memory, update_memory=True)
            report['practice'].append(row)
            report['checkpoint'] = memory.evidence.copy()
            print('practice', seed, row['longest'], row['after'], flush=True)
            _write(path, report)
        checkpoint = 'rally-learning-checkpoint.json.gz' if args.rival == 'anchor' else f'rally-learning-{args.rival}-checkpoint.json.gz'
        _write(args.folder/checkpoint, dict(schema='tt22-rally-learning-checkpoint-1', rival=args.rival, spread=args.spread,
                                            pace=args.pace, practice_seed=PRACTICE, practice=report['practice'],
                                            evidence=report['checkpoint']))
        for i, seed in enumerate(assess):
            branches = ('frozen', 'online') if i%2 == 0 else ('online', 'frozen')
            pair = {name: station.point(seed, memory=memory if name == 'online' else None,
                                        update_memory=name == 'online') for name in branches}
            report['pairs'].append(pair)
            report['running_evidence'] = memory.evidence.copy()
            report['metrics'] = {name: rates([item[name] for item in report['pairs']]) for name in branches}
            print('held', seed, {name: row['longest'] for name, row in pair.items()}, pair['online']['after'], flush=True)
            _write(path, report)
    finally:
        station.close()
    rows = report['practice'] + [row for pair in report['pairs'] for row in pair.values()]
    report['verified'] = replay(rows, args.spread, args.pace, args.rival)
    _write(path, report)
    print(report['metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--rival', choices=('coach', 'stock', 'anchor', 'varied'), default='anchor')
    parser.add_argument('--seed', type=int, default=PILOT)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--spread', type=float, default=.30)
    parser.add_argument('--pace', type=float, default=.82)
    parser.add_argument('--learn', action='store_true')
    parser.add_argument('--practice', type=int, default=4)
    parser.add_argument('--teacher', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--zero-guard', action='store_true')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.verify:
        data = _read(args.verify)
        rows = data.get('rows')
        if rows is None:
            rows = data['practice'] + [row for pair in data['pairs'] for row in pair.values()]
        print('replayed', replay(rows, data['spread'], data['pace'], data['rival']))
    elif args.learn:
        if args.teacher or args.checkpoint is not None or args.zero_guard or args.refine:
            parser.error('the optical learning comparison requires unprivileged input')
        experiment(args)
    else:
        if sum((args.teacher, args.checkpoint is not None, args.zero_guard)) > 1:
            parser.error('choose one held intervention at a time')
        qualify(args)
