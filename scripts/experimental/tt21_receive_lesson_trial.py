"""One-camera receiving practice and frozen held tests on the normal TT20 clamp."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt19_flight import optical_start
from .tt19_model_rollout import DrySimulator, RolloutPlayer
from .tt20_camera_carry import SETTINGS
from .tt20_curriculum import OUTPUT as FACTORY, configuration, model_arena
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt20_opponent import install_opponent
from .tt21_receive_lesson import StrikeMemory, VARIANTS, candidate, compare_proposals, strike_cue, visual_credit


OUTPUT = ROOT/'artifacts/tt21'
PRACTICE = 'receive-lesson-practice.json.gz'
HELD = 'receive-lesson-held.json.gz'
GUARDED = 'receive-lesson-guarded.json.gz'
PRACTICE_SEED = 1068045844
HELD_SEED = 2273458831
SPREAD_SEED = 2971039016
GUARDED_SEED = 1340765032
EXPLORATION_SEED = 1632780470
LIMITED_WIDTH = .28
EYE = 'left'


class CenteredDraws:
    """Narrow the existing serve-release random variables within their ranges."""

    def __init__(self, source):
        self.source = source

    def uniform(self, low, high):
        center = .5 * np.add(low, high)
        return center + LIMITED_WIDTH * (self.source.uniform(low, high) - center)


class LessonArena(TT20Arena):
    def __init__(self, *, spread):
        self.spread = spread
        super().__init__(**SETTINGS, view_mode=EYE)

    def _opening(self):
        if self.spread or self.server != -1:
            super()._opening()
            return
        source = self._rng
        self._rng = CenteredDraws(source)
        try:
            super()._opening()
        finally:
            self._rng = source


def apparatus(spread):
    arena = LessonArena(spread=spread)
    install_opponent(arena)
    return arena


class Station:
    def __init__(self, spread=False):
        self.arena = apparatus(spread)
        _, catalogue, law, response, _ = configuration(FACTORY)
        self.dry = RolloutPlayer(catalogue, law, response)
        self.dry.sim.close()
        self.dry.sim = DrySimulator(law, arena=model_arena())
        self.sensor = NaturalViewTransport(EYE, lens_fov={EYE: self.arena.lens_intrinsics()[EYE]}, law=law)

    def close(self):
        self.dry.close()
        self.arena.close()

    def point(self, seed, memory, variant, *, allow_recall, guarded=False):
        arena, dry = self.arena, self.dry
        arena.reset(seed=seed, server=-1, render=False)
        dry.reset()
        tracker = self.sensor.tracker()
        sightings, commands = [], []
        movement = np.zeros(5)
        stroke = None
        guard = []
        for tick in range(arena.max_steps):
            if (stroke is None and tick <= 28 or
                    stroke is not None and stroke['start']-2 <= tick <= stroke['start']+44 and tick % 2 == 0):
                images = arena.render_cameras()
                if tuple(images) != (EYE,):
                    raise ValueError('the receiving policy received multiple streams')
                tracker.update(images, tick=tick)
                point = tracker.ball
                position = None if point is None or not point.observed else np.asarray(point.position)
                dry.history.append((tick, position))
                if position is not None:
                    sightings.append((tick, position.tolist()))
                if stroke is None and tick >= 8 and position is not None and -.81 < position[1] < -.24:
                    speed = tracker.ball_velocity
                    estimate = (position, speed) if speed is not None else optical_start(dry.history, dry.law.gravity)
                    if estimate is not None and estimate[1][1] < -.031:
                        cue = strike_cue(dry.law, estimate, tick)
                        if cue is not None:
                            recalled = memory.retrieve(cue, tick) if allow_recall else None
                            if guarded or recalled is None:
                                stock = candidate(dry, estimate, cue, tick, variant)
                                stroke = stock
                                if stock is not None and recalled is not None:
                                    stroke, decision = compare_proposals(dry, estimate, tick, stock, recalled)
                                    guard.append(decision)
                            else:
                                stroke = recalled
            action = np.zeros(6)
            if stroke is not None and tick >= stroke['start']:
                age = tick - stroke['start']
                action[:5] = stroke['rates'][age] if age < len(stroke['rates']) else -np.clip(movement, -1., 1.)
                movement += action[:5]
            commands.append(action.tolist())
            _, reward, ended, capped, _ = arena.step(action, render=False)
            if ended or capped:
                break
        return dict(seed=seed, variant=variant, recall_enabled=allow_recall, actions=commands,
                    stroke=stroke, guard=guard, rgb_sha256=tracker.digest.hexdigest(), optical=sightings,
                    optical_credit=visual_credit(sightings, reward, stroke, dry.law), reward=reward,
                    result=arena.point_result(), events=arena.events)


def starts(seed, count):
    schedule = np.random.default_rng(seed).integers(2**31, size=count).tolist()
    earlier = previous_seeds(FACTORY) | previous_seeds(ROOT/'artifacts/tt19')
    if len(set(schedule)) != count or set(schedule) & earlier:
        raise ValueError('the receiving lesson schedule is not independent of older starts')
    return schedule


def summarize(rows):
    return dict(points=len(rows), first_returns=sum(row['result']['legal_returns'] > 0 for row in rows),
                hits=sum(row['result']['contacts'] > 0 for row in rows),
                legal_rival_serves=sum(row['result']['legal_serve'] for row in rows),
                wins=sum(row['reward'] > 0 for row in rows), frames=sum(len(row['actions']) for row in rows),
                attempted=sum(row['stroke'] is not None for row in rows),
                recalled=sum(row['stroke'] is not None and 'recalled' in row['stroke'] for row in rows))


def train(folder):
    schedule = starts(PRACTICE_SEED, 24)
    chooser = np.random.default_rng(EXPLORATION_SEED)
    variants = np.concatenate([chooser.permutation(len(VARIANTS)) for _ in range(4)]).tolist()
    policy, station, rows = StrikeMemory(), Station(), []
    report = dict(schema='tt21-receive-lesson-practice-1', arena=SETTINGS, eye=EYE, serve_width_fraction=LIMITED_WIDTH,
                  seed=PRACTICE_SEED, exploration_seed=EXPLORATION_SEED, variants=VARIANTS, schedule=schedule,
                  factory_sha256=_hash(FACTORY/'curriculum-factory.json.gz'), starting_examples=0,
                  outcome_inputs=['left RGB optical trajectory', 'point reward'], complete=False)
    try:
        for i, (seed, variant) in enumerate(zip(schedule, variants, strict=True)):
            row = station.point(seed, policy, variant, allow_recall=i >= 8 and i % 2 == 1)
            policy.observe(row['stroke'], row['optical_credit'], i)
            rows.append(row)
            print('practice', i+1, variant, row['result']['legal_returns'],
                  round(row['optical_credit'], 3), len(policy.rows), flush=True)
            if (i+1) % 6 == 0:
                report.update(rows=rows, checkpoint=policy.rows, summary=summarize(rows))
                _write(folder/PRACTICE, report)
    finally:
        station.close()
    report.update(complete=True, verified=replay(rows, spread=False))
    _write(folder/PRACTICE, report)
    return report


def assess_section(practice, *, spread, guarded=False):
    seed = GUARDED_SEED if guarded else SPREAD_SEED if spread else HELD_SEED
    schedule = starts(seed, 12 if spread or guarded else 20)
    if set(schedule) & set(practice['schedule']):
        raise ValueError('the frozen test reuses practice tosses')
    station, memory = Station(spread), StrikeMemory(practice['checkpoint'])
    rows = []
    try:
        for i, seed in enumerate(schedule):
            branches = ('cold', 'learned') if i % 2 == 0 else ('learned', 'cold')
            pair = {branch: station.point(seed, memory, 0, allow_recall=branch == 'learned',
                                          guarded=guarded and branch == 'learned') for branch in branches}
            rows.append(pair)
            print('guarded' if guarded else 'spread' if spread else 'limited', i+1, seed,
                  {branch: row['result']['legal_returns'] for branch, row in pair.items()}, flush=True)
    finally:
        station.close()
    summary = {branch: summarize([pair[branch] for pair in rows]) for branch in ('cold', 'learned')}
    gains, losses = [], []
    for pair in rows:
        cold, ready = (bool(pair[mode]['result']['legal_returns']) for mode in ('cold', 'learned'))
        if cold != ready:
            (gains if ready else losses).append(pair['cold']['seed'])
    return dict(schedule=schedule, pairs=rows, metrics=summary, gains=gains, losses=losses,
                verified=replay([row for pair in rows for row in pair.values()], spread=spread))


def held(folder, practice):
    if not practice['complete'] or practice['factory_sha256'] != _hash(FACTORY/'curriculum-factory.json.gz'):
        raise ValueError('the frozen policy needs the complete lesson and factory checkpoint')
    report = dict(schema='tt21-receive-lesson-held-1', source=PRACTICE, source_sha256=_hash(folder/PRACTICE),
                  limited_seed=HELD_SEED, spread_seed=SPREAD_SEED, spread_trigger=3,
                  learned_examples=len(practice['checkpoint']))
    limited = report['limited'] = assess_section(practice, spread=False)
    _write(folder/HELD, report)
    print('limited summary', limited['metrics'], flush=True)
    if len(limited['gains']) - len(limited['losses']) >= report['spread_trigger']:
        report['spread'] = assess_section(practice, spread=True)
        _write(folder/HELD, report)
        print('spread summary', report['spread']['metrics'], flush=True)


def guarded_test(folder, practice):
    original = _read(folder/HELD)
    planned = set(starts(GUARDED_SEED, 12))
    prior = {seed for mode in ('limited', 'spread') if mode in original for seed in original[mode]['schedule']}
    if planned & prior or not practice['complete'] or original['source_sha256'] != _hash(folder/PRACTICE):
        raise ValueError('the guarded comparison requires the same completed lesson and new tosses')
    section = assess_section(practice, spread=False, guarded=True)
    _write(folder/GUARDED, dict(schema='tt21-guarded-receiving-lesson-1', seed=GUARDED_SEED, result=section,
                                practice_sha256=_hash(folder/PRACTICE), original_sha256=_hash(folder/HELD)))
    print('guarded summary', section['metrics'], flush=True)


def replay(rows, *, spread):
    arena = apparatus(spread)
    try:
        for row in rows:
            arena.reset(seed=row['seed'], server=-1, render=False)
            for action in row['actions']:
                _, reward, _, _, _ = arena.step(action, render=False)
            if (row['result'] != arena.point_result() or row['reward'] != reward
                    or json.dumps(row['events']) != json.dumps(arena.events)):
                raise ValueError(f"receiving lesson replay changed at {row['seed']}")
    finally:
        arena.close()
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=('both', 'train', 'held', 'guarded', 'verify'), default='both')
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.stage == 'verify':
        practice = _read(args.folder/PRACTICE)
        report = _read(args.folder/HELD)
        print('practice', replay(practice['rows'], spread=False))
        for label in ('limited', 'spread'):
            if label in report:
                print(label, replay([row for pair in report[label]['pairs'] for row in pair.values()], spread=label == 'spread'))
        if (args.folder/GUARDED).exists():
            guarded = _read(args.folder/GUARDED)['result']
            print('guarded', replay([row for pair in guarded['pairs'] for row in pair.values()], spread=False))
    else:
        practice = train(args.folder) if args.stage in ('both', 'train') else _read(args.folder/PRACTICE)
        if args.stage in ('both', 'held'):
            held(args.folder, practice)
        elif args.stage == 'guarded':
            guarded_test(args.folder, practice)
