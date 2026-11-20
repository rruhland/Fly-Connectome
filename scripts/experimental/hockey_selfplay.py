"""Adapt a carried camera controller by playing equally mobile frozen rivals."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import numpy as np

from air_hockey import HEIGHT, MALLET_RADIUS
from air_hockey_experiment import ROOT, read, sha, state_hash, write
from air_hockey_learning import BOUNDS, GoalMemory, HockeyPlayer, REACH
from arcade_wall_learn import WallMemory, project_crossing
from hockey_head_to_head import TwoPlayers, opposite_view


TACTICS = ('carried', 'center', 'early_center', 'late_center',
           'early_edge', 'late_edge', 'late_bank')
DEPTHS = {'early_center': .420, 'late_center': .242, 'early_edge': .420,
          'late_edge': .242, 'late_bank': .275}
CONTEXTS = 4
SEED = 353149395
SCHEMA = 'camera-air-hockey-selfplay-v1'


class Tactics:
    """Update a small table of whole-point returns with the public final goal."""

    def __init__(self, state=None):
        if state is None:
            self.sums = np.zeros((CONTEXTS, len(TACTICS)))
            self.counts = np.zeros((CONTEXTS, len(TACTICS)), dtype=int)
        else:
            self.sums = np.asarray(state['sums'], dtype=float).copy()
            self.counts = np.asarray(state['counts'], dtype=int).copy()

    def state(self):
        return {'sums': self.sums.tolist(), 'counts': self.counts.tolist()}

    def select(self, context, rng, *, exploration=0.):
        counts = self.counts[context] + .45 * self.counts.sum(axis=0)
        sums = self.sums[context] + .45 * self.sums.sum(axis=0)
        score = sums / (counts + 3)
        # Keep the carried behavior when score evidence is absent.
        score[0] += .008
        if exploration:
            if rng.random() < exploration:
                return rng.choices(range(len(TACTICS)), weights=1 / np.sqrt(counts + 2))[0]
            score += .16 / np.sqrt(counts + 2)
        return int(np.argmax(score))

    def update(self, context, tactic, score):
        self.sums[context, tactic] += score
        self.counts[context, tactic] += 1


class FlexibleHockey(HockeyPlayer):
    """Use the carried visual tracker with a selected candidate approach."""

    def __init__(self, state, seed, tactic):
        super().__init__(WallMemory.from_state(state['wall']), GoalMemory.from_state(state['goal']),
                         seed=seed, fixed_center=tactic != 'carried')
        self.tactic = tactic
        self.lane = HEIGHT / 2

    def _new_aim(self, crossing):
        super()._new_aim(crossing)
        if self.tactic not in DEPTHS or self.tactic.endswith('center'):
            return
        rival_y = self.image.rival[1] if self.image.rival is not None else HEIGHT / 2
        relative = rival_y - HEIGHT / 2
        if abs(relative) < .035:
            relative = crossing - HEIGHT / 2
        side = -1 if relative > 0 else 1
        self.lane = HEIGHT / 2 + side * .078
        if self.tactic.endswith('bank'):
            wall = BOUNDS[0] if side < 0 else BOUNDS[1]
            self.lane = wall - (self.lane - wall) / max(.35, self.wall.coefficient)

    def _normal(self, crossing, plane, vx, vy, lunge):
        if self.tactic not in DEPTHS:
            return super()._normal(crossing, plane, vx, vy, lunge)
        # With the inherited center ray, a shifted intercept encodes the
        # chosen visible lane while preserving the same impulse planner.
        center_equivalent = crossing + HEIGHT / 2 - self.lane
        return super()._normal(center_equivalent, plane, vx, vy, lunge)

    def _intercept(self, x, y, vx, vy, player_x):
        if self.tactic not in DEPTHS:
            return super()._intercept(x, y, vx, vy, player_x)
        if x < .12 and player_x > x:
            return max(MALLET_RADIUS, x - .065), y
        plane = min(DEPTHS[self.tactic], max(.12, x - .014))
        crossing = project_crossing((x, y), (vx, vy), plane, BOUNDS, self.wall.coefficient)
        crossing = y if crossing is None else crossing
        next_y = project_crossing((x, y), (vx, vy), plane - .015, BOUNDS, self.wall.coefficient)
        terminal_vy = vy if next_y is None else (next_y - crossing) * -vx / .015
        if self.plan is None:
            self._new_aim(crossing)
        normal = self._normal(crossing, plane, vx, terminal_vy, .98)
        target_x = plane - REACH * math.cos(normal)
        target_y = min(HEIGHT - MALLET_RADIUS, max(MALLET_RADIUS, crossing - REACH * math.sin(normal)))
        if .004 < x - plane < .040 and abs(self.image.player[1] - target_y) < .014:
            target_x += .033
        return target_x, target_y


class AdaptivePlayer:
    """Use initial camera motion to select a tactic for the rest of the point."""

    def __init__(self, state, seed, *, memory=None, exploration=0., tactic=None):
        self.state = state
        self.seed = seed
        self.rng = random.Random(f'hockey self-play/{seed}')
        self.memory = memory
        self.exploration = exploration
        self.tactic = tactic
        self.context = None
        self.history = []
        default = 'carried' if tactic is None else TACTICS[tactic]
        self.player = FlexibleHockey(state, seed, default)

    def action(self):
        return self.player.action()

    def observe(self, frame, reward=0, terminated=False, truncated=False):
        self.player.observe(frame, reward, terminated, truncated)
        if self.tactic is not None:
            return
        self.history.append(frame)
        velocity = self.player.flight.velocity
        if len(self.history) < 5:
            return
        incoming = velocity is None or velocity[0] < 0
        puck = self.player.image.puck
        outer = puck is not None and abs(puck[1] - HEIGHT / 2) > .105
        self.context = int(outer) + 2 * int(incoming)
        self.tactic = self.memory.select(self.context, self.rng, exploration=self.exploration)
        if TACTICS[self.tactic] != 'carried':
            self.player = FlexibleHockey(self.state, self.seed, TACTICS[self.tactic])
            for image in self.history:
                self.player.observe(image)
        self.history.clear()


def play(job, *, record=False):
    case, state, memory_state, exploration, mode = job
    seed, side, rival = case['seed'], case['side'], case['rival']
    arena = TwoPlayers(seed)
    memory = Tactics(memory_state) if mode == 'adaptive' else None
    own = AdaptivePlayer(state, seed, memory=memory, exploration=exploration,
                         tactic=None if mode == 'adaptive' else TACTICS.index(mode))
    opponent = HockeyPlayer(WallMemory.from_state(state['wall']), GoalMemory(), seed=seed,
                            fixed_center=rival == 'center')
    agents = (own, opponent) if side == 0 else (opponent, own)
    frame = arena.frame()
    initial_sha = sha256(frame.tobytes()).hexdigest()
    frames = [frame] if record else None
    agents[0].observe(frame)
    agents[1].observe(opposite_view(frame))
    actions = bytearray()
    while not (arena.terminated or arena.truncated):
        left, right = (agent.action() for agent in agents)
        actions.extend((left, right))
        arena.right_action = right
        frame, reward, done, capped = arena.step(left)
        agents[0].observe(frame, reward, done, capped)
        agents[1].observe(opposite_view(frame), -reward, done, capped)
        if record and (arena.frames % 4 == 0 or done or capped):
            frames.append(frame)
    touches = [event['side'] for event in arena.events if event['kind'] == 'contact']
    body = 'player' if side == 0 else 'rival'
    reward = arena.reward if side == 0 else -arena.reward
    row = {**case, 'mode': mode, 'tactic': own.tactic, 'context': own.context, 'reward': reward,
           'frames': arena.frames, 'own_touches': touches.count(body), 'first_touch': bool(touches and touches[0] == body),
           'first_touch_goal': bool(reward > 0 and touches and touches[0] == body),
           'last_touch_goal': bool(reward > 0 and touches and touches[-1] == body),
           'initial_sha256': initial_sha, 'actions_sha256': sha256(actions).hexdigest(),
           'events_sha256': state_hash(arena.events)}
    return row, frames


def simulate(job):
    return play(job)[0]


def schedule(practice, holdout):
    rng = random.Random(SEED)
    seeds = rng.sample(range(100_000_000, 2**31-1), practice + holdout)
    learning = [dict(seed=seed, side=i % 2, rival='untrained' if (i // 2) % 2 else 'center')
                for i, seed in enumerate(seeds[:practice])]
    evaluation = [dict(seed=seed, side=side, rival=rival)
                  for seed in seeds[practice:] for rival in ('untrained', 'center') for side in (0, 1)]
    return {'practice': learning, 'heldout': evaluation}


def summarize(rows):
    return {'points': len(rows), 'goals': sum(row['reward'] > 0 for row in rows),
            'conceded': sum(row['reward'] < 0 for row in rows),
            'timeouts': sum(row['reward'] == 0 for row in rows),
            'net': sum(row['reward'] for row in rows),
            'goals_after_first_touch': sum(row['first_touch_goal'] for row in rows),
            'goals_after_last_touch': sum(row['last_touch_goal'] for row in rows),
            'total_first_touches': sum(row['first_touch'] for row in rows)}


def paired(adapted, baseline):
    key = lambda row: (row['seed'], row['rival'], row['side'], row['initial_sha256'])
    if list(map(key, adapted)) != list(map(key, baseline)):
        raise ValueError('matched camera starts required')
    cluster = {}
    for a, b in zip(adapted, baseline, strict=True):
        cluster.setdefault(a['seed'], []).append([a['reward'] - b['reward'],
                                                  int(a['first_touch_goal']) - int(b['first_touch_goal'])])
    values = np.array([np.mean(records, axis=0) for records in cluster.values()])
    rng = np.random.default_rng(230916)
    draws = values[rng.integers(len(values), size=(12000, len(values)))].mean(axis=1)
    return {metric: {'mean': float(values[:, i].mean()),
                      'interval_95': np.quantile(draws[:, i], [.025, .975]).tolist()}
            for i, metric in enumerate(('net_per_point', 'first_touch_goal'))}


def run(checkpoint, output, *, practice, heldout, workers):
    state = read(checkpoint)
    schedule_ = schedule(practice, heldout)
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), *(Path(__file__).with_name(name) for name in
             ('hockey_head_to_head.py', 'air_hockey.py', 'air_hockey_learning.py',
              'arcade_wall_learn.py', 'run_hockey_matches.py'))]
    registration = {'schema': SCHEMA, 'checkpoint_sha256': sha(checkpoint), 'schedule': schedule_,
                    'tactics': TACTICS, 'primary': 'net scored goals per point versus the carried policy',
                    'secondary': 'goals on points where this mallet made the first contact',
                    'files': {str(file.relative_to(ROOT)): sha(file) for file in files},
                    'update_batch_points': 12, 'heldout_frozen': True}
    write(output / 'registration.json', registration)

    memory = Tactics()
    practice_rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, practice, registration['update_batch_points']):
            exploration = .80 if offset < practice / 2 else .39
            group = schedule_['practice'][offset:offset + registration['update_batch_points']]
            jobs = [(case, state, memory.state(), exploration, 'adaptive') for case in group]
            for row in pool.map(simulate, jobs):
                memory.update(row['context'], row['tactic'], row['reward'])
                practice_rows.append(row)
            if offset % 60 == 0:
                print('practice', len(practice_rows), summarize(practice_rows), flush=True)
        retained = {'schema': SCHEMA, 'source': state, 'tactics': memory.state(),
                    'selected': [TACTICS[memory.select(context, random.Random(0))] for context in range(CONTEXTS)],
                    'registration_sha256': sha(output / 'registration.json')}
        write(output / 'checkpoint.json', retained)
        write(output / 'practice.json', practice_rows)
        modes = ('adaptive', 'carried', 'center')
        jobs = [(case, state, memory.state(), 0., mode) for case in schedule_['heldout'] for mode in modes]
        results = {mode: [] for mode in modes}
        for row in pool.map(simulate, jobs, chunksize=4):
            results[row['mode']].append(row)

    summary = {}
    for rival in ('all', 'untrained', 'center'):
        subset = {mode: [row for row in rows if rival == 'all' or row['rival'] == rival]
                  for mode, rows in results.items()}
        summary[rival] = {'scores': {mode: summarize(rows) for mode, rows in subset.items()},
                          'paired': {mode: paired(subset['adaptive'], subset[mode])
                                     for mode in ('carried', 'center')}}
    write(output / 'heldout.json', dict(rows=results, summary=summary,
                                        checkpoint_sha256=sha(output / 'checkpoint.json')))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--practice', type=int, default=360)
    parser.add_argument('--heldout', type=int, default=48, help='seeds, each replayed from both sides against two rivals')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if min(args.practice, args.heldout, args.workers) < 1:
        parser.error('positive practice, evaluation and worker counts required')
    run(args.checkpoint, args.output, practice=args.practice, heldout=args.heldout, workers=args.workers)


if __name__ == '__main__':
    main()
