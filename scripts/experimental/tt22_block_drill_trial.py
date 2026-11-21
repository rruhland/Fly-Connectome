"""Screen a physical block and learn a fixed near posture from return credit."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt22_block_drill import (BlockArena, BlockCoach, DEVELOP, FarTape, HOME, PREP_TICKS, first_return,
                               play_block, prepare, reference_pulse, screen)
from .tt22_rally_trial import alternating


OUTPUT = ROOT / 'artifacts' / 'tt22'
PRACTICE = 127070327
ASSESS = 1758374125
TRANSFER = 1791114522
MOTOR_AXES = np.array([1, 2, 3])


class RewardSearch:
    """Explore commands around joint home. Learn only from the public return bit."""

    def __init__(self, low, high):
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.home = HOME[MOTOR_AXES]
        self.generator = qmc.Sobol(3, scramble=True, seed=508154)
        self.rng = np.random.default_rng(15568913)
        self.table = {}

    def _quality(self, value):
        wins, n = self.table[value]
        return (wins + .32) / (n + 1.20), wins

    def best(self):
        if not self.table:
            return tuple(self.home)
        return max(self.table, key=self._quality)

    def choose(self, index):
        winners = any(wins > 0 for wins, _ in self.table.values())
        if not winners or index % 6 == 0:
            u = self.generator.random(1).ravel()
            q = self.low - .14 + (self.high - self.low + .28) * u
        elif index % 3 != 0:
            q = self.best()
        else:
            q = self.best() + self.rng.normal(0., (.085,.102,.098))
        return np.clip(q, self.low, self.high)

    def observe(self, candidate, legal):
        key = tuple(candidate)
        wins, n = self.table.get(key, (0, 0))
        self.table[key] = [wins+int(legal), n+1]

    def supported(self):
        return self.table.get(self.best(), [0, 0])[0] >= 7


def posture(offset):
    target = HOME.copy()
    target[MOTOR_AXES] = offset
    return target


def _qualify_reference(refs, path, limit):
    rows = []
    for reference in refs[:limit]:
        arena = BlockArena(reference['opening'])
        coach = BlockCoach(arena, reference['joints'])
        try:
            row = play_block(arena, DEVELOP, reference['joints'], FarTape(arena), coach, settled=True)
            rows.append(dict(reference=reference, row=row, replayed=_replay_full(reference['opening'], [row]),
                             legal_alternating=alternating(row['events'])))
            _write(path, dict(schema='tt22-block-witness-1', source_seed=DEVELOP, rows=rows, replayed=len(rows)))
            print('reference', reference['designed'], row['outcome'], flush=True)
            if row['outcome']['legal_returns'] >= 3:
                break
        finally:
            coach.close()
            arena.close()
    return max(rows, key=lambda result: (result['legal_alternating'], result['row']['outcome']['legal_returns']))


def train(opening, budget, path, *, release_seed=None):
    arena = BlockArena(opening, release_seed=release_seed)
    drives = arena._arm_ids[1][2][MOTOR_AXES]
    low = np.maximum(arena._joint_low[drives], HOME[MOTOR_AXES] - 1.2)
    high = np.minimum(arena._joint_high[drives], HOME[MOTOR_AXES] + 1.2)
    search = RewardSearch(low, high)
    seeds = np.random.default_rng(PRACTICE).integers(2**31, size=budget).tolist()
    rows = []
    start = perf_counter()
    try:
        for index, seed in enumerate(seeds):
            candidate = search.choose(index)
            row = first_return(arena, seed, posture(candidate), settled=True)
            search.observe(candidate, row['legal'])
            rows.append(row)
            if row['legal'] or (index+1) % 16 == 0:
                print('scratch', index+1, row['legal'], search.best(), search.table[search.best()], flush=True)
            if index >= 31 and search.supported():
                break
    finally:
        arena.close()
    selected = search.best()
    data = dict(schema='tt22-reward-only-block-1', opening=opening, release_seed=release_seed,
                learner_used_pose_labels=False,
                public_axes=MOTOR_AXES.tolist(), starting_goal=HOME.tolist(), search_uses='binary strict return credit',
                searched_bounds=[low.tolist(), high.tolist()], practice_generator=PRACTICE,
                practice=rows, preparations=sum(len(row['preparation']) for row in rows),
                scored_frames=sum(len(row['actions']) for row in rows), selected=posture(selected).tolist(),
                selected_stats=search.table.get(selected, [0,0]), elapsed=perf_counter()-start)
    _write(path, data)
    return data


def _replay_full(opening, rows, *, release_seed=None):
    arena = BlockArena(opening, release_seed=release_seed)
    try:
        for row in rows:
            warmup = prepare(arena, row['seed'], row['goal'])
            if warmup != row['preparation']:
                raise ValueError('motor preparation changed')
            for index, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
                arena._rival_attempt = True
                arena._aim = lambda *args: np.asarray(far)
                _, reward, done, cap, _ = arena.step(near, render=False)
                if (done or cap) != (index + 1 == len(row['actions'])):
                    raise ValueError('the physical tapes changed when the point stopped')
            if (arena.point_result() != row['outcome'] or reward != row['reward']
                    or digest_json(arena.events) != digest_json(row['events'])):
                raise ValueError('the two physical command tapes changed the replay')
    finally:
        arena.close()
    return len(rows)


def qualify_push(reference, path):
    """Check a bounded set of prepared optical pulse recipes on the development release."""
    rows = []
    opening, posture = reference['opening'], reference['joints']
    for cue, pace in product((-.63, -.71, -.56), (.04, .065)):
        pulse = reference_pulse(posture, cue, pace)
        arena = BlockArena(opening)
        coach = BlockCoach(arena, posture, pulse=pulse)
        try:
            row = play_block(arena, DEVELOP, posture, FarTape(arena), coach, settled=True, pulse=pulse)
            row.update(pulse=pulse, longest=alternating(row['events']))
            rows.append(row)
            print('prepared push', cue, pace, row['longest'], row['outcome'], flush=True)
        finally:
            coach.close()
            arena.close()
    _write(path, dict(reference=reference, rows=rows, replayed=_replay_full(opening, rows)))


def evaluate(reference, learned, path, first_count, full_count):
    rng = np.random.default_rng(ASSESS)
    seeds = rng.integers(2**31, size=first_count + full_count).tolist()
    if set(seeds) & ({row['seed'] for row in learned['practice']} | {DEVELOP}) or len(seeds) != len(set(seeds)):
        raise ValueError('practice and held schedules overlap')
    opening = reference['opening']
    release_seed = learned['release_seed']
    targets = {'neutral': HOME.tolist(), 'learned': learned['selected'], 'reference': reference['joints']}
    data = dict(schema='tt22-block-assessment-1', opening=opening, release_seed=release_seed,
                prep_ticks=PREP_TICKS, targets=targets,
                seed_generator=ASSESS, starts=seeds, first_count=first_count, first={}, uninterrupted={})
    for mode, target in targets.items():
        arena = BlockArena(opening, release_seed=release_seed)
        try:
            data['first'][mode] = [first_return(arena, seed, target, settled=True) for seed in seeds[:first_count]]
        finally:
            arena.close()
        print('held initial returns', mode, sum(row['legal'] for row in data['first'][mode]), first_count, flush=True)
        _write(path, data)
    data['replayed_points'] = 0
    for mode, target in targets.items():
        arena = BlockArena(opening, release_seed=release_seed)
        coach = BlockCoach(arena, reference['joints'])
        try:
            tape = FarTape(arena)
            rows = data['uninterrupted'][mode] = []
            for seed in seeds[first_count:]:
                row = play_block(arena, seed, target, tape, coach, settled=True)
                rows.append(row)
                print('held uninterrupted', mode, seed, alternating(row['events']), row['outcome'], flush=True)
                _write(path, data)
        finally:
            coach.close()
            arena.close()
        data['replayed_points'] += _replay_full(opening, rows, release_seed=release_seed)
    _write(path, data)


def transfer(source, path):
    saved = _read(source)
    seeds = np.random.default_rng(TRANSFER).integers(2**31, size=12).tolist()
    if set(seeds) & {row['seed'] for row in saved['practice']}:
        raise ValueError('the jittered transfer points overlap practice')
    targets = dict(neutral=HOME, carried=np.asarray(saved['selected']))
    data = dict(schema='tt22-block-fixed-to-jitter-1', source=source.name, sha256=_hash(source),
                generator=TRANSFER, seeds=seeds, opening=saved['opening'], arms={})
    for mode, target in targets.items():
        arena = BlockArena(saved['opening'])
        try:
            data['arms'][mode] = [first_return(arena, seed, target, settled=True) for seed in seeds]
        finally:
            arena.close()
    _write(path, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=('screen', 'witness', 'push', 'learn', 'transfer'), required=True)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--reference-index', type=int, default=0)
    parser.add_argument('--budget', type=int, default=144)
    parser.add_argument('--first-count', type=int, default=12)
    parser.add_argument('--full-count', type=int, default=3)
    parser.add_argument('--fixed-release', action='store_true')
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    reference_path = args.folder / 'tt22_block_drill-screen.json.gz'
    if args.operation == 'screen':
        _write(reference_path, screen(settled=True))
    elif args.operation == 'transfer':
        transfer(args.folder/'tt22_block_drill-scratch-fixed.json.gz', args.folder/'tt22_block_drill-transfer.json.gz')
    else:
        references = _read(reference_path)['valid']
        if args.operation == 'witness':
            _qualify_reference(references, args.folder/'tt22_block_drill-witness.json.gz', 6)
        elif args.operation == 'push':
            qualify_push(references[args.reference_index], args.folder/'tt22_block_drill-push.json.gz')
        else:
            reference = references[args.reference_index]
            suffix = '-fixed' if args.fixed_release else ''
            learned = train(reference['opening'], args.budget, args.folder/f'tt22_block_drill-scratch{suffix}.json.gz',
                            release_seed=DEVELOP if args.fixed_release else None)
            evaluate(reference, learned, args.folder/f'tt22_block_drill-held{suffix}.json.gz',
                     args.first_count, args.full_count)
