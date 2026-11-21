"""Evaluate a hidden terminal-payout reversal with matched visual motor learners."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from scripts.experimental.tt3_learning_stroke import StrokeMemory
from tt3_arena import TT3Arena, VARIANTS, heldout_cases
from tt3_learning import ReturnMemory, play_point
from tt3_numeric_transfer import VisualPlayer
from tt3_visual import RoleModel


CONDITIONS = ('reversed_live', 'reversed_locked', 'unchanged_live')
OPTIONS = ('project', 'pursue', 'hold', 'recede')
METRICS = ('public_reward', 'landed', 'points_for', 'points_against', 'contacts',
           'legal_returns', 'rival_returns', 'probes', 'frames')


class RecentUtility:
    """Assign recency-weighted terminal payouts to alternative visual relations."""
    def __init__(self, return_state, offsets):
        source = ReturnMemory.restore(return_state)
        means = source.means[:len(OPTIONS)].copy()
        # The acquired stroke has a zero offset for the positive and negative aliases.
        # Reuse the strongest measured confidence for that shared projection option.
        if offsets[1] == -1.:
            means[0] = max(means[0], source.means[4], source.means[5])
        self.counts = np.minimum(4., source.counts[:len(OPTIONS)])
        self.scores = means * self.counts
        self.learn = True
        self.stroke = StrokeMemory(source, offsets)

    def choose(self, rng, explore=0.):
        if rng.random() < explore:
            return int(rng.integers(len(OPTIONS)))
        evidence = self.scores / np.maximum(1., self.counts)
        utility = evidence + .38 / np.sqrt(1. + self.counts)
        candidates = np.flatnonzero(np.isclose(utility, utility.max()))
        return int(rng.choice(candidates))

    def observe(self, option, reward):
        if not self.learn:
            return
        self.counts *= .78
        self.scores *= .78
        self.scores[option] += float(np.clip(reward, -1., 1.))
        self.counts[option] += 1.

    def goal(self, option, *args, **kwargs):
        return self.stroke.goal(option, *args, **kwargs)

    def relation_name(self, option):
        return OPTIONS[option]

    def state(self):
        return dict(options=OPTIONS, scores=self.scores.tolist(), counts=self.counts.tolist(),
                    learn=self.learn)


def terminal_payout(arena, objective, sign):
    def payout(reward, done):
        if not done:
            return 0.
        value = reward if objective == 'score' else 2 * (arena.legal_returns > 0) - 1
        return float(sign * value)

    return payout


def _point(case, learner, *, view_seed, objective, sign):
    arena = TT3Arena(case.seed, view_seed=view_seed, variant=case.variant)
    result = play_point(arena, learner, reward_map=terminal_payout(arena, objective, sign))
    events = result.pop('events')
    result['landed'] = int(result['legal_returns'] > 0)
    result['legal_landing_locations'] = [event['location'] for event in events
        if event['kind'] == 'legal_return' and event['side'] == 1]
    result['point_reason'] = arena.point_result()['reason']
    result['payout_sign'] = sign
    result['case'] = dict(asdict(case), view_seed=view_seed)
    return result


def one_course(job):
    index, seed, objective, warmup, early, late, checkpoint = job
    variant = VARIANTS[index % len(VARIANTS)]
    cases = heldout_cases(f'{seed}/{objective}/{index}', warmup + early + late, variant=variant)
    view_seed = cases[0].view_seed
    memory = RecentUtility(checkpoint['return_memory'], checkpoint['stroke'])
    player = VisualPlayer(RoleModel(**checkpoint['role_model']), memory, seed=seed + index,
                          explore=.16, forecast=True, recall=True, learn_roles=True)
    initial = memory.state()
    started = time.monotonic()
    prefix = [_point(case, player, view_seed=view_seed, objective=objective, sign=1)
              for case in cases[:warmup]]
    at_switch = memory.state()
    runs = {}
    for condition in CONDITIONS:
        branch = deepcopy(player)
        branch.memory.learn = condition != 'reversed_locked'
        sign = 1 if condition == 'unchanged_live' else -1
        points = [_point(case, branch, view_seed=view_seed, objective=objective, sign=sign)
                  for case in cases[warmup:]]
        runs[condition] = dict(points=points, memory=branch.memory.state())
    blind_actions = {run['points'][0]['action_sha256'] for run in runs.values()}
    if len(blind_actions) != 1:
        raise AssertionError('the first switched point changed behavior before feedback')
    if any(runs['reversed_locked']['memory'][key] != at_switch[key] for key in ('scores', 'counts')):
        raise AssertionError('locked objective preferences changed after the switch')
    return dict(course=index, variant=variant, view_seed=view_seed, prefix=prefix,
                initial_memory=initial, memory_at_switch=at_switch, arms=runs,
                wall_seconds=time.monotonic() - started)


def _aggregate(rows):
    return dict(points=len(rows), **{metric: sum(row[metric] for row in rows) for metric in METRICS},
                choices=dict(Counter(row['option'] for row in rows)),
                sustained=sum(row['legal_returns'] >= 2 for row in rows),
                switches_to_hold_or_recede=sum(row['option'] in ('hold', 'recede') for row in rows))


def _paired(courses, start, end, field, comparator):
    scores = []
    for course in courses:
        live = course['arms']['reversed_live']['points'][start:end]
        other = course['arms'][comparator]['points'][start:end]
        if [row['case'] for row in live] != [row['case'] for row in other]:
            raise AssertionError('the paired result has different starts')
        scores.append(np.mean([first[field] - second[field] for first, second in zip(live, other)]))
    rng = np.random.default_rng(722339)
    draw = rng.choice(scores, (5000, len(courses)), replace=True)
    return dict(mean=float(np.mean(scores)), ci95=np.quantile(draw.mean(axis=1), [.025, .975]).tolist(),
                course_deltas=scores)


def summarize(courses, early):
    out = {'prefix': _aggregate([row for course in courses for row in course['prefix']])}
    for name, start, end in (('early', 0, early), ('late', early, None)):
        out[name] = {arm: _aggregate([row for course in courses
            for row in course['arms'][arm]['points'][start:end]]) for arm in CONDITIONS}
        out[name]['paired_reversed_live_minus_locked'] = {
            key: _paired(courses, start, end, key, 'reversed_locked')
            for key in ('public_reward', 'landed', 'points_for')}
        out[name]['paired_reversed_live_minus_unchanged'] = {
            key: _paired(courses, start, end, key, 'unchanged_live')
            for key in ('landed', 'points_for')}
    return out


def _load(path):
    open_checkpoint = gzip.open if path.suffix == '.gz' else Path.open
    with open_checkpoint(path, 'rt') as handle:
        payload = json.load(handle)
    return {key: payload[key] for key in ('role_model', 'return_memory', 'stroke')}


def run(args):
    if min(args.courses, args.warmup, args.early, args.late, args.workers) < 1:
        raise ValueError('course, phase and worker counts must be positive')
    sources = [Path(__file__).with_name(name) for name in ('tt3_reward_change.py', 'tt3_learning.py',
        'tt3_numeric_transfer.py', 'tt3_arena.py', 'tt3_visual.py', 'tt3_motor.py', 'tt3_continuous.py',
        'tt3_flight.py', 'tt3_learning_stroke.py')]
    hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in sources}
    checkpoint = _load(args.checkpoint)
    jobs = [(i, args.seed, args.objective, args.warmup, args.early, args.late, checkpoint)
            for i in range(args.courses)]
    if args.workers == 1:
        courses = []
        for job in jobs:
            courses.append(one_course(job))
            print(f'{args.objective} course {len(courses)}/{len(jobs)} complete', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            courses = list(pool.map(one_course, jobs))
    stats = summarize(courses, args.early)
    receipt = dict(seed=args.seed, objective=args.objective, warmup=args.warmup,
                   early=args.early, late=args.late, source=checkpoint, source_sha256=hashes,
                   checkpoint_sha256=sha256(args.checkpoint.read_bytes()).hexdigest(),
                   summary=stats, courses=courses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, 'wt') if args.out.suffix == '.gz' else args.out.open('w') as handle:
        json.dump(receipt, handle, separators=(',', ':'))
        handle.write('\n')
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--objective', choices=('score', 'landing'), default='score')
    parser.add_argument('--seed', type=int, default=71530695)
    parser.add_argument('--courses', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--early', type=int, default=6)
    parser.add_argument('--late', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1)
    run(parser.parse_args())
