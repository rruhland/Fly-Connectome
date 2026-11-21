"""Test a credited taught stroke repertoire on new strict single-eye receive points."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from scipy.stats import binomtest

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt20_camera_carry import SETTINGS, Station, summarize, verify
from .tt21_demonstration import LIMIT, SOURCE, distill


OUTPUT = ROOT / 'artifacts/tt21'
DEST = OUTPUT / 'demonstration-held.json.gz'
SCHEDULE = 2087154831
MODES = ('frozen', 'taught', 'shuffled')
ATLAS = SOURCE / 'curriculum-dry-catalog.npz'


class PlannerBudget:
    def __init__(self, player):
        self.original = player.choose
        self.reset()

    def reset(self):
        self.seconds = 0.
        self.requests = 0

    def choose(self, estimate):
        self.requests += 1
        start = process_time()
        try:
            return self.original(estimate)
        finally:
            self.seconds += process_time() - start


def _point(station, budget, seed, condition, indexes):
    planner = station.player.player
    budget.reset()
    start_eval = planner.evaluated
    started = perf_counter()
    row = station.point(seed, -1, 0)
    row['budget'] = dict(wall_s=perf_counter() - started, planner_cpu_s=budget.seconds,
                         requests=budget.requests, physical_rollouts=planner.evaluated - start_eval)
    row['plans'] = [dict(plan, index=int(indexes[plan['index']])) for plan in planner.plans]
    row['condition'] = condition
    return row


def _pair(rows, alternative, threshold):
    changes = Counter((pair['frozen']['outcome']['legal_returns'] >= threshold,
                       pair[alternative]['outcome']['legal_returns'] >= threshold) for pair in rows)
    gain, loss = changes[False, True], changes[True, False]
    return dict(threshold=threshold, gained=gain, lost=loss, paired=(gain, loss),
                exact_p=float(binomtest(gain, gain+loss).pvalue) if gain+loss else 1.)


def _metrics(rows):
    comparisons = {mode: {'returns_ge_1': _pair(rows, mode, 1), 'returns_ge_2': _pair(rows, mode, 2)}
                   for mode in ('taught', 'shuffled')}
    metrics = {}
    for mode in MODES:
        points = [row[mode] for row in rows]
        metrics[mode] = dict(**summarize(points),
                             **{key: sum(row['budget'][key] for row in points) for key in points[0]['budget']},
                             contacts=sum(any(e['kind']=='contact' and e.get('side')==1 for e in row['events'])
                                          for row in points),
                             unique_planned_tapes=len({plan['index'] for row in points for plan in row['plans']}))
    return metrics, comparisons


def _fresh(count, source_seeds):
    found = set(source_seeds)
    for folder in (ROOT/'artifacts/tt19', SOURCE, OUTPUT):
        found.update(previous_seeds(folder))
    # A rerun must reproduce its predetermined evaluation schedule.
    if DEST.exists():
        found.difference_update(_read(DEST)['held_seeds'])
    seeds = np.random.default_rng(SCHEDULE).integers(2**31, size=count).tolist()
    if len(set(seeds)) != len(seeds) or set(seeds) & found:
        raise ValueError('the held schedule overlaps a pre-existing source or practice')
    return seeds


def run(output, count):
    output.parent.mkdir(exist_ok=True)
    with np.load(ATLAS) as archive:
        start = process_time()
        memory = distill(archive['tapes'], LIMIT)
        teacher_cpu_s = process_time() - start
    held = _fresh(count, (d['seed'] for d in memory['records']))
    rows = []
    station = Station('left', ranging='temporal')
    player = station.player.player
    poses, tapes = player.poses, player.tapes
    full = np.arange(len(tapes))
    indices = {mode: np.array(memory[mode], dtype=int) if mode in memory else full for mode in MODES}
    budget = PlannerBudget(player)
    player.choose = budget.choose
    try:
        for ordinal, seed in enumerate(held):
            row = {}
            order = (*MODES[ordinal % len(MODES):], *MODES[:ordinal % len(MODES)])
            for condition in order:
                bank = indices[condition]
                player.poses, player.tapes = poses[bank], tapes[bank]
                row[condition] = _point(station, budget, seed, condition, bank)
            rows.append(row)
            print(ordinal+1, seed, {mode: (row[mode]['outcome']['legal_returns'], row[mode]['reward'],
                                           row[mode]['budget']['physical_rollouts']) for mode in MODES}, flush=True)
    finally:
        station.close()
    report = dict(schema='tt21-demonstration-1', eye='left', arena=SETTINGS, strong=False,
                  held_seed=SCHEDULE, held_seeds=held, rounds=rows, training=memory, teacher_cpu_s=teacher_cpu_s,
                  live_training_points=0, full_catalogue_len=len(full), catalogue_sha256=_hash(ATLAS),
                  scope='native strict opponent-service points from ordinary reset',
                  implementation={name: _hash(Path(__file__).with_name(name)) for name in (
                      'tt21_demonstration.py', 'tt21_demonstration_trial.py', 'tt19_model_rollout.py',
                      'tt20_camera_carry.py', 'tt20_natural.py')})
    report['metrics'], report['paired_vs_frozen'] = _metrics(rows)
    report['replayed_points'] = sum(verify(dict(report, points=[row[mode] for row in rows])) for mode in MODES)
    _write(output, report)
    print(report['metrics'], flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--output', type=Path, default=DEST)
    args = parser.parse_args()
    run(args.output, args.points)
