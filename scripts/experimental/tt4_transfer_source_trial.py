"""Compare first-person mini donors on new matched receiving-table points."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np

from .tt4_transfer_trial import ROOT, _course, _hash, _metrics, _read, _seeds, _summary, _write, load_sources


CONDITIONS = ('wide_first_person_mini', 'first_person_mini', 'pong_only')
PAIRS = tuple((condition, baseline) for index, condition in enumerate(CONDITIONS)
              for baseline in CONDITIONS[index + 1:])


def compare_donors(courses, seed):
    draws = np.random.default_rng(seed).integers(len(courses), size=(8192, len(courses)))
    comparisons = {}
    for condition, baseline in PAIRS:
        comparison = {}
        for phase, stop in (('first', 1), ('first_eight', 8), ('all', None)):
            paired = [(_metrics(course['runs'][condition]['points'][:stop]),
                       _metrics(course['runs'][baseline]['points'][:stop])) for course in courses]
            comparison[phase] = {}
            for field in ('with_return', 'player_returns', 'reply_after_return', 'sustained', 'points_for'):
                delta = np.array([left[field] - right[field] for left, right in paired])
                comparison[phase][field] = dict(by_installation=delta.tolist(), total=int(delta.sum()),
                                                mean=float(delta.mean()), bootstrap95=np.quantile(
                                                    delta[draws].mean(axis=1), [.025, .975]).tolist())
        comparisons[f'{condition}_minus_{baseline}'] = comparison
    return comparisons


def qualify(reference, regular_mini, wide_mini, *, seed, installations, points, workers):
    old = _read(reference)
    geometry = old['geometry']
    native, states, sources = load_sources(regular_mini)
    _, alternative, alt_hashes = load_sources(wide_mini)
    states['wide_first_person_mini'] = alternative['first_person_mini']
    sources['wide_first_person_mini'] = alt_hashes['first_person_mini']
    number_probes = len(old['schedules'][0]['calibration'])
    schedules = [_seeds(seed, index, number_probes, points) for index in range(installations)]
    prior_points = {point for case in old['schedules'] for key in ('calibration', 'tests') for point in case[key]}
    if any(prior_points.intersection(case['calibration'] + case['tests']) for case in schedules):
        raise ValueError('the independent comparison overlaps the original receiving schedule')
    view = 'first_person'
    jobs = [(schedule, view, CONDITIONS, native, states, geometry) for schedule in schedules]
    began = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        courses = []
        for row in pool.map(_course, jobs):
            courses.append(row)
            metrics = {name: _metrics(values['points']) for name, values in row['runs'].items()}
            print(row['schedule']['index'], metrics, flush=True)
    summary = _summary(courses, CONDITIONS, points, seed)
    summary['comparisons'] = compare_donors(courses, seed)
    return dict(schema='tt4-paired-mini-donor-comparison-v1', seed=seed, schedules=schedules,
                conditions={view: CONDITIONS}, inputs_sha256=sources,
                receiving_protocol_sha256=_hash(reference), geometry=geometry,
                courses={view: courses}, summaries={view: summary},
                wall_seconds=time.perf_counter() - began)


def main():
    artifacts = ROOT / 'artifacts' / 'tt4'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path, help='sealed receiving protocol and excluded schedule')
    parser.add_argument('wide_mini', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--regular-mini', type=Path, default=artifacts / 'mini-first-person-player.json.gz')
    parser.add_argument('--seed', type=int, default=73184025)
    parser.add_argument('--installations', type=int, default=3)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if min(args.installations, args.points, args.workers) < 1:
        parser.error('at least one installation, point and worker is required')
    options = vars(args).copy()
    options.pop('output')
    result = qualify(**options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, result)


if __name__ == '__main__':
    main()
