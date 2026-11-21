"""Compare early table performance with matched body learning and different Pong priors."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np

from .tt3_arena import HZ, heldout_cases
from .tt3_body_retention import ResponseConditionedMemory, observing_factory
from .tt3_learning import ReturnMemory
from .tt3_learning_stroke import stroke_learner_factory
from .tt3_learning_trial import play_course, summarize
from .tt3_pong import state_sha256
from .tt3_visual import RoleModel


CONDITIONS = ('trained', 'carried_roles_reset_values', 'untrained')
REGIMES = ('zero_target_practice', 'common_calibrated_stroke')
MAX_POINT_FRAMES = 300


def _time_to_first_return(course):
    elapsed = 0
    for row in course['points']:
        first = next((event for event in row['events']
                      if event['kind'] == 'legal_return' and event['side'] == 1), None)
        if first is not None:
            return dict(frames=elapsed + first['frame'], point=row['point'], censored=False,
                        capped_frames=elapsed + first['frame'])
        elapsed += row['frames']
    return dict(frames=None, point=None, censored=True, capped_frames=MAX_POINT_FRAMES * len(course['points']))


def _play(job):
    regime, name, index, cases, seed, source, offsets, forecast, integrated = job
    roles = RoleModel.untrained() if name == 'untrained' else RoleModel(**source['role_model'])
    values = source['return_memory'] if name == 'trained' else ReturnMemory().state()
    memory = ResponseConditionedMemory(values, observed_rank=1)
    memory.prepare_stroke(offsets)
    factory = observing_factory(stroke_learner_factory(forecast=forecast, integrated_motor=integrated))
    course = play_course(cases, roles, memory, seed=seed, learner_factory=factory)
    timing = _time_to_first_return(course)
    return dict(regime=regime, condition=name, index=index, seed=seed,
                first_return_timing=timing, **course)


def _population_summary(courses, points=None):
    totals = summarize(courses, points)
    selected = [row for course in courses for row in course['points'][:points]]
    totals['earned_after_return'] = sum(row['points_for'] > 0 and row['legal_returns'] > 0 for row in selected)
    return totals


def _paired(challenger, reference, *, samples):
    pairs = list(zip(challenger, reference, strict=True))
    if any(left['index'] != right['index'] or
           [row['case'] for row in left['points']] != [row['case'] for row in right['points']]
           for left, right in pairs):
        raise ValueError('paired courses must use identical schedules')
    metrics = {}
    for points in (4, 8, len(challenger[0]['points'])):
        for field in ('any_legal_return', 'player_returns', 'wins', 'earned_after_return'):
            metrics[f'first_{points}_{field}'] = np.array([
                _population_summary([left], points)[field] - _population_summary([right], points)[field]
                for left, right in pairs
            ])
    metrics['capped_frames_saved'] = np.array([right['first_return_timing']['capped_frames'] -
                                                left['first_return_timing']['capped_frames'] for left, right in pairs])
    result = {name: dict(mean=float(delta.mean()), by_course=delta.tolist(),
                         match_bootstrap_interval95=np.quantile(delta[samples].mean(axis=1), [.025, .975]).tolist())
              for name, delta in metrics.items()}
    observed = [(pair[0]['index'], pair[1]['first_return_timing']['frames'] - pair[0]['first_return_timing']['frames'])
                for pair in pairs if all(not course['first_return_timing']['censored'] for course in pair)]
    result['observed_frames_saved'] = dict(by_course=observed,
                                          mean=float(np.mean([difference for _, difference in observed])) if observed else None)
    return result


def qualification(source, stroke_receipt, *, seed, courses, points, workers, progress=None):
    fresh = [heldout_cases(f'{seed}/body-conditioned-entry/{index}', points) for index in range(courses)]
    trained_cases = {row['case']['seed'] for generation in stroke_receipt['training']
                     for candidate in generation['runs'] for row in candidate['course']['points']}
    trained_cases.update(row['case']['seed'] for reference in stroke_receipt['qualification']['results'].values()
                         for row in reference['course']['points'])
    if any(case.seed in trained_cases for course in fresh for case in course):
        raise ValueError('the new table schedule overlaps the stroke receipt')
    learner_seeds = [int(np.random.SeedSequence([seed, index]).generate_state(1)[0]) for index in range(courses)]
    offsets = {REGIMES[0]: [0., 0., 0.], REGIMES[1]: stroke_receipt['stroke']}
    jobs = [(regime, name, index, cases, learner_seeds[index], source, offsets[regime],
             stroke_receipt['forecast'], stroke_receipt['integrated_motor'])
            for index, cases in enumerate(fresh) for regime in REGIMES for name in CONDITIONS]
    results = {regime: {name: [] for name in CONDITIONS} for regime in REGIMES}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(_play, jobs):
            results[row['regime']][row['condition']].append(row)
            if progress:
                progress(row)
    draws = np.random.default_rng(250199583).integers(courses, size=(8192, courses))
    summaries = {}
    for regime, by_condition in results.items():
        summaries[regime] = {
            'arms': {name: {
                'first_four': _population_summary(values, 4), 'first_eight': _population_summary(values, 8),
                'full': _population_summary(values),
                'first_return_timing': {
                    'courses_reaching_return': sum(not row['first_return_timing']['censored'] for row in values),
                    'mean_capped_frames': float(np.mean([row['first_return_timing']['capped_frames'] for row in values])),
                    'by_course': [row['first_return_timing'] for row in values],
                },
                'elapsed_wall_seconds': sum(row['wall_seconds'] for row in values),
            } for name, values in by_condition.items()},
            'trained_minus': {name: _paired(by_condition['trained'], values, samples=draws)
                              for name, values in by_condition.items() if name != 'trained'},
        }
    calibration = [candidate['course'] for generation in stroke_receipt['training'] for candidate in generation['runs']]
    selected = [generation['runs'][generation['selected']]['course'] for generation in stroke_receipt['training']]
    return dict(seed=seed, courses=courses, points_per_course=points, camera_hz=HZ,
                time_limit_frames_per_point=MAX_POINT_FRAMES, source_state_sha256=state_sha256(source),
                stroke_receipt_sha256=state_sha256(stroke_receipt), offsets=offsets,
                common_stroke_calibration=dict(candidate_courses=len(calibration),
                                               physical_points=sum(len(course['points']) for course in calibration),
                                               selected_candidate_points=sum(len(course['points']) for course in selected),
                                               physical_frames=sum(row['frames'] for course in calibration for row in course['points']),
                                               candidate_wall_seconds=sum(course['wall_seconds'] for course in calibration),
                                               role_and_preference_estimates_carried=False),
                schedules=[[asdict(case) for case in course] for course in fresh],
                results=results, summaries=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('stroke_receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=1826639275)
    parser.add_argument('--courses', type=int, default=6)
    parser.add_argument('--points', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.courses < 1 or args.points < 8 or args.workers < 1:
        parser.error('at least one course of eight points and one worker is required')
    source = json.loads(args.source.read_text())
    opener = gzip.open if args.stroke_receipt.suffix == '.gz' else open
    with opener(args.stroke_receipt, 'rt') as handle:
        strokes = json.load(handle)

    def progress(row):
        print(json.dumps(dict(regime=row['regime'], condition=row['condition'], course=row['index'],
                              first_eight=summarize([row], 8), first_return=row['first_return_timing'])), flush=True)

    result = qualification(source, strokes, seed=args.seed, courses=args.courses, points=args.points,
                           workers=args.workers, progress=progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n').encode()
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_bytes(gzip.compress(body, mtime=0))
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
