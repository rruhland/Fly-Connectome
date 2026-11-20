"""Confirm the frozen visual rebound checkpoint on new full and sparse boards."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from time import monotonic

import numpy as np

from run_breakout_visual_rebound import CHECKPOINT, TEST_ARMS, board_cases, run_arm, save, schedules


CONFIRM_SEEDS = {'ordinary': (185932133, 2016687407), 'lag': (743189081, 1190577317)}
AGENT_SEED = 912833511
BOOTSTRAP_SEED = 631037041
EARLY_PRACTICE_LIVES = 4
EARLY_TEST_LIVES = 12


def matched_effects(trained, comparison):
    pairs = tuple(zip(trained, comparison, strict=True))
    if any(any(a[key] != b[key] for key in ('case', 'layout', 'agent_seed', 'board')) for a, b in pairs):
        raise ValueError('confirmation requires matched cases, layouts and agent seeds')
    result = {}
    for board in ('all', 'full', 'sparse'):
        pairs_in_board = [(a, b) for a, b in pairs if board in ('all', a['board'])]
        scores = np.array([[a['post_own_bricks'], b['post_own_bricks']] for a, b in pairs_in_board])
        seconds = np.array([[a['elapsed_seconds'], b['elapsed_seconds']] for a, b in pairs_in_board])
        misses = np.array([[row['own_opportunities'] - row['own_returns'] for row in pair] for pair in pairs_in_board])
        count = len(scores)
        difference = scores[:, 0] - scores[:, 1]
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(count, size=(5000, count))
        rates = 60. * scores[draws].sum(axis=1) / seconds[draws].sum(axis=1)
        point_rates = 60. * scores.sum(axis=0) / seconds.sum(axis=0)
        result[board] = dict(
            pairs=count, ahead=int((difference > 0).sum()), tied=int((difference == 0).sum()),
            behind=int((difference < 0).sum()), missed_returns=misses.sum(axis=0).tolist(),
            bricks_per_life=float(difference.mean()),
            bricks_per_life_ci95=np.quantile(difference[draws].mean(axis=1), [.025, .975]).tolist(),
            bricks_per_minute=float(point_rates[0] - point_rates[1]),
            bricks_per_minute_ci95=np.quantile(rates[:, 0] - rates[:, 1], [.025, .975]).tolist())
    return result


def read(path):
    return json.loads(path.read_text())


def verify_reference(scout_spec, cases):
    used_boards, used_seeds = set(), set()
    for phase, rows in scout_spec['phase_cases'].items():
        if len(cases[phase]) != len(rows):
            raise ValueError('original scout case count changed')
        for case, row in zip(cases[phase], rows, strict=True):
            layout = tuple(map(tuple, row['layout']['cells']))
            if (case['layout'].cells != layout or asdict(case['episode']) != row['episode']
                    or case['agent_seed'] != row['agent_seed']):
                raise ValueError('original scout specification could not be reproduced')
            used_boards.add(layout)
            used_seeds.add(case['episode'].seed)
    return used_boards, used_seeds


def run(output, scout, checkpoint_path, *, ordinary_lives, lag_lives, lag_frames, workers):
    if ordinary_lives < 12 or ordinary_lives % 4 or lag_lives < 0 or lag_lives % 4 or workers < 1 or lag_frames < 1:
        raise ValueError('use balanced multiples of four lives, positive lag/worker count and at least 12 ordinary lives')
    began = monotonic()
    scout_spec = read(scout / 'spec.json')
    reference = read(scout / 'practice-summary.json')['final_state']
    checkpoint_data = checkpoint_path.read_bytes()
    digest = hashlib.sha256(checkpoint_data).hexdigest()
    if scout_spec['checkpoint_sha256'] != digest or not scout_spec['public_only_transition']:
        raise ValueError('visual rebound confirmation requires the original checkpoint and public pilot')
    checkpoint = json.loads(checkpoint_data)
    duration = scout_spec['duration']
    pilot_cases = schedules(len(scout_spec['phase_cases']['practice']), len(scout_spec['phase_cases']['test']))
    seen, used_seeds = verify_reference(scout_spec, pilot_cases)
    requested = [('ordinary', ordinary_lives)]
    if lag_lives:
        requested.append(('lag', lag_lives))
    cases = board_cases(requested, CONFIRM_SEEDS, agent_seed=AGENT_SEED, seen=seen)
    new_seeds = [case['episode'].seed for rows in cases.values() for case in rows]
    if used_seeds.intersection(new_seeds) or len(new_seeds) != len(set(new_seeds)):
        raise ValueError('confirmation requires seeds disjoint from the scout')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'spec.json', dict(
        scout_checkpoint=reference, checkpoint_sha256=digest, duration=duration,
        phase_cases={phase: [dict(case, episode=asdict(case['episode']), layout=asdict(case['layout']))
                             for case in rows] for phase, rows in cases.items()},
        lag_frames=lag_frames, test_arms=TEST_ARMS, primary='ordinary visual_trained vs visual_empty',
        secondary=['full and sparse subsets', 'forecast center', 'early snapshot', 'lag'],
        early_practice_lives=EARLY_PRACTICE_LIVES, early_test_lives=EARLY_TEST_LIVES,
        cpu_scope='sum of process CPU time for played arms and early practice replay'))
    early = run_arm(output, checkpoint, 'early-practice', 'visual_live',
                    pilot_cases['practice'][:EARLY_PRACTICE_LIVES], duration)
    save(output / 'early-practice-summary.json', early)
    result = {'early_practice': early}
    cpu_seconds = early['cpu_seconds']
    limit = min(workers, len(TEST_ARMS))
    with ProcessPoolExecutor(max_workers=limit, mp_context=get_context('spawn')) as pool:
        for phase, rows in cases.items():
            delay = lag_frames if phase == 'lag' else 0
            jobs = {arm: pool.submit(run_arm, output, checkpoint, phase, arm, rows, duration, reference, delay)
                    for arm in TEST_ARMS}
            ordinary = phase == 'ordinary'
            if ordinary:
                early_job = pool.submit(run_arm, output, checkpoint, 'early-test', 'visual_trained',
                                        rows[:EARLY_TEST_LIVES], duration, early['final_state'])
            reports = {arm: job.result() for arm, job in jobs.items()}
            cpu_seconds += sum(report['cpu_seconds'] for report in reports.values())
            receipts = {arm: read(output / f'{phase}-{arm}.json') for arm in TEST_ARMS}
            effects = {arm: matched_effects(receipts['visual_trained'], receipts[arm]) for arm in TEST_ARMS[1:]}
            result[phase] = dict(arms=reports, matched_effects=effects)
            save(output / f'{phase}-summary.json', result[phase])
            if ordinary:
                early_report = early_job.result()
                cpu_seconds += early_report['cpu_seconds']
                early_rows = read(output / 'early-test-visual_trained.json')
                result['early_test'] = dict(arm=early_report, matched_effects={
                    arm: matched_effects(early_rows, receipts[arm][:EARLY_TEST_LIVES]) for arm in TEST_ARMS})
            result.update(cpu_seconds=cpu_seconds, wall_seconds=monotonic() - began)
            save(output / 'summary.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('scout', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--ordinary-lives', type=int, default=32)
    parser.add_argument('--lag-lives', type=int, default=16)
    parser.add_argument('--lag-frames', type=int, default=6)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.scout, args.checkpoint, ordinary_lives=args.ordinary_lives,
        lag_lives=args.lag_lives, lag_frames=args.lag_frames, workers=args.workers)
