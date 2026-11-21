"""Check stored puck scoring and aggregate paired outcomes across the four arms."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt16_payload import (BLOCK_FRAMES, CASES, EARLY_BLOCKS, FIXTURES, GOAL_BLOCKS, GOAL_RADIUS, HOLD_BLOCKS,
                           STOP_SPEED, TARGETS, summarize)


def check_run(run):
    for name in CASES:
        result = run['cases'][name]
        rows = result['blocks']
        if len(rows) != TARGETS * GOAL_BLOCKS or result['summary'] != summarize(rows):
            raise ValueError('length or summary differs from the block tape')
        puck, velocity = np.zeros((2, 2))
        coupling, retention = np.asarray(run['coupling']), np.asarray(run['retention'])
        for tick, row in enumerate(rows):
            target, offset = divmod(tick, GOAL_BLOCKS)
            if offset == 0:
                streak, scored = 0, False
            velocity = retention @ velocity + coupling @ row['tool']
            puck += velocity
            distance = float(np.linalg.norm(puck - run['goals'][target]))
            speed = float(np.linalg.norm(velocity))
            streak = streak + 1 if distance <= GOAL_RADIUS and speed <= STOP_SPEED else 0
            reward = int(streak >= HOLD_BLOCKS and not scored)
            scored |= bool(reward)
            if puck.tolist() != row['puck'] or (distance, speed, reward) != (row['distance'], row['speed'], row['reward']):
                raise ValueError('inertial trajectory or reward differs from the physical tool tape')


def metrics(runs):
    budget = BLOCK_FRAMES * GOAL_BLOCKS
    result = {}
    for name in CASES:
        rows = [r['cases'][name]['summary'] for r in runs]
        first = [s['first_goal'] for s in rows]
        map_error = []
        for run in runs:
            coupling = np.asarray(run['coupling'])
            estimate = np.asarray(run['cases'][name]['object'])[:, 2:]
            map_error.append(np.linalg.norm(estimate - coupling) / np.linalg.norm(coupling))
        result[name] = dict(
            first_by_early_cutoff=sum(t is not None and t <= EARLY_BLOCKS * BLOCK_FRAMES for t in first),
            first_acquired=sum(t is not None for t in first),
            first_acquisition_mean_capped_frames=float(np.mean([budget if t is None else t for t in first])),
            goal_points=sum(s['points'] for s in rows), goal_opportunities=len(rows) * TARGETS,
            early_mean_error_m=float(np.mean([s['early_error'] for s in rows])),
            last_goal_mean_error_m=float(np.mean([s['late_error'] for s in rows])),
            coupling_relative_error=float(np.mean(map_error)),
            total_rate_l1_frames=sum(s['rate_l1_frames'] for s in rows),
            scene_edge_blocks=sum(s['scene_edge_blocks'] for s in rows),
        )
    paired = {}
    for name in ('wrong', 'cold'):
        delta = np.array([r['cases']['carried']['summary']['early_error'] - r['cases'][name]['summary']['early_error']
                          for r in runs])
        first = np.array([
            [r['cases'][mode]['summary']['first_goal'] or budget + 1 for mode in ('carried', name)]
            for r in runs
        ])
        paired[name] = dict(early_error_delta_m=float(delta.mean()),
                            carried_lower=int((delta < 0).sum()), equal=int((delta == 0).sum()),
                            carried_higher=int((delta > 0).sum()),
                            first_goal_carried_sooner=int((first[:, 0] < first[:, 1]).sum()),
                            first_goal_tied=int((first[:, 0] == first[:, 1]).sum()),
                            first_goal_carried_later=int((first[:, 0] > first[:, 1]).sum()))
    return dict(courses=len(runs), performance=result, paired=paired)


def report(folder):
    receipts, combined = {}, []
    implementations = set()
    for fixture in FIXTURES:
        path = folder / f'payload-{fixture}.json.gz'
        evidence = _read(path)
        if evidence['fixture'] != fixture or not evidence['replayed']:
            raise ValueError('unmatched fixture or missing physical replay')
        implementations.add(evidence['implementation_sha256'])
        for run in evidence['runs']:
            check_run(run)
        combined.extend(evidence['runs'])
        receipts[fixture] = dict(file=path.name, sha256=_hash(path), **metrics(evidence['runs']))
    if implementations != {_hash(Path(__file__).with_name('tt16_payload.py'))}:
        raise ValueError('held-out runs use a different simulation or policy')
    if len({r['world_seed'] for r in combined}) != len(combined):
        raise ValueError('duplicate puck world seeds across fixtures')
    summary = dict(schema='tt16-payload-summary-v1', implementation_sha256=implementations.pop(),
                   fixtures=receipts, overall=metrics(combined), replayed_courses=len(combined) * len(CASES),
                   replayed_blocks=len(combined) * len(CASES) * TARGETS * GOAL_BLOCKS)
    _write(folder / 'payload-summary.json.gz', summary)
    print(summary['overall'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    report(parser.parse_args().folder)
