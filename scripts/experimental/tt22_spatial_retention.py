"""Apply SO-101's updated numeric preferences back in native Pong."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .tt3_numeric import NumericMemory
from .tt3_pong import cases_from_seeds, evaluate_pong, state_sha256, summarize_pong
from .tt3_pong_probe import ShorterDwell
from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_spatial_trial import OUTPUT, SOURCE_REFINEMENT


NATIVE_RETURN = 1558469201
MATCHES = 6
POINTS_PER_MATCH = 4


def returned_state(checkpoint, *, after):
    totals = np.asarray(checkpoint['prior_total'], dtype=float)
    counts = np.asarray(checkpoint['prior_count'], dtype=int)
    if after:
        totals += np.asarray(checkpoint['totals'], dtype=float)
        counts += np.asarray(checkpoint['counts'], dtype=int)
    return NumericMemory(checkpoint['coefficients'], totals, counts)


def run(path, output):
    torch.set_num_threads(1)
    result = _read(path)
    target = result['scores']
    if target['pong']['first_returns'] <= target['scratch']['first_returns']:
        raise ValueError('the prespecified target condition for a native return check was not met')
    qualification = _read(SOURCE_REFINEMENT)
    roles = RoleModel(**qualification['role_model'])
    seeds = np.random.default_rng(NATIVE_RETURN).choice(40_000_000, MATCHES, replace=False)+610_000_000
    used = {row['serve'] for key in ('screen_cases', 'validation_cases') for row in qualification[key]}
    if used.intersection(seeds):
        raise ValueError('native return repeats a source-training start')
    cases = cases_from_seeds(seeds)
    groups = {}
    for phase in ('before', 'after'):
        policy = returned_state(result['final']['pong'], after=phase == 'after')
        state = policy.state()
        rows = evaluate_pong(cases, roles, policy, max_ticks=qualification['max_ticks'],
                             opponent_speed=qualification['opponent_speed'], points_per_match=POINTS_PER_MATCH,
                             motor_factory=ShorterDwell)
        groups[phase] = dict(memory=state, memory_sha256=state_sha256(state),
                             summary=summarize_pong(rows), points=rows)
        print(phase, groups[phase]['summary'], flush=True)
    a, b = groups['before']['points'], groups['after']['points']
    pairs = list(zip(a, b, strict=True))
    if any((left['case'], left['point']) != (right['case'], right['point']) for left, right in pairs):
        raise ValueError('native return changed the paired start schedule')
    data = dict(schema='tt22-spatial-return-pong-1', target_checkpoint_sha256=_hash(path),
                source_receipt_sha256=_hash(SOURCE_REFINEMENT), seed=NATIVE_RETURN,
                matched= [asdict(case) for case in cases], runs=groups,
                matching_actions=sum(left['action_sha256'] == right['action_sha256'] for left, right in pairs),
                matching_outcomes=sum((left['score'], left['contacts']) == (right['score'], right['contacts'])
                                      for left, right in pairs))
    _write(output, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    args = parser.parse_args()
    prefix = 'spatial-direct' if args.direct else 'spatial-score'
    run(args.folder/f'{prefix}.json.gz', args.folder/f'{prefix}-pong-return.json.gz')
