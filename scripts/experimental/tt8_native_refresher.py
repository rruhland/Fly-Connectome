"""Revalidate carried numeric choices by playing fresh points before testing retention."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .tt3_numeric_retention import _hashes, _native
from .tt3_numeric_transfer import PortableMemory
from .tt3_pong import assess_retention, cases_from_seeds, practice_pong, state_sha256
from .tt3_visual import RoleModel
from .tt4_transfer_trial import ROOT, _hash, _read, _write


def _options(memory):
    return memory.scores / np.maximum(1, memory.counts)


def _leader(memory):
    return int(np.argmax(_options(memory)))


def _revalidate(rows, nominee, incumbent, *, seed):
    # Cluster repeated points from one starting scene. Only the score stream
    # enters this decision. Contacts and return statistics stay in the receipt.
    paired = {}
    for point in rows:
        key = (point['match']['serve'], point['point'])
        scores = paired.setdefault(key, {})
        if point['relation'] in scores:
            raise ValueError('duplicate policy in a revalidation pair')
        scores[point['relation']] = point['score']
    groups = {}
    for (start, _), scores in paired.items():
        groups.setdefault(start, []).append(scores[nominee] - scores[incumbent])
    difference = np.array([np.mean(scores) for scores in groups.values()])
    clusters = np.random.default_rng(seed).integers(len(difference), size=(8192, len(difference)))
    lower = float(np.quantile(difference[clusters].mean(axis=1), .10))
    return dict(starts=len(difference), mean=float(difference.mean()), lower90=lower, promoted=lower > 0.)


def _practice(cases, memory, roles, options, max_ticks, speed, points):
    return practice_pong([case for case in cases for _ in options], roles, memory,
                         options=list(options) * len(cases), max_ticks=max_ticks,
                         opponent_speed=speed, points_per_match=points)


def _checkpoint(source, memory):
    state = memory.state()
    return source | {'numeric': [state['coefficients'], state['totals'], state['counts']]}


def run(source_path, reference_path, output, checkpoint, *, seed, starts, validation_starts,
        practice_checkpoint=None):
    reference = _read(reference_path)
    source = _read(source_path)
    if any(reference['implementation_sha256'].get(name) != digest for name, digest in _hashes().items()):
        raise ValueError('the reference uses a different native Pong implementation')
    features, totals, counts = source['numeric']
    carried = PortableMemory(features, totals, counts)
    roles = RoleModel(**source['roles'])
    schedule = reference['schedule']
    before = reference['runs']['after']['points']
    held = {p['case']['serve'] for p in before}
    candidates = range(len(features))
    rng = np.random.default_rng(seed)
    seeds = 1_400_000_000 + rng.choice(200_000_000, size=starts + validation_starts, replace=False)
    cases = cases_from_seeds(seeds)
    ticks, speed = schedule['max_ticks'], schedule['opponent_speed']
    incumbent = _leader(carried)
    practiced = PortableMemory(features)
    discovery = _practice(cases[:starts], practiced, roles, candidates, ticks, speed, 2)
    leaders = np.flatnonzero(np.isclose(_options(practiced), _options(practiced).max()))
    nominee = incumbent if incumbent in leaders else int(rng.choice(leaders))
    print('Practice scores', practiced.scores.tolist(), 'incumbent', incumbent, 'nominee', nominee, flush=True)
    checked = PortableMemory(features)
    validation = []
    decision = dict(promoted=False)
    if nominee != incumbent:
        validation = _practice(cases[starts:], checked, roles, [incumbent, nominee], ticks, speed, 3)
        decision = _revalidate(validation, nominee, incumbent, seed=seed)
    discovery_serves = {p['case']['serve'] for p in discovery}
    validation_serves = {p['case']['serve'] for p in validation}
    if (discovery_serves | validation_serves) & held or discovery_serves & validation_serves:
        raise ValueError('native practice and evaluation starts overlap')
    print('Independent score revalidation', decision, flush=True)

    # Keep the paired discovery totals when promotion is justified. The
    # narrower validation set serves as a gate and cannot downsample the bank.
    deployed = practiced if decision['promoted'] else carried
    accumulated = PortableMemory(features, carried.scores + practiced.scores + checked.scores,
                                  carried.counts + practiced.counts + checked.counts)
    variants = {'accumulated': accumulated, 'practice_only': practiced, 'confidence_gated': deployed}
    report = dict(schema='tt8-native-refresher-v1', seed=seed, source_sha256=_hash(source_path),
                  reference_sha256=_hash(reference_path), evaluation_schedule_sha256=state_sha256(schedule),
                  incumbent=incumbent, nominee=nominee, decision=decision, practice=discovery,
                  independent_validation=validation, practice_state=practiced.state(),
                  validation_state=checked.state(), states={k: v.state() for k, v in variants.items()},
                  evaluation={})
    _write(checkpoint, _checkpoint(source, deployed))
    if practice_checkpoint is not None:
        _write(practice_checkpoint, _checkpoint(source, practiced))
    _write(output, report)
    played = {}
    for name, memory in variants.items():
        choice = _leader(memory)
        # With one leader all native choices are deterministic given the case.
        # A tied bank gets its own run because exploration RNG may break a tie.
        tied = int(np.isclose(_options(memory), _options(memory).max()).sum()) > 1
        key = state_sha256(memory.state()) if tied else choice
        if key not in played:
            played[key] = _native(schedule, dict(memory=memory.state(), roles=source['roles']))
        result = played[key]
        differences = sum(a['action_sha256'] != b['action_sha256'] for a, b in zip(before, result['points'], strict=True))
        report['evaluation'][name] = dict(changed_action_tapes=differences, choice=choice,
                                          retention=assess_retention(before, result['points']), run=result)
        _write(output, report)
        print(name, choice, differences, result['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/tt8/after-three-games.json.gz')
    parser.add_argument('--reference', type=Path, default=ROOT / 'artifacts/tt6/pong-retention.json.gz')
    parser.add_argument('--seed', type=int, default=70351829)
    parser.add_argument('--starts', type=int, default=16)
    parser.add_argument('--validation-starts', type=int, default=16)
    parser.add_argument('--practice-checkpoint', type=Path, help='save the discovery-only control for paired play')
    args = parser.parse_args()
    if min(args.starts, args.validation_starts) < 1:
        parser.error('practice and confirmation require positive start counts')
    torch.set_num_threads(1)
    run(args.source, args.reference, args.output, args.checkpoint, seed=args.seed,
        starts=args.starts, validation_starts=args.validation_starts, practice_checkpoint=args.practice_checkpoint)
