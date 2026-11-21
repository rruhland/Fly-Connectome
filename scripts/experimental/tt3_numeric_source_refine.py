"""Refine portable numeric hypotheses using matched native Pong payouts only."""

import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from tt3_numeric import NumericMemory
from tt3_pong import _play_pong, cases_from_seeds, summarize_pong
from tt3_pong_probe import ShorterDwell
from tt3_visual import RoleModel


SEARCH_SEED = 641273907
SCREEN_OFFSET = 237_000_000
VALIDATION_OFFSET = 273_000_000
POOL_SIZE = 17_000_000
OPPONENT_SPEED = .35
MAX_TICKS = 1440
SOURCE_POINTS = (2, 4)
KEEP = 4


def _read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def candidates(generations):
    if len(generations) != 5:
        raise ValueError('provide all five successive numeric source generations')
    offered = {}
    centers = []

    def add(coefficients, label):
        vector = tuple(float(value) for value in coefficients)
        offered.setdefault(vector, []).append(label)

    for index, generation in enumerate(generations):
        coefficients = np.asarray(generation['state']['coefficients'])
        payout = np.asarray(generation['public_points'])
        order = np.argsort(-payout, kind='stable')
        for elite in order[:2]:
            add(coefficients[elite], f'cross_entropy_generation_{index + 1}_rank_{list(order).index(elite) + 1}')
        upper = order[:max(2, len(order) // 3)]
        weights = np.exp((payout[upper] - payout[upper[0]]) / 2.)
        centers.append(np.average(coefficients[upper], weights=weights, axis=0))

    center = np.average(centers, weights=np.arange(1, len(centers) + 1), axis=0)
    add(center, 'score_weighted_centroid')
    neutral = center.copy()
    neutral[0] = 0.
    add(neutral, 'zero_offset_centroid')
    for direction in (-1., 1.):
        signed = center.copy()
        signed[0] = np.clip(signed[0] + direction * .044, -.16, .16)
        add(signed, 'signed_offset_minus' if direction < 0 else 'signed_offset_plus')
    return [dict(coefficients=list(vector), origin=labels) for vector, labels in offered.items()]


def _cases(random, count, offset):
    return cases_from_seeds(random.choice(POOL_SIZE, count, replace=False) + offset)


def _evaluate(knowledge, choices, cases, points, roles):
    memory = NumericMemory([row['coefficients'] for row in knowledge])
    schedule = [case for case in cases for index in choices]
    options = [index for case in cases for index in choices]
    rows = _play_pong(schedule, roles, memory, options, MAX_TICKS, OPPONENT_SPEED, 48,
                      points_per_match=points, update_memory=False, motor_factory=ShorterDwell)
    results = []
    for index in choices:
        subset = [row for row in rows if row['relation'] == index]
        result = dict(candidate=int(index), payout=sum(row['score'] for row in subset),
                      summary=summarize_pong(subset), rows=subset)
        results.append(result)
    return results


def _order(results):
    return sorted(results, key=lambda row: (-row['payout'], row['candidate']))


def _freeze(knowledge, shortlist, screen):
    first = {row['candidate']: row for row in screen}
    combined = []
    for row in shortlist:
        index = row['candidate']
        total = row['payout'] + first[index]['payout']
        count = len(row['rows']) + len(first[index]['rows'])
        combined.append(dict(candidate=index, payout=total, count=count, utility=total / count))
    order = sorted(combined, key=lambda row: (-row['utility'], row['candidate']))
    best = order[0]
    # A deterministic, materially distinct opening preference can be restored by the ordinary
    # memory class. Equal-score hypotheses do not enter the bank by unexplained RNG tie-break.
    retained = [best] + [row for row in order[1:] if row['utility'] < best['utility']]
    state = NumericMemory([knowledge[row['candidate']]['coefficients'] for row in retained],
                          [row['payout'] for row in retained], [row['count'] for row in retained]).state()
    return state, order, retained


def run(args):
    torch.set_num_threads(1)
    source_roles = _read(args.roles)
    roles = RoleModel(**source_roles.get('role_model', source_roles))
    knowledge = candidates(_read(args.practice))
    random = np.random.default_rng(args.seed)
    screen_cases = _cases(random, args.screen, SCREEN_OFFSET)
    validation_cases = _cases(random, args.validation, VALIDATION_OFFSET)
    print('Native score screen:', len(knowledge), 'hypotheses on', len(screen_cases), 'matches', flush=True)
    screen = _evaluate(knowledge, list(range(len(knowledge))), screen_cases, SOURCE_POINTS[0], roles)
    shortlist = [row['candidate'] for row in _order(screen)[:KEEP]]
    print('Native public scores:', [(row['candidate'], row['payout']) for row in _order(screen)], flush=True)
    validation = _evaluate(knowledge, shortlist, validation_cases, SOURCE_POINTS[1], roles)
    state, combined, retained = _freeze(knowledge, validation, screen)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / 'numeric-state.json'
    checkpoint.write_text(json.dumps(state, indent=2) + '\n')
    source_files = [Path(__file__).with_name(filename) for filename in (
        'tt3_numeric_source_refine.py', 'tt3_numeric.py', 'tt3_pong.py', 'tt3_pong_probe.py', 'tt3_visual.py')]
    receipt = dict(seed=args.seed, source_sha256={path.name: sha256(path.read_bytes()).hexdigest()
        for path in [*source_files, args.roles, args.practice]}, role_model=asdict(roles), motor='ShorterDwell',
        max_ticks=MAX_TICKS, opponent_speed=OPPONENT_SPEED, candidates=knowledge,
        screen_cases=[asdict(case) for case in screen_cases], screen=screen, points_per_match=SOURCE_POINTS,
        validation_cases=[asdict(case) for case in validation_cases], validation=validation,
        combined_scores=combined, retained=retained, portable_memory=state,
        numeric_state_sha256=sha256(checkpoint.read_bytes()).hexdigest())
    with gzip.open(args.out / 'numeric-source-refinement.json.gz', 'wt') as handle:
        json.dump(receipt, handle, allow_nan=False, separators=(',', ':'))
    print('Frozen:', json.dumps(dict(checkpoint=str(checkpoint), retained=retained, state=state)), flush=True)
    for result in _order(validation):
        print('Native validation:', result['candidate'], result['payout'], result['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roles', type=Path)
    parser.add_argument('practice', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--seed', type=int, default=SEARCH_SEED)
    parser.add_argument('--screen', type=int, default=12)
    parser.add_argument('--validation', type=int, default=20)
    run(parser.parse_args())
