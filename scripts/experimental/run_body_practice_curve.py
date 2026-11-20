"""Measure how much source practice helps the first reach with new arm motors."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from body_effect_learner import BodyLearner
from body_visual_utility import (
    ACTION_DIM, FAMILIES, TARGET_BODY, ChoiceArena, UtilityLearner, code_signatures,
    run_course, run_trial, sample_layout, sample_rule, summarize,
)
from run_position_arm_transfer import position_arm


PRACTICE_LENGTHS = (0, 2, 4, 8, 12, 24)


def first_reaches(seed, family):
    rule = sample_rule(seed, family)
    practice, full_prior = run_course(seed, 'carriage', rule, source=True)
    layout_seed, tie_seed, motor_seed = np.random.SeedSequence([seed, 7]).spawn(3)
    layout = sample_layout(np.random.default_rng(layout_seed))
    tie = int(np.random.default_rng(tie_seed).integers(2))
    utility = UtilityLearner()
    source_commands = 0
    measurements = {}
    for count in range(max(PRACTICE_LENGTHS) + 1):
        if count:
            trial = practice[count - 1]
            source_commands += trial['commands']
            feedback = trial['feedback']
            if feedback is not None:
                utility.observe(np.array(feedback['difference']), feedback['label'])
        if count in PRACTICE_LENGTHS:
            arena = ChoiceArena(TARGET_BODY, layout, rule, False, position_arm)
            motor = BodyLearner(action_dim=ACTION_DIM, seed=motor_seed)
            target = run_trial(arena, motor, UtilityLearner(utility.checkpoint()), tie)
            measurements[count] = {'target': target, 'source_commands': source_commands}
    if full_prior != utility.checkpoint():
        raise ValueError('source prefix replay diverged from the live carriage learner')
    return {'seed': seed, 'family': family, 'measurements': measurements}


def run(first_seed, count):
    rows = [first_reaches(first_seed + i, FAMILIES[i % len(FAMILIES)]) for i in range(count)]
    summary = {}
    for length in PRACTICE_LENGTHS:
        records = [row['measurements'][length] for row in rows]
        summary[length] = {
            **summarize([r['target'] for r in records]),
            'mean_source_commands': sum(r['source_commands'] for r in records) / len(records),
            'first_hold_by_family': {
                family: sum(row['measurements'][length]['target']['first_physical_hold_rewarded']
                            for row in rows if row['family'] == family) for family in FAMILIES
            },
        }
    hashes = code_signatures()
    hashes[Path(__file__).name] = sha256(Path(__file__).read_bytes()).hexdigest()
    return {'schema': 'carriage-practice-first-arm-reach-v1', 'first_seed': first_seed,
            'source_sha256': hashes, 'summary': summary, 'streams': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--first-seed', type=int, default=27119901)
    parser.add_argument('--streams', type=int, default=64)
    args = parser.parse_args()
    if args.streams < 1:
        parser.error('--streams must be positive')
    result = run(args.first_seed, args.streams)
    with args.output.open('x') as out:
        json.dump(result, out, separators=(',', ':'), allow_nan=False)
    print(json.dumps(result['summary'], indent=2))
