"""Test a utility-only response to sustained visual reward surprise."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic

import numpy as np

from body_visual_utility import (
    ACTION_DIM, TARGET_BODY, ChoiceArena, UtilityLearner, code_signatures,
    fingerprint, position_arm, run_course, run_trial, sample_layout, sample_rule,
    summarize, write_json,
)
from body_effect_learner import BodyLearner


FIRST_SEED = 53890041
STREAMS_PER_MODE = 32
MODES = ('stable', 'inverted', 'replacement')
BEFORE = 8
AFTER = 16
MARGIN_THRESHOLD = .25
CONSECUTIVE_SURPRISES = 3


class SurpriseUtility(UtilityLearner):
    """Restart utility evidence on consecutive confident contradictions."""

    def __init__(self, prior):
        super().__init__(prior)
        self.resets = 0
        self._contradictions = []

    def observe(self, difference, label):
        prediction = self.weights @ (difference / np.linalg.norm(difference))
        if prediction * label <= -MARGIN_THRESHOLD:
            self._contradictions.append((difference.copy(), label))
        else:
            self._contradictions.clear()

        if len(self._contradictions) == CONSECUTIVE_SURPRISES:
            super().__init__()
            for recent_difference, recent_label in self._contradictions:
                super().observe(recent_difference, recent_label)
            self._contradictions.clear()
            self.resets += 1
        else:
            super().observe(difference, label)


def schedule():
    for mode_index, mode in enumerate(MODES):
        for index in range(STREAMS_PER_MODE):
            family = ('linear', 'quadratic')[index % 2]
            yield FIRST_SEED + mode_index * STREAMS_PER_MODE + index, family, mode


def replacement_rule(seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
    family = ('linear', 'quadratic')[int(rng.integers(2))]
    return sample_rule(int(rng.integers(2**32)), family)


def signatures():
    path = Path(__file__)
    return {**code_signatures(), path.name: fingerprint(path)}


def fit(directory):
    directory.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    source, checkpoints = {}, {}
    for seed, family, _ in schedule():
        rule = sample_rule(seed, family)
        trials, checkpoints[str(seed)] = run_course(seed, 'carriage', rule, source=True)
        source[str(seed)] = {'rule': asdict(rule), 'trials': trials}
    report = {
        'signatures': signatures(),
        'summary': summarize([trial for row in source.values() for trial in row['trials']]),
        'streams': source,
        'elapsed_seconds': monotonic() - started,
    }
    write_json(directory / 'source.json', report)
    write_json(directory / 'frozen.json', {
        'signatures': report['signatures'],
        'source_sha256': fingerprint(directory / 'source.json'),
        'checkpoints': checkpoints,
    })
    return report


def target_course(seed, mode, old_rule, new_rule, prior, detect):
    rng_seed, motor_seed, tie_seed = np.random.SeedSequence([seed, 11]).spawn(3)
    rng = np.random.default_rng(rng_seed)
    ties = np.random.default_rng(tie_seed).integers(2, size=BEFORE + AFTER)
    motor = BodyLearner(action_dim=ACTION_DIM, seed=motor_seed)
    utility = SurpriseUtility(prior) if detect else UtilityLearner(prior)
    trials = []
    for index, tie in enumerate(ties):
        active_rule = new_rule if index >= BEFORE and mode == 'replacement' else old_rule
        reversed_payoff = index >= BEFORE and mode == 'inverted'
        layout = sample_layout(rng)
        arena = ChoiceArena(TARGET_BODY, layout, active_rule, reversed_payoff, position_arm)
        trial = run_trial(arena, motor, utility, int(tie))
        old_winner = old_rule.winner(layout.markers, layout.reference, False)
        new_winner = active_rule.winner(layout.markers, layout.reference, reversed_payoff)
        trials.append(dict(trial, layout=asdict(layout), lifetime_motor_updates=motor.updates,
                           utility_resets=utility.resets if detect else 0,
                           old_rule_disagrees=old_winner != new_winner))
    return trials


def detection_summary(streams, mode):
    reset_trials = [
        [i for i, row in enumerate(stream['detector'])
         if row['utility_resets'] > (stream['detector'][i - 1]['utility_resets'] if i else 0)]
        for stream in streams
    ]
    first_post_change_delays = [
        next(index - BEFORE + 1 for index in resets if index >= BEFORE)
        for resets in reset_trials if any(index >= BEFORE for index in resets)
    ]
    return {
        'streams_with_any_reset': sum(bool(row) for row in reset_trials),
        'resets_all': sum(len(row) for row in reset_trials),
        'false_resets': sum(index < BEFORE or mode == 'stable' for row in reset_trials for index in row),
        'post_boundary_streams_with_reset': len(first_post_change_delays),
        'median_first_post_boundary_reset_trial': (
            float(np.median(first_post_change_delays)) if first_post_change_delays else None
        ),
        'first_post_boundary_reset_trial_by_detected_stream': first_post_change_delays,
        'mean_disagreement_between_old_and_active_rule': float(np.mean([
            row['old_rule_disagrees'] for stream in streams for row in stream['detector'][BEFORE:]
        ])),
    }


def period_summary(streams, window):
    outcomes = {
        label: summarize([trial for stream in streams for trial in stream[label][window]])
        for label in ('adaptive', 'detector')
    }
    rng = np.random.default_rng(63944075)
    indices = rng.integers(0, len(streams), size=(4000, len(streams)))
    contrasts = {}
    for field, direction in (('capped_commands', -1), ('first_physical_hold_rewarded', 1),
                             ('choice_correct', 1), ('success', 1)):
        changes = np.array([
            direction * (np.mean([row[field] for row in stream['detector'][window]])
                         - np.mean([row[field] for row in stream['adaptive'][window]]))
            for stream in streams
        ])
        contrasts[field] = {
            'mean_gain_from_detector': float(changes.mean()),
            'ci95': np.quantile(changes[indices].mean(axis=1), (.025, .975)).tolist(),
        }
    return {'outcomes': outcomes, 'paired_contrasts': contrasts}


def evaluate(directory):
    started = monotonic()
    lock = directory / 'frozen.json'
    frozen = json.loads(lock.read_text())
    if frozen['signatures'] != signatures() or frozen['source_sha256'] != fingerprint(directory / 'source.json'):
        raise ValueError('frozen source, detector, or dependency changed')
    streams = {}
    for seed, family, mode in schedule():
        original = sample_rule(seed, family)
        replacement = replacement_rule(seed)
        prior = frozen['checkpoints'][str(seed)]
        streams[str(seed)] = {
            'mode': mode,
            'replacement': asdict(replacement),
            'adaptive': target_course(seed, mode, original, replacement, prior, False),
            'detector': target_course(seed, mode, original, replacement, prior, True),
        }
    windows = {
        'all': slice(None),
        'before_boundary': slice(BEFORE),
        'first_after_boundary': slice(BEFORE, BEFORE + 1),
        'early_after_boundary': slice(BEFORE, BEFORE + 6),
        'late_after_boundary': slice(-6, None),
        'all_after_boundary': slice(BEFORE, None),
    }
    report = {
        'frozen_sha256': fingerprint(lock),
        'streams_per_mode': STREAMS_PER_MODE,
        'trials_before_boundary': BEFORE,
        'trials_after_boundary': AFTER,
        'detector_margin_threshold': MARGIN_THRESHOLD,
        'detector_consecutive_surprises': CONSECUTIVE_SURPRISES,
        'streams': streams,
        'modes': {},
        'elapsed_seconds': 0,
    }
    for mode in MODES:
        matched = [row for row in streams.values() if row['mode'] == mode]
        report['modes'][mode] = {
            'detection': detection_summary(matched, mode),
            'periods': {name: period_summary(matched, window) for name, window in windows.items()},
        }
    report['elapsed_seconds'] = monotonic() - started
    write_json(directory / 'target.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'evaluate'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'fit':
        print(json.dumps(fit(args.directory)['summary']))
    else:
        report = evaluate(args.directory)
        print(json.dumps({mode: row['detection'] for mode, row in report['modes'].items()}))
