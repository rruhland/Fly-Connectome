"""Learn visual choice utility on a carriage and test it on a new arm."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from time import monotonic

import numpy as np

from body_effect_learner import BodyLearner
from body_reaching_arena import (
    CAMERA_CENTER, CAMERA_SIZE, DEFAULT_MAX_STEPS, GOAL_TOLERANCE,
    MIN_START_DISTANCE, SUCCESS_TICKS, TASK_X, TASK_Y, TICK_SECONDS,
    ReachCase, ReachingArena, camera_pixel, decode_markers, render_markers,
)
from body_strategy_transfer import VISUAL_HOLD_RADIUS, VISUAL_HOLD_TICKS
from planar_position_arm import ACTION_DIM
from run_position_arm_transfer import position_arm


FIRST_SEED = 17382001
STREAMS = 64
SOURCE_TRIALS = 24
BEFORE_REVERSAL = 8
AFTER_REVERSAL = 12
BOOTSTRAP_SEED = 179591
TARGET_BODY = 'three_joint_position_arm'
FAMILIES = ('linear', 'quadratic')
CONDITIONS = ('carried', 'reset', 'shuffled', 'carried_frozen')
SCHEMA = 'visual-pairwise-polynomial-v1'
FEATURE_COUNT = 5
OLD_EVIDENCE = .91
RIDGE = .12
INITIAL_PRECISION = .65


def offset_features(markers, reference):
    """Enumerate nonconstant monomials through degree two in camera offsets."""
    x, y = (markers - reference).T
    return np.column_stack((x, y, x * x, x * y, y * y))


class UtilityLearner:
    """Fit preferences with discounted ridge regression on paired visual features."""

    def __init__(self, prior=None, *, updates_enabled=True):
        weight = np.zeros(FEATURE_COUNT)
        if prior is not None:
            if set(prior) != {'schema', 'weights'} or prior['schema'] != SCHEMA:
                raise ValueError('incompatible utility checkpoint')
            weight = np.asarray(prior['weights'], dtype=float)
            if weight.shape != (FEATURE_COUNT,) or not np.isfinite(weight).all():
                raise ValueError('expected five finite utility coefficients')
        self.weights = weight.copy()
        self.updates_enabled = updates_enabled
        self.precision = INITIAL_PRECISION * np.eye(FEATURE_COUNT)
        self.cross = INITIAL_PRECISION * weight

    def choose(self, features, tie_index):
        first, second = features @ self.weights
        if abs(first - second) <= 1e-12:
            return tie_index
        return int(second > first)

    def observe(self, difference, label):
        if self.updates_enabled:
            direction = difference / np.linalg.norm(difference)
            self.precision *= OLD_EVIDENCE
            self.precision.flat[::FEATURE_COUNT + 1] += RIDGE * (1. - OLD_EVIDENCE)
            self.precision += np.outer(direction, direction)
            self.cross = OLD_EVIDENCE * self.cross + label * direction
            self.weights = np.linalg.solve(self.precision, self.cross)

    def checkpoint(self):
        return {'schema': SCHEMA, 'weights': self.weights.tolist()}


@dataclass(frozen=True)
class Layout:
    start: tuple[float, float]
    markers: tuple[tuple[float, float], tuple[float, float]]
    reference: tuple[float, float]


def sample_layout(rng):
    while True:
        start, a, b = zip(rng.uniform(*TASK_X, 3), rng.uniform(*TASK_Y, 3))
        if (min(math.dist(start, a), math.dist(start, b)) >= MIN_START_DISTANCE
                and math.dist(a, b) >= .18):
            reference = (rng.uniform(-.10, .90), rng.uniform(-.60, .60))
            return Layout(start, (a, b), reference)


@dataclass(frozen=True)
class PayoffRule:
    """Evaluator-only utility. The learner sees only the contact scalar."""

    family: str
    angle: float
    signed_strengths: tuple[float, float]

    def winner(self, markers, reference, reversed_payoff):
        cosine, sine = math.cos(self.angle), math.sin(self.angle)
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        projections = (np.asarray(markers) - reference) @ rotation
        if self.family == 'linear':
            utilities = projections[:, 0] * self.signed_strengths[0]
        else:
            utilities = projections ** 2 @ self.signed_strengths
        return int(np.argmax(-utilities if reversed_payoff else utilities))


def sample_rule(seed, family):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    angle = rng.uniform(0., 2. * math.pi)
    strengths = rng.choice((-1., 1.), size=2) * rng.uniform((.8, .3), (1.3, .65))
    return PayoffRule(family, angle, tuple(strengths))


def decode_choice_frame(frame):
    tip, reference = decode_markers(frame[[0, 2]])
    rows, columns = np.nonzero(frame[1])
    if len(rows) != 2:
        raise ValueError('expected exactly two unlabelled choice markers')
    markers = np.column_stack((columns / CAMERA_CENTER - 1, 1 - rows / CAMERA_CENTER))
    return tip, markers, reference


def read_tip(plane):
    row, column = divmod(int(np.argmax(plane)), CAMERA_SIZE)
    return np.array(((column + 1) / CAMERA_CENTER - 1, 1 - (row + 1) / CAMERA_CENTER))


class ChoiceArena:
    """Expose public marker pixels and scalar reward during ordinary movement."""

    def __init__(self, body, layout, rule, reversed_payoff, arena_factory):
        # Use the raster's order to align the audit with the two indistinct markers.
        self._markers = sorted(layout.markers, key=camera_pixel)
        self._winner = rule.winner(self._markers, layout.reference, reversed_payoff)
        case = ReachCase(layout.start, self._markers[self._winner])
        self._native = arena_factory(body, case, max_steps=DEFAULT_MAX_STEPS)
        self._frame = np.zeros((3, CAMERA_SIZE, CAMERA_SIZE), dtype=bool)
        self._frame[2] = render_markers(layout.start, layout.reference)[1]
        for marker in self._markers:
            self._frame[(1, *camera_pixel(marker))] = True
        self._held = 0
        self._last_contact = None
        self._contacts = []

    @property
    def done(self):
        return self._native.done

    def frame(self):
        self._frame[0] = self._native.frame()[0]
        return self._frame.copy()

    def step(self, command):
        self._native.step(command)
        position = self._native.ground_truth_position
        contact = next((index for index, marker in enumerate(self._markers)
                        if math.dist(position, marker) <= GOAL_TOLERANCE), None)
        self._held = self._held + 1 if contact is not None and contact == self._last_contact else 1
        self._last_contact = contact
        reward = int(self._native.success)
        if contact is not None and self._held == SUCCESS_TICKS:
            self._contacts.append({'tick': self._native.steps, 'marker': contact, 'reward': reward})
        return self.frame(), reward

    def audit(self, first_choice, feedback, action_digest):
        return {
            'first_choice': first_choice,
            'choice_correct': first_choice == self._winner,
            'first_physical_hold_rewarded': bool(self._contacts and self._contacts[0]['reward']),
            'success': self._native.success,
            'unrewarded_holds': sum(contact['reward'] == 0 for contact in self._contacts),
            'commands': self._native.steps,
            'capped_commands': self._native.steps if self._native.success else self._native.max_steps + 1,
            'feedback': feedback,
            'feedback_verified': feedback is not None and any(
                contact['tick'] <= feedback['tick'] and contact['marker'] == feedback['marker']
                and contact['reward'] == feedback['scalar_reward'] for contact in self._contacts
            ),
            'contacts': self._contacts,
            'action_sha256': action_digest,
        }


def run_trial(arena, motor, utility, tie_index):
    motor.reset_episode()
    tip, markers, reference = decode_choice_frame(arena.frame())
    features = offset_features(markers, reference)
    difference = features[0] - features[1]
    first_choice = chosen = utility.choose(features, tie_index)
    command_count = visible_hold = 0
    feedback = None
    action_digest = sha256()
    while not arena.done:
        command = motor.act(tip, markers[chosen])
        action_digest.update(np.asarray(command, dtype='<f8').tobytes())
        frame, reward = arena.step(command)
        command_count += 1
        next_tip = read_tip(frame[0])
        motor.observe(tip, command, next_tip)
        tip = next_tip
        if feedback is not None:
            continue
        ranges = np.linalg.norm(markers - tip, axis=1)
        visible_hold = visible_hold + 1 if ranges[chosen] <= VISUAL_HOLD_RADIUS else 0
        if reward:
            seen = int(np.argmin(ranges))
        elif visible_hold >= VISUAL_HOLD_TICKS:
            seen = chosen
        else:
            continue
        # Positive payoff identifies the current marker. A reliably held zero
        # payoff makes the untried marker preferable in this two-choice task.
        preferred = seen if reward else 1 - seen
        label = 1 if preferred == 0 else -1
        utility.observe(difference, label)
        feedback = {'tick': command_count, 'marker': seen, 'scalar_reward': reward,
                    'difference': difference.tolist(), 'label': label}
        chosen = preferred
    return arena.audit(first_choice, feedback, action_digest.hexdigest())


def run_course(seed, body, rule, prior=None, *, source=False, updates_enabled=True):
    phase = 2 if source else 3
    layout_seed, tie_seed, motor_seed = np.random.SeedSequence([seed, phase]).spawn(3)
    rng = np.random.default_rng(layout_seed)
    count = SOURCE_TRIALS if source else BEFORE_REVERSAL + AFTER_REVERSAL
    ties = np.random.default_rng(tie_seed).integers(2, size=count)
    layouts = [sample_layout(rng) for _ in range(count)]
    motor = BodyLearner(action_dim=2 if source else ACTION_DIM, seed=motor_seed)
    utility = UtilityLearner(prior, updates_enabled=updates_enabled)
    factory = ReachingArena if source else position_arm
    trials = []
    for trial, (layout, tie) in enumerate(zip(layouts, ties, strict=True)):
        reversed_payoff = not source and trial >= BEFORE_REVERSAL
        arena = ChoiceArena(body, layout, rule, reversed_payoff, factory)
        result = run_trial(arena, motor, utility, int(tie))
        trials.append(dict(result, layout=asdict(layout)))
    return trials, utility.checkpoint()


def shuffled_prior(seed, trials):
    """Replay source comparisons with payoff assignment permuted across layouts."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    comparisons = [trial['feedback'] for trial in trials if trial['feedback'] is not None]
    orientation = rng.choice((-1, 1), size=len(comparisons))
    labels = [sign * item['label'] for sign, item in zip(orientation, comparisons, strict=True)]
    learner = UtilityLearner()
    for sign, item, label in zip(orientation, comparisons, rng.permutation(labels), strict=True):
        learner.observe(sign * np.asarray(item['difference']), int(label))
    return learner.checkpoint()


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def code_signatures():
    files = ('body_visual_utility.py', 'body_effect_learner.py', 'body_reaching_arena.py',
             'body_strategy_transfer.py', 'planar_position_arm.py', 'run_position_arm_transfer.py')
    return {name: fingerprint(Path(__file__).with_name(name)) for name in files}


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


def seed_and_family():
    for index in range(STREAMS):
        yield FIRST_SEED + index, FAMILIES[index % len(FAMILIES)]


def summarize(trials):
    count = len(trials)
    return {
        'trials': count,
        'first_choices_correct': sum(row['choice_correct'] for row in trials),
        'first_physical_holds_rewarded': sum(row['first_physical_hold_rewarded'] for row in trials),
        'successes': sum(row['success'] for row in trials),
        'total_commands': sum(row['commands'] for row in trials),
        'mean_capped_commands': sum(row['capped_commands'] for row in trials) / count,
        'unrewarded_holds': sum(row['unrewarded_holds'] for row in trials),
        'observed_feedback': sum(row['feedback'] is not None for row in trials),
        'unverified_feedback': sum(row['feedback'] is not None and not row['feedback_verified'] for row in trials),
    }


def fit(directory):
    directory.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    report = {
        'signatures': code_signatures(),
        'numpy': np.__version__,
        'source_trials_per_stream': SOURCE_TRIALS,
        'streams': {},
    }
    frozen = {}
    for seed, family in seed_and_family():
        rule = sample_rule(seed, family)
        trials, carried = run_course(seed, 'carriage', rule, source=True)
        frozen[str(seed)] = {
            'carried': carried,
            'reset': UtilityLearner().checkpoint(),
            'shuffled': shuffled_prior(seed, trials),
        }
        report['streams'][str(seed)] = {'rule': asdict(rule), 'trials': trials}
    all_trials = [trial for stream in report['streams'].values() for trial in stream['trials']]
    report['all'] = summarize(all_trials)
    report['by_family'] = {
        family: summarize([trial for stream in report['streams'].values()
                           if stream['rule']['family'] == family for trial in stream['trials']])
        for family in FAMILIES
    }
    report['seconds_of_carriage_control'] = report['all']['total_commands'] * TICK_SECONDS
    report['elapsed_seconds'] = monotonic() - started
    write_json(directory / 'source.json', report)
    write_json(directory / 'frozen.json', {
        'source_sha256': fingerprint(directory / 'source.json'),
        'signatures': report['signatures'],
        'utility_checkpoints': frozen,
    })
    return report


def paired_effects(streams, window):
    count = len(streams)
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(0, count, size=(4000, count))
    comparisons = {}
    for condition in CONDITIONS[1:]:
        contrasts = {}
        for field, sign in (('capped_commands', -1), ('choice_correct', 1),
                            ('first_physical_hold_rewarded', 1), ('success', 1)):
            difference = sign * np.array([
                np.mean([row[field] for row in stream['carried'][window]])
                - np.mean([row[field] for row in stream[condition][window]]) for stream in streams
            ])
            contrasts[field] = {
                'mean_gain_from_carried': float(difference.mean()),
                'ci95': np.quantile(difference[bootstrap].mean(axis=1), (.025, .975)).tolist(),
            }
        comparisons[condition] = contrasts
    return comparisons


def target_summary(streams):
    recovery = BEFORE_REVERSAL
    windows = {
        'first': slice(1),
        'before_reversal': slice(recovery),
        'first_reversed': slice(recovery, recovery + 1),
        'early_reversed': slice(recovery, recovery + 4),
        'late_reversed': slice(-4, None),
        'all_reversed': slice(recovery, None),
    }
    return {
        name: {
            'conditions': {condition: summarize([trial for stream in streams for trial in stream[condition][window]])
                           for condition in CONDITIONS},
            'paired': paired_effects(streams, window),
        }
        for name, window in windows.items()
    }


def evaluate(directory):
    started = monotonic()
    lock_path = directory / 'frozen.json'
    lock = json.loads(lock_path.read_text())
    if lock['signatures'] != code_signatures() or lock['source_sha256'] != fingerprint(directory / 'source.json'):
        raise ValueError('source and target code or frozen source evidence differ')
    report = {
        'frozen_sha256': fingerprint(lock_path),
        'target_body': TARGET_BODY,
        'target_action_dim': ACTION_DIM,
        'before_reversal': BEFORE_REVERSAL,
        'after_reversal': AFTER_REVERSAL,
        'horizon': DEFAULT_MAX_STEPS,
        'tick_seconds': TICK_SECONDS,
        'streams': {},
    }
    for seed, family in seed_and_family():
        rule = sample_rule(seed, family)
        snapshots = lock['utility_checkpoints'][str(seed)]
        arms = {'family': family}
        for condition in CONDITIONS:
            base = 'carried' if condition == 'carried_frozen' else condition
            arms[condition], _ = run_course(seed, TARGET_BODY, rule, snapshots[base],
                                            updates_enabled=condition != 'carried_frozen')
        report['streams'][str(seed)] = arms
    all_streams = list(report['streams'].values())
    report['all'] = target_summary(all_streams)
    report['by_family'] = {family: target_summary([row for row in all_streams if row['family'] == family])
                           for family in FAMILIES}
    report['elapsed_seconds'] = monotonic() - started
    write_json(directory / 'target.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'evaluate'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'fit':
        print(json.dumps(fit(args.directory)['all']))
    else:
        summary = evaluate(args.directory)['all']
        print(json.dumps({window: metrics['conditions'] for window, metrics in summary.items()}))
