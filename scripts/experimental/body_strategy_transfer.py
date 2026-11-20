"""Carry a visually grounded reward association across simulated bodies."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from body_effect_learner import BodyLearner
from body_reaching_arena import (
    CAMERA_CENTER,
    CAMERA_SIZE,
    DEFAULT_MAX_STEPS,
    GOAL_TOLERANCE,
    MIN_START_DISTANCE,
    SUCCESS_TICKS,
    TASK_X,
    TASK_Y,
    TICK_SECONDS,
    ReachCase,
    ReachingArena,
    camera_pixel,
    decode_markers,
    render_markers,
)


RELATIONS = ('near', 'far')
CONDITIONS = ('carried', 'reset', 'source_reward_masked')
SOURCE_BODY = 'carriage'
TARGET_BODY = 'arm_remapped'
SCALAR = 'scalar'
HELD_EVENT = 'held-event'
ASSOCIATION_SCHEMA = 'visual-reward-association-v1'
# The relative position of two rounded camera centers can differ by one pixel
# per axis. Leave that geometric margin and one extra tick before rejecting.
VISUAL_HOLD_RADIUS = GOAL_TOLERANCE - math.sqrt(2) / CAMERA_CENTER
VISUAL_HOLD_TICKS = SUCCESS_TICKS + 1


@dataclass(frozen=True)
class CueLayout:
    start: tuple[float, float]
    markers: tuple[tuple[float, float], tuple[float, float]]
    reference: tuple[float, float]

    def by_distance(self):
        return sorted(self.markers, key=lambda marker: math.dist(marker, self.reference))


def sample_layout(rng):
    """Sample both choices from the shared workspace and keep both visually distinct."""
    while True:
        start, a, b = zip(rng.uniform(*TASK_X, 3), rng.uniform(*TASK_Y, 3))
        reference = tuple(rng.uniform(-.75, .75, 2))
        if (min(math.dist(start, a), math.dist(start, b)) >= MIN_START_DISTANCE
                and math.dist(a, b) >= .18
                and abs(math.dist(a, reference) - math.dist(b, reference)) >= .10):
            return CueLayout(start, (a, b), reference)


def decode_relation_frame(frame):
    """Decode the tip and rank two same-appearance cue pixels by reference distance."""
    tip, reference = decode_markers(frame[[0, 2]])
    rows, columns = np.nonzero(frame[1])
    if len(rows) != len(RELATIONS):
        raise ValueError('expected two visual candidate markers')
    markers = np.column_stack((columns / CAMERA_CENTER - 1, 1 - rows / CAMERA_CENTER))
    order = np.argsort(np.linalg.norm(markers - reference, axis=1))
    return tip, markers[order]


class RelationArena:
    """Expose the shared cue image and ordinary scalar reward to the learner."""

    def __init__(self, body, layout, rewarded_relation, max_steps, *, feedback=SCALAR,
                 arena_factory=ReachingArena):
        self._goals = layout.by_distance()
        self._rewarded_relation = rewarded_relation
        self._feedback = feedback
        case = ReachCase(layout.start, self._goals[rewarded_relation])
        self._arena = arena_factory(body, case, max_steps=max_steps)
        self._frame = np.zeros((3, CAMERA_SIZE, CAMERA_SIZE), dtype=bool)
        self._frame[2] = render_markers(layout.start, layout.reference)[1]
        for marker in layout.markers:
            self._frame[(1, *camera_pixel(marker))] = True
        self._last_contact = None
        self._held = 0
        self._contacts = []

    @property
    def done(self):
        return self._arena.done

    @property
    def steps(self):
        return self._arena.steps

    @property
    def success(self):
        return self._arena.success

    def frame(self):
        # Replace the native private goal plane with a public, unlabelled choice.
        self._frame[0] = self._arena.frame()[0]
        return self._frame.copy()

    def step(self, action):
        self._arena.step(action)
        position = self._arena.ground_truth_position
        contact = next((index for index, goal in enumerate(self._goals)
                        if math.dist(position, goal) <= GOAL_TOLERANCE), None)
        self._held = self._held + 1 if contact is not None and contact == self._last_contact else 1
        self._last_contact = contact
        reward = int(self.success)
        new_hold = contact is not None and self._held == SUCCESS_TICKS
        if new_hold:
            self._contacts.append({'tick': self.steps, 'relation': RELATIONS[contact], 'reward': reward})
        if self._feedback == HELD_EVENT and not new_hold:
            reward = None
        return self.frame(), reward

    def audit(self, learning_events):
        """Check image-based inferences against physical contact after the trial."""
        rejections = []
        approach_start = 0
        for event in learning_events:
            if event['reward'] != 0:
                continue
            relation = event['relation']
            verified = any(approach_start < contact['tick'] <= event['tick']
                           and contact['relation'] == relation and contact['reward'] == 0
                           for contact in self._contacts)
            rewarded_relation = relation == RELATIONS[self._rewarded_relation]
            rejections.append(dict(event, physical_hold_verified=verified,
                                   rejected_rewarded_relation=rewarded_relation))
            approach_start = event['tick']
        return {'contacts': self._contacts, 'rejections': rejections}


class RewardAssociation:
    """Remember which visual relation last paid off and clear it on a verified miss."""

    def __init__(self, preferred=None):
        self.preferred = preferred

    def observe(self, relation, reward):
        if reward == 1:
            self.preferred = relation
        elif reward == 0 and self.preferred == relation:
            self.preferred = None

    def export_prior(self):
        relation = None if self.preferred is None else RELATIONS[self.preferred]
        return {'schema': ASSOCIATION_SCHEMA, 'preferred_relation': relation}

    @classmethod
    def from_prior(cls, prior):
        if (not isinstance(prior, dict) or set(prior) != {'schema', 'preferred_relation'}
                or prior['schema'] != ASSOCIATION_SCHEMA):
            raise ValueError('incompatible visual reward association checkpoint')
        relation = prior['preferred_relation']
        if relation is not None and relation not in RELATIONS:
            raise ValueError('checkpoint relation must be near, far, or null')
        return cls(None if relation is None else RELATIONS.index(relation))


def run_trial(arena, motor, memory, first_guess, *, feedback=SCALAR):
    motor.reset_episode()
    tip, markers = decode_relation_frame(arena.frame())
    selected = first_guess if memory.preferred is None else memory.preferred
    first_choice = RELATIONS[selected]
    learning_events = []
    visible_hold = 0
    action_digest = sha256()
    while not arena.done:
        command = motor.act(tip, markers[selected])
        action_digest.update(np.asarray(command, dtype='<f8').tobytes())
        frame, reward = arena.step(command)
        next_tip, markers = decode_relation_frame(frame)
        motor.observe(tip, command, next_tip)
        tip = next_tip
        ranges = np.linalg.norm(markers - tip, axis=1)
        visible_hold = visible_hold + 1 if ranges[selected] <= VISUAL_HOLD_RADIUS else 0
        if reward == 1 or (feedback == HELD_EVENT and reward == 0):
            related = int(np.argmin(ranges))
        elif feedback == SCALAR and visible_hold >= VISUAL_HOLD_TICKS:
            related = selected
        else:
            continue
        memory.observe(related, reward)
        learning_events.append({'tick': arena.steps, 'relation': RELATIONS[related], 'reward': reward})
        if reward == 0:
            selected = 1 - related
            visible_hold = 0
    return {'first_choice': first_choice, 'steps': arena.steps, 'success': arena.success,
            'action_sha256': action_digest.hexdigest(), 'learning_events': learning_events,
            **arena.audit(learning_events)}


def _course(body, layouts, rewards, guesses, constructor_seed, memory, max_steps, feedback, *,
            action_dim=2, arena_factory=ReachingArena):
    motor = BodyLearner(action_dim=action_dim, seed=constructor_seed)
    return [run_trial(RelationArena(body, layout, reward, max_steps, feedback=feedback,
                                   arena_factory=arena_factory),
                      motor, memory, guess, feedback=feedback)
            for layout, reward, guess in zip(layouts, rewards, guesses, strict=True)]


def _stream(seed, source_trials, target_trials, reverse_after, max_steps, feedback, *,
            target_body, target_action_dim, target_arena_factory):
    *streams, source_motor_seed, target_motor_seed = np.random.SeedSequence(seed).spawn(7)
    if feedback == HELD_EVENT:
        # Preserve the original scouting schedule and motor seeds for this mode.
        source_motor_seed, target_motor_seed = streams[1:3]
    rule_rng, source_rng, target_rng, source_ties, target_ties = (
        np.random.default_rng(stream) for stream in streams)
    relation = int(rule_rng.integers(2))
    source_layouts = [sample_layout(source_rng) for _ in range(source_trials)]
    target_layouts = [sample_layout(target_rng) for _ in range(target_trials)]
    source_guesses = list(map(int, source_ties.integers(2, size=source_trials)))
    target_guesses = list(map(int, target_ties.integers(2, size=target_trials)))
    target_rewards = [relation] * reverse_after + [1 - relation] * (target_trials - reverse_after)
    source_memory = RewardAssociation()
    source = _course(SOURCE_BODY, source_layouts, [relation] * source_trials, source_guesses,
                     source_motor_seed, source_memory, max_steps, feedback)
    masked_memory = RewardAssociation()
    for trial in source:
        for event in trial['learning_events']:
            masked_memory.observe(RELATIONS.index(event['relation']), None)
    snapshots = {label: memory.export_prior() for label, memory in zip(
        CONDITIONS, (source_memory, RewardAssociation(), masked_memory), strict=True)}
    result = {
        'seed': seed,
        'source_rewarded_relation': RELATIONS[relation],
        'source_layouts': [asdict(layout) for layout in source_layouts],
        'source': source,
        'target_layouts': [asdict(layout) for layout in target_layouts],
        'target_rewarded_relations': [RELATIONS[value] for value in target_rewards],
        'initial_associations': {label: prior['preferred_relation'] for label, prior in snapshots.items()},
        'association_checkpoints': snapshots,
    }
    for label, prior in snapshots.items():
        target_memory = RewardAssociation.from_prior(json.loads(json.dumps(prior)))
        result[label] = _course(target_body, target_layouts, target_rewards, target_guesses,
                                target_motor_seed, target_memory, max_steps, feedback,
                                action_dim=target_action_dim, arena_factory=target_arena_factory)
    return result


def _summary(trials, max_steps):
    successes = sum(trial['success'] for trial in trials)
    rejections = [rejection for trial in trials for rejection in trial['rejections']]
    return {
        'trials': len(trials),
        'successes': successes,
        'timeouts': len(trials) - successes,
        'first_contact_rewarded': sum(bool(trial['contacts'] and trial['contacts'][0]['reward'])
                                      for trial in trials),
        'rewarded_contacts': sum(contact['reward'] for trial in trials for contact in trial['contacts']),
        'unrewarded_contacts': sum(not contact['reward'] for trial in trials for contact in trial['contacts']),
        'rejections': len(rejections),
        'unverified_rejections': sum(not rejection['physical_hold_verified'] for rejection in rejections),
        'rejected_rewarded_relation': sum(rejection['rejected_rewarded_relation'] for rejection in rejections),
        'mean_capped_ticks': sum(_capped_ticks(trial, max_steps) for trial in trials) / len(trials),
    }


def _capped_ticks(trial, max_steps):
    return trial['steps'] if trial['success'] else max_steps + 1


def _paired_first_ticks(streams, max_steps):
    savings = np.array([_capped_ticks(row['reset'][0], max_steps)
                        - _capped_ticks(row['carried'][0], max_steps) for row in streams])
    rng = np.random.default_rng(120799)
    bootstrap = rng.choice(savings, size=(4000, len(savings)), replace=True).mean(axis=1)
    return {'mean': float(savings.mean()), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'ahead': int((savings > 0).sum()), 'tied': int((savings == 0).sum()),
            'behind': int((savings < 0).sum())}


def run_experiment(seeds=range(32), *, source_trials=4, target_trials=8, reverse_after=4,
                   max_steps=DEFAULT_MAX_STEPS, feedback=SCALAR, target_body=TARGET_BODY,
                   target_action_dim=2, target_arena_factory=ReachingArena):
    """Run matched trials with a hidden goal convention and fresh receiving motor learners."""
    seeds = list(map(int, seeds))
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError('provide distinct nonnegative stream seeds')
    if source_trials < 1 or not 0 < reverse_after < target_trials or max_steps < SUCCESS_TICKS:
        raise ValueError('provide source trials, target trials before and after reversal, and enough hold ticks')
    if feedback not in (SCALAR, HELD_EVENT):
        raise ValueError('choose scalar or held-event feedback')
    streams = [_stream(seed, source_trials, target_trials, reverse_after, max_steps, feedback,
                       target_body=target_body, target_action_dim=target_action_dim,
                       target_arena_factory=target_arena_factory) for seed in seeds]
    periods = {'first': slice(0, 1), 'before_reversal': slice(0, reverse_after),
               'reversal': slice(reverse_after, reverse_after + 1),
               'after_reversal': slice(reverse_after, None)}
    summaries = {}
    for period, window in periods.items():
        summaries[period] = {}
        for label in CONDITIONS:
            trials = [trial for row in streams for trial in row[label][window]]
            summaries[period][label] = _summary(trials, max_steps)
    paths = (Path(__file__), Path(__file__).with_name('body_effect_learner.py'),
             Path(__file__).with_name('body_reaching_arena.py'))
    return {
        'schema': 'visual-reward-association-transfer-v1',
        'source_sha256': {path.name: sha256(path.read_bytes()).hexdigest() for path in paths},
        'source_body': SOURCE_BODY, 'target_body': target_body, 'target_action_dim': target_action_dim,
        'tick_seconds': TICK_SECONDS,
        'feedback': feedback,
        'visual_hold_radius': VISUAL_HOLD_RADIUS, 'visual_hold_ticks': VISUAL_HOLD_TICKS,
        'max_steps': max_steps, 'source_trials': source_trials, 'target_trials': target_trials,
        'reverse_after': reverse_after,
        'source': _summary([trial for row in streams for trial in row['source']], max_steps),
        'paired_first_ticks_saved': _paired_first_ticks(streams, max_steps),
        'summary': summaries, 'streams': streams,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=range(32))
    parser.add_argument('--source-trials', type=int, default=4)
    parser.add_argument('--target-trials', type=int, default=8)
    parser.add_argument('--reverse-after', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument('--feedback', choices=(SCALAR, HELD_EVENT), default=SCALAR)
    parser.add_argument('--target-body', default=TARGET_BODY)
    parser.add_argument('--target-action-dim', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
