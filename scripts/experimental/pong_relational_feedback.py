"""Compact generic two-entity feedback prior, with no predefined object roles.

The search selects cardinality, entity indices, axis, polarity and five numeric
parameters from actual own returns. This module only executes a frozen policy.
"""
import json
from pathlib import Path

import numpy as np

SCHEMA = 'compact-anonymous-relational-feedback-v1'
ORDER = (1, 0, 2, 3, 4, 5)
COEFFICIENT_NAMES = ('velocity_i', 'velocity_j', 'offset', 'deadzone', 'hysteresis')
CHOICE_NAMES = ('observed_cardinality', 'entity_i', 'entity_j', 'axis', 'polarity')


def integer_choice(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def load_calibration_scales(path):
    """Use the already declared 2,048-scene, reward-free sensor calibration."""
    report = json.loads(Path(path).read_text())
    if report['rows'] != 2048:
        raise ValueError('the fixed 2048-scene calibration is required')
    scales = np.maximum(np.asarray(report['0']['standard_deviations'], float),
        float(report['0']['floor']))
    if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('four finite positive unlabeled sensor scales required')
    return scales


class RelationalPolicy:
    requires_stateful_reward_history = True
    requires_stateful_action_history = True

    def __init__(self, n, i, j, axis, polarity, coefficients, scales):
        values = (n, i, j, axis, polarity)
        if not all(integer_choice(value) for value in values):
            raise ValueError('discrete choices must be integers')
        self.n, self.i, self.j, self.axis, self.polarity = map(int, values)
        if (self.n not in (2, 3, 4) or not 0 <= self.i < self.j < self.n
                or self.axis not in (0, 1) or self.polarity not in (-1, 1)):
            raise ValueError('valid cardinality, unordered pair, coordinate axis and polarity required')
        self.coefficients = np.asarray(coefficients, np.float64).copy()
        self.scales = np.asarray(scales, np.float64).copy()
        if (self.coefficients.shape != (5,) or not np.isfinite(self.coefficients).all()
                or np.any(self.coefficients[3:] < 0)):
            raise ValueError('five finite coefficients with nonnegative deadzone/hysteresis required')
        if self.scales.shape != (4,) or not np.isfinite(self.scales).all() or np.any(self.scales <= 0):
            raise ValueError('four fixed positive sensor scales required')
        self.coefficients.flags.writeable = False
        self.scales.flags.writeable = False

    @property
    def choices(self):
        return self.n, self.i, self.j, self.axis, self.polarity

    def inspect(self, scene, previous_action=1):
        if not integer_choice(previous_action) or not 0 <= previous_action < 3:
            raise ValueError('previous actual action must be a legal command index')
        rows = np.asarray(scene, np.float64)
        if (rows.ndim != 2 or rows.shape[1:] != (6,) or not np.isfinite(rows).all()
                or not np.isin(rows[:, 4:], [0, 1]).all()):
            return dict(action=None, supported=False, representation_supported=False,
                mode='baseline_invalid_scene', observed_entity_count=None)
        # Select only actual current observations. Forecast/remembered entities
        # have no implicit object role and cannot silently shift observed ranks.
        observed = rows[rows[:, 4] == 1]
        if len(observed) != self.n:
            return dict(action=None, supported=False, representation_supported=False,
                mode='baseline_cardinality_mismatch', observed_entity_count=len(observed))
        observed = observed[np.lexsort(observed[:, [5, 4, 3, 2, 0, 1]].T)]
        left, right = observed[self.i], observed[self.j]
        position = float((right[self.axis] - left[self.axis]) / self.scales[self.axis])
        vi = float(left[self.axis + 2] / self.scales[self.axis + 2]) if left[5] else 0.
        vj = float(right[self.axis + 2] / self.scales[self.axis + 2]) if right[5] else 0.
        ci, cj, offset, deadzone, hysteresis = self.coefficients
        previous_drive = int(previous_action) - 1
        score = float(self.polarity * (position + ci * vi + cj * vj + offset)
            + hysteresis * previous_drive)
        action = 2 if score > deadzone else 0 if score < -deadzone else 1
        return dict(action=action, supported=True, representation_supported=True,
            mode='relational_feedback', observed_entity_count=len(observed),
            position_difference=position, velocity_i=vi, velocity_j=vj,
            score=score, deadzone=float(deadzone), previous_drive=previous_drive,
            entity_i=self.i, entity_j=self.j, axis=self.axis, polarity=self.polarity)

    def save(self, path):
        np.savez_compressed(Path(path), kind=SCHEMA, choices=np.asarray(self.choices, np.int64),
            coefficients=self.coefficients, scales=self.scales, order=np.asarray(ORDER))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['kind']) != SCHEMA or tuple(data['order']) != ORDER:
                raise ValueError('unknown relational policy schema')
            return cls(*[int(x) for x in data['choices']], data['coefficients'], data['scales'])

    def frozen_parameters(self):
        return (SCHEMA, ORDER, CHOICE_NAMES, COEFFICIENT_NAMES, self.choices,
            self.coefficients, self.scales, 'observed-only', 'unknown-velocity-zero',
            'strict-threeway-quantizer', 'previous-actual-command-hysteresis')


class RelationalController:
    def __init__(self, policy):
        self.policy = policy
        self.past_hit = False
        self.previous_action = 1  # Neutral before any actual command.
        self.pending_action = None
        self.ordinal = 0

    def choose(self, scene, baseline_action):
        if not integer_choice(baseline_action) or not 0 <= baseline_action < 3:
            raise ValueError('legal frozen baseline command required')
        if not self.past_hit:
            action = int(baseline_action)
            decision = dict(mode='baseline_before_hit', supported=False, representation_supported=False)
        else:
            decision = self.policy.inspect(scene, self.previous_action)
            action = int(baseline_action) if not decision['supported'] else int(decision['action'])
        self.pending_action = action
        decision.update(action=action, baseline_action=int(baseline_action), ordinal=self.ordinal,
            past_hit=self.past_hit, previous_actual_action=self.previous_action)
        return action, decision

    def observe(self, hits, natural_terminal=False):
        if self.pending_action is None:
            raise ValueError('an actual chosen action must precede observe')
        self.previous_action = self.pending_action
        self.pending_action = None
        self.ordinal += 1
        self.past_hit |= bool(hits)


def structural_choices():
    """All 40 categorical settings; no named controlled/target object."""
    return [(n, i, j, axis, sign) for n in (2, 3, 4) for i in range(n)
        for j in range(i + 1, n) for axis in (0, 1) for sign in (-1, 1)]
