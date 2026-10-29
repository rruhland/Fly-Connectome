"""Use public visual lineage memory around a frozen learned relational servo.

This module neither predicts new coordinates nor fits policy parameters. A full
observed scene anchors the original content-ranked pair; incomplete observation
can continue only the same IDs still published by the existing visual tracker.
"""
from pathlib import Path

import numpy as np

from pong_relational_feedback import RelationalPolicy, ORDER, integer_choice

SCHEMA = 'persistent-public-visual-relational-feedback-v1'
RULE = 'reanchor-exact-n;carry-same-published-ids-below-n;clear-absent-invalid-excess'
PUBLIC_FIELDS = ('id', 'normalized_position', 'observed_velocity', 'observed',
                 'velocity_known', 'observation_age_seconds',
                 'velocity_age_seconds', 'association_strength')


class PersistentRelationalPolicy:
    requires_public_visual_entities = True
    requires_stateful_reward_history = True
    requires_stateful_action_history = True

    def __init__(self, original_policy, require_first_hit=True):
        if not isinstance(original_policy, RelationalPolicy):
            raise ValueError('a frozen original relational policy is required')
        if not isinstance(require_first_hit, (bool, np.bool_)):
            raise ValueError('opening gate must be Boolean')
        # Copy the original so runtime configuration cannot alias another arm.
        self.original = RelationalPolicy(*original_policy.choices,
            original_policy.coefficients, original_policy.scales)
        self.require_first_hit = bool(require_first_hit)

    def frozen_parameters(self):
        return (SCHEMA, RULE, PUBLIC_FIELDS, self.require_first_hit,
                self.original.frozen_parameters(), 'existing-tracker-lifetime-only')

    def save(self, path):
        np.savez_compressed(Path(path), kind=SCHEMA, rule=RULE,
            choices=np.asarray(self.original.choices, np.int64),
            coefficients=self.original.coefficients, scales=self.original.scales,
            order=np.asarray(ORDER), require_first_hit=self.require_first_hit)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if (str(data['kind']) != SCHEMA or str(data['rule']) != RULE
                    or tuple(data['order']) != ORDER):
                raise ValueError('unknown persistent visual binding schema')
            gate = data['require_first_hit']
            if gate.shape != () or gate.dtype.kind != 'b':
                raise ValueError('serialized opening gate must be Boolean')
            original = RelationalPolicy(*[int(x) for x in data['choices']],
                data['coefficients'], data['scales'])
            return cls(original, bool(gate))


def public_rows(entities):
    """Validate supplied public metadata; never infer absent identity/geometry."""
    rows, metadata = [], []
    if not isinstance(entities, (list, tuple)):
        raise ValueError('public entity list required')
    seen = set()
    for entity in entities:
        if not isinstance(entity, dict) or not all(k in entity for k in PUBLIC_FIELDS):
            raise ValueError('complete public visual metadata required')
        identity = entity['id']
        if not integer_choice(identity) or identity < 0 or int(identity) in seen:
            raise ValueError('unique nonnegative public visual IDs required')
        identity = int(identity); seen.add(identity)
        p = np.asarray(entity['normalized_position'], np.float64)
        v = np.asarray(entity['observed_velocity'], np.float64)
        if p.shape != (2,) or v.shape != (2,) or not np.isfinite(np.r_[p, v]).all():
            raise ValueError('finite supplied public coordinates and velocities required')
        observed, known = entity['observed'], entity['velocity_known']
        if not all(isinstance(x, (bool, np.bool_)) for x in (observed, known)):
            raise ValueError('Boolean public observation flags required')
        age, velocity_age = entity['observation_age_seconds'], entity['velocity_age_seconds']
        strength = entity['association_strength']
        if (not np.isscalar(age) or not np.isfinite(age) or age < 0
                or (velocity_age is not None and
                    (not np.isscalar(velocity_age) or not np.isfinite(velocity_age) or velocity_age < 0))
                or not np.isscalar(strength) or not np.isfinite(strength) or strength < 0):
            raise ValueError('finite nonnegative public ages and strength required')
        rows.append([*p, *v, float(observed), float(known)])
        metadata.append(dict(id=identity, observed=bool(observed),
            velocity_known=bool(known), observation_age_seconds=float(age),
            velocity_age_seconds=None if velocity_age is None else float(velocity_age),
            association_strength=float(strength)))
    return np.asarray(rows, np.float64).reshape(-1, 6), metadata


class PersistentRelationalController:
    def __init__(self, policy):
        if not isinstance(policy, PersistentRelationalPolicy):
            raise ValueError('persistent relational policy required')
        self.policy = policy
        self.reset()

    def reset(self):
        self.past_hit = False
        self.previous_action = 1
        self.pending_action = None
        self.ordinal = 0
        self.bound_ids = None

    def _feedback(self, entities):
        original = self.policy.original
        previous_binding = self.bound_ids
        try:
            rows, metadata = public_rows(entities)
        except (ValueError, TypeError, KeyError, OverflowError):
            self.bound_ids = None
            return dict(action=None, supported=False, representation_supported=False,
                mode='baseline_invalid_public_entities', observed_entity_count=None,
                bound_ids=None, selected_entities=[], binding_changed=previous_binding is not None)
        observed = np.flatnonzero(rows[:, 4] == 1)
        count = len(observed)
        mode = 'persistent_visual_carry'
        if count == original.n:
            ordered = observed[np.lexsort(rows[observed][:, [5, 4, 3, 2, 0, 1]].T)]
            self.bound_ids = (metadata[ordered[original.i]]['id'],
                              metadata[ordered[original.j]]['id'])
            mode = 'observed_pair_reanchor'
        elif count > original.n:
            self.bound_ids = None
            mode = 'baseline_excess_observations'
        lookup = {entry['id']: i for i, entry in enumerate(metadata)}
        if self.bound_ids is not None and not all(x in lookup for x in self.bound_ids):
            self.bound_ids = None
            mode = 'baseline_selected_public_id_absent'
        if self.bound_ids is None:
            if mode == 'persistent_visual_carry':
                mode = 'baseline_no_public_binding'
            return dict(action=None, supported=False, representation_supported=False,
                mode=mode, observed_entity_count=count, bound_ids=None,
                selected_entities=[], binding_changed=previous_binding is not None)
        indices = [lookup[x] for x in self.bound_ids]
        left, right = rows[indices]
        position = float((right[original.axis] - left[original.axis]) / original.scales[original.axis])
        vi = float(left[original.axis + 2] / original.scales[original.axis + 2]) if left[5] else 0.
        vj = float(right[original.axis + 2] / original.scales[original.axis + 2]) if right[5] else 0.
        ci, cj, offset, deadzone, hysteresis = original.coefficients
        previous_drive = self.previous_action - 1
        score = float(original.polarity * (position + ci * vi + cj * vj + offset)
            + hysteresis * previous_drive)
        action = 2 if score > deadzone else 0 if score < -deadzone else 1
        return dict(action=action, supported=True, representation_supported=True,
            mode=mode, observed_entity_count=count, bound_ids=list(self.bound_ids),
            selected_entities=[metadata[i] for i in indices],
            binding_changed=self.bound_ids != previous_binding,
            position_difference=position, velocity_i=vi, velocity_j=vj,
            score=score, deadzone=float(deadzone), previous_drive=previous_drive,
            entity_i=original.i, entity_j=original.j,
            axis=original.axis, polarity=original.polarity)

    def choose(self, public_entities, baseline_action):
        if not integer_choice(baseline_action) or not 0 <= baseline_action < 3:
            raise ValueError('legal frozen baseline action required')
        if self.pending_action is not None:
            raise ValueError('observe the actual transition before choosing again')
        # Both opening modes update identical sensory binding before output gate.
        decision = self._feedback(public_entities)
        decision['binding_mode'] = decision['mode']
        if self.policy.require_first_hit and not self.past_hit:
            action = int(baseline_action)
            decision.update(mode='baseline_before_hit', supported=False)
        else:
            action = int(decision['action']) if decision['supported'] else int(baseline_action)
        self.pending_action = action
        decision.update(action=action, baseline_action=int(baseline_action),
            ordinal=self.ordinal, past_hit=self.past_hit,
            previous_actual_action=self.previous_action,
            require_first_hit=self.policy.require_first_hit)
        return action, decision

    def observe(self, hits, natural_terminal=False):
        if self.pending_action is None:
            raise ValueError('an actual chosen command must precede observe')
        self.previous_action = self.pending_action
        self.pending_action = None
        self.ordinal += 1
        self.past_hit |= bool(hits)
