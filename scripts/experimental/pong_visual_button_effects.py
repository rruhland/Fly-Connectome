"""Translate a visual interception intent using measured effects of held requests."""

import copy
import math

import numpy as np

from pong_relational_persistent_binding import PersistentRelationalController

SCHEMA = 'public-visual-inverse-button-effects-v1'
PROBE_SETTLE = 14
PROBE_MEASURE = 10
MIN_EFFECT_SEPARATION = 2 / 64
PROBE_START_RADIUS = 5 / 64
STALL_RADIUS = 4 / 64
STALL_MOTION = 1 / 64
REVERSED_MOTION = 3 / 64


class VisualButtonEffects:
    """Keep the latest well-separated displacement ordering through each game."""

    def __init__(self, random_seed=711093, *, updates_enabled=True):
        self.rng = np.random.default_rng(random_seed)
        self.updates_enabled = updates_enabled
        self.effects = None
        self.measurements = [None] * 3
        self.probe_queue = []
        self.recalibration_requested = True
        self.calibrations = 0
        self.completed_calibrations = 0
        self.contradictions = 0
        self.dither_count = 0
        self.updates = 0

    def request(self, intended):
        if self.effects is None:
            return intended
        # The inherited requested drives (-1, 0, +1) describe downward,
        # stationary, and upward intention in the observed vertical coordinate.
        ordered = sorted(range(3), key=self.effects.__getitem__, reverse=True)
        return ordered[intended]

    def start_probes(self):
        self.probe_queue = self.rng.permutation(3).tolist()
        self.measurements = [None] * 3
        self.recalibration_requested = False
        self.calibrations += 1

    def record_probe(self, displacement):
        request = self.probe_queue.pop(0)
        self.measurements[request] = displacement
        self.updates += 1
        if self.probe_queue:
            return
        lower, rest, upper = sorted(self.measurements)
        if (lower <= -MIN_EFFECT_SEPARATION and upper >= MIN_EFFECT_SEPARATION
                and rest - lower >= MIN_EFFECT_SEPARATION
                and upper - rest >= MIN_EFFECT_SEPARATION):
            self.effects = self.measurements.copy()
            self.completed_calibrations += 1
        else:
            self.recalibration_requested = True

    def flag_contradiction(self):
        if not self.recalibration_requested:
            self.contradictions += 1
            self.recalibration_requested = True

    def state(self):
        return dict(schema=SCHEMA, **copy.deepcopy({k: v for k, v in vars(self).items() if k != 'rng'}),
                    rng_state=copy.deepcopy(self.rng.bit_generator.state))

    @classmethod
    def from_state(cls, state):
        values = copy.deepcopy(state)
        if values.pop('schema', None) != SCHEMA:
            raise ValueError('unknown visual button response checkpoint')
        rng_state = values.pop('rng_state')
        learner = cls()
        if set(values) != set(vars(learner)) - {'rng'}:
            raise ValueError('visual button response checkpoint fields differ')
        learner.__dict__.update(values)
        learner.rng.bit_generator.state = rng_state
        return learner

    def frozen_branch(self):
        frozen = self.from_state(self.state())
        frozen.updates_enabled = False
        return frozen


class VisualIntentController:
    """Keep the inherited visual set point and change which request executes it."""

    def __init__(self, policy, learner):
        self.inherited = PersistentRelationalController(policy)
        self.learner = learner
        self.motor = None
        self.episode_anchor = None
        self.pending = None
        self.last_action = None
        self.dwell = 0
        self.anchor = None
        self.probe_active = False
        self.probe_attempted_here = False

    def _motor_position(self, entities):
        for entity in entities:
            if entity['id'] == self.motor and entity['observed']:
                return float(entity['normalized_position'][self.inherited.policy.original.axis])
        return None

    def _reset_dwell(self):
        self.last_action = None
        self.dwell = 0
        self.anchor = None

    def choose(self, entities, baseline_action):
        intended, decision = self.inherited.choose(entities, baseline_action)
        if decision['bound_ids'] is not None and decision['bound_ids'][0] != self.motor:
            self.motor = decision['bound_ids'][0]
            self.episode_anchor = None
            self._reset_dwell()
        position = self._motor_position(entities)
        if self.episode_anchor is None and position is not None:
            self.episode_anchor = position
        if (self.learner.updates_enabled and self.learner.recalibration_requested
                and not self.probe_attempted_here and position is not None
                and abs(position - self.episode_anchor) <= PROBE_START_RADIUS):
            self.learner.start_probes()
            self.probe_attempted_here = True
        active = self.learner.updates_enabled and bool(self.learner.probe_queue) and position is not None
        if active != self.probe_active:
            self._reset_dwell()
        self.probe_active = active
        if active:
            action = self.learner.probe_queue[0]
            self.learner.dither_count += 1
        else:
            action = self.learner.request(intended)
        self.pending = action, intended, position
        # Keep the relational layer on its native signed intent and observations.
        return action, dict(intended=intended, action=action, probing=active,
                            effective_requests=[self.learner.request(i) for i in range(3)])

    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds, public_entities):
        if self.pending is None or elapsed_seconds <= 0:
            raise ValueError('a chosen request and positive public elapsed time are required')
        action, intended, before = self.pending
        self.pending = None
        self.inherited.observe(own_hits, terminated)
        if not self.learner.updates_enabled:
            return
        after = self._motor_position(public_entities)
        if before is None or after is None:
            self._reset_dwell()
            return
        if action != self.last_action:
            self._reset_dwell()
            self.last_action = action
        self.dwell += 1
        if self.dwell == PROBE_SETTLE:
            self.anchor = after
        if self.dwell != PROBE_SETTLE + PROBE_MEASURE:
            return
        displacement = after - self.anchor
        if not math.isfinite(displacement):
            raise ValueError('finite visual displacement required')
        if self.probe_active:
            self.learner.record_probe(displacement)
            self._reset_dwell()
            return
        if self.learner.effects is None:
            return
        if self.learner.request(1) == action:
            contradicted = abs(displacement) >= 2 * REVERSED_MOTION
        else:
            expected = 1 if self.learner.request(0) == action else -1
            reversed_direction = expected * displacement <= -REVERSED_MOTION
            center_stall = (abs(displacement) <= STALL_MOTION
                            and abs(after - self.episode_anchor) <= STALL_RADIUS)
            contradicted = reversed_direction or center_stall
        if contradicted:
            self.learner.flag_contradiction()
        self.dwell = PROBE_SETTLE
        self.anchor = after
