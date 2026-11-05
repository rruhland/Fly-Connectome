"""Learn separate continuous visual responses to three anonymous requests.

Each request has 32 continuous impulse-response taps. Unissued pre-game history
is an all-zero feature vector, distinct from every actual request, including the
request that originally meant neutral. No action permutation is represented or
selected explicitly. The test's motor mapping never enters this module.
"""
import copy
from functools import lru_cache
import math

import numpy as np

from pong_visual_motor_fir import VisualMotorFIR, VisualMotorController

SCHEMA = 'public-categorical-three-request-motor-fir-v1'
PROBE_HOLD_FRAMES = 4


@lru_cache(maxsize=16)
def indices(lags, horizon):
    result = lags + np.arange(horizon)[:, None] - np.arange(lags)[None, :]
    result.flags.writeable = False
    return result


class CategoricalMotorFIR(VisualMotorFIR):
    def __init__(self, random_seed=2026110501, *, settings=None, updates_enabled=True):
        super().__init__(random_seed, settings=settings, updates_enabled=updates_enabled)
        dimension = 3 * self.settings['lags']
        self.theta = np.zeros(dimension)
        self.covariance = np.eye(dimension) * self.settings['initial_covariance']
        self.probe_remaining = 0
        self.probe_action = 1
        self.probe_blocks = 0

    def interval_features(self, commands, elapsed_seconds):
        """Latest requested command is lag0; no request means no active channel."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('positive actual public elapsed time required')
        recent = list(commands)[-self.settings['lags']:][::-1]
        features = np.zeros((3, self.settings['lags']))
        for lag, drive in enumerate(recent):
            if drive not in (-1, 0, 1):
                raise ValueError('actual anonymous request history required')
            features[int(drive) + 1, lag] = elapsed_seconds
        return features.ravel()

    def _future_feature_rows(self, commands):
        lags, horizon = self.settings['lags'], self.settings['horizon']
        past = np.full(lags, -1, dtype=np.int64)
        actual = np.asarray(list(commands)[-lags:], dtype=np.int64) + 1
        if len(actual):
            if not np.isin(actual, (0, 1, 2)).all():
                raise ValueError('actual anonymous request history required')
            past[-len(actual):] = actual
        history = np.empty((3, lags + horizon), dtype=np.int64)
        history[:, :lags] = past
        history[:, lags:] = np.arange(3)[:, None]
        requests = history[:, indices(lags, horizon)]
        terms = (requests[:, :, None, :] == np.arange(3)[None, None, :, None]).astype(float)
        terms *= self.settings['nominal_seconds']
        return np.cumsum(terms, axis=1)[:, -1].reshape(3, 3 * lags)

    def future_features(self, commands, command):
        if command not in (-1, 0, 1):
            raise ValueError('legal anonymous requested drive required')
        return self._future_feature_rows(commands)[int(command) + 1]

    def candidate_displacements(self, commands):
        return [self.predict(row) for row in self._future_feature_rows(commands)]

    def ready(self, predictions):
        return (self.updates >= self.settings['minimum_updates']
            and (max(predictions) - min(predictions)) / 2 >= self.settings['minimum_horizon_gain']
            and max(map(abs, predictions)) <= self.settings['maximum_predicted_displacement'])

    def perturb(self, action, supported):
        # A missing supported scene cancels a transient probe hold. Knowledge,
        # update counts and RNG state are retained through game resets.
        if not supported:
            self.probe_remaining = 0
            return int(action), False
        if self.probe_remaining:
            self.probe_remaining -= 1
            self.dither_count += 1
            return self.probe_action, True
        probability = self.settings['initial_dither_probability'] if self.updates < self.settings['initial_dither_updates'] else self.settings['continuing_dither_probability']
        if self.rng.random() < probability:
            self.probe_action = int(self.rng.integers(3))
            self.probe_remaining = PROBE_HOLD_FRAMES - 1
            self.probe_blocks += 1
            self.dither_count += 1
            return self.probe_action, True
        return int(action), False

    def state(self):
        state = super().state()
        state['schema'] = SCHEMA
        return state

    @classmethod
    def from_state(cls, state):
        values = copy.deepcopy(state)
        if values.pop('schema', None) != SCHEMA:
            raise ValueError('unknown categorical motor-response checkpoint')
        other = cls(settings=values['settings'], updates_enabled=values['updates_enabled'])
        rng_state = values.pop('rng_state')
        if set(values) != set(vars(other)) - {'rng'}:
            raise ValueError('categorical checkpoint fields differ')
        for key, value in values.items():
            setattr(other, key, np.asarray(value, float) if key in ('theta', 'covariance') else value)
        n = 3 * other.settings['lags']
        if (other.theta.shape != (n,) or other.covariance.shape != (n, n)
                or not np.isfinite(other.theta).all() or not np.isfinite(other.covariance).all()
                or not 0 <= other.probe_remaining < PROBE_HOLD_FRAMES
                or other.probe_action not in (0, 1, 2)):
            raise ValueError('invalid categorical checkpoint arrays/probe state')
        other.rng.bit_generator.state = rng_state
        return other


class CategoricalMotorController(VisualMotorController):
    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds, public_entities):
        if self.pending_action is None:
            raise ValueError('a chosen actual command is required')
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('positive actual public elapsed time required')
        self.inherited.observe(own_hits, terminated)
        self.learner.command_counts[self.pending_action] += 1
        self.commands.append(self.pending_action - 1)
        self.features.append(self.learner.interval_features(self.commands, elapsed_seconds))
        self.seconds += elapsed_seconds
        lookup = self._lookup(public_entities)
        self.positions.append(self._position(lookup, self.motor_id))
        self.times.append(self.seconds)
        self.target_positions.append((self.seconds, self._position(lookup, self.target_id)))
        self.last_update = None
        if (len(self.features) == self.learner.settings['window']
                and len(self.positions) == self.learner.settings['window'] + 1
                and self.positions[0] is not None and self.positions[-1] is not None):
            self.last_update = self.learner.observe(np.sum(self.features, axis=0),
                self.positions[-1] - self.positions[0], elapsed_seconds)
        self.pending_action = None
        self.latest_entities = public_entities
        return self.last_update
