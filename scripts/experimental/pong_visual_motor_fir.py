"""Causal numeric visual motor-response learning over an inherited relation.

Only the controlled member of the previously acquired anonymous relation is
inherited. Its displacement response to requested commands is estimated online;
no actuator mapping, simulator state or physical coefficient enters this module.
The learned FIR is a local linear approximation, not an exact physics model.
"""
from collections import deque
import copy
import math

import numpy as np

from pong_relational_persistent_binding import PersistentRelationalController, public_rows

SCHEMA = 'public-visual-motor-fir-numeric-learning-v1'
DEFAULTS = dict(lags=32, window=4, horizon=16, nominal_seconds=.02,
    forgetting_seconds=20., initial_covariance=100., minimum_updates=96,
    minimum_horizon_gain=1 / 128, maximum_predicted_displacement=.5,
    target_lead_limit=1 / 16, command_cost=(1 / 128) ** 2,
    initial_dither_updates=256, initial_dither_probability=.10,
    continuing_dither_probability=.01)


class VisualMotorFIR:
    """Predict displacement from elapsed-time-weighted command histories.

    For every interval, feature k+1 is dt times the command requested k
    intervals ago; feature0 is dt. A four-interval sum averages visual raster
    quantization. The current target is never a regression input or label.
    Forgetting advances by the actual duration of each newly admitted observed
    transition; unsupported gaps do not age the model. Its clock is therefore
    supported-observation exposure rather than total game time.
    """
    def __init__(self, random_seed=2026110501, *, settings=None, updates_enabled=True):
        self.settings = dict(DEFAULTS if settings is None else settings)
        if set(self.settings) != set(DEFAULTS):
            raise ValueError('complete fixed numeric-learning settings required')
        s = self.settings
        if any(type(s[k]) is not int or s[k] < 1 for k in
               ('lags', 'window', 'horizon', 'minimum_updates', 'initial_dither_updates')):
            raise ValueError('positive integer history lengths required')
        if (any(not math.isfinite(float(v)) for v in s.values())
                or s['nominal_seconds'] <= 0 or s['forgetting_seconds'] <= 0
                or s['initial_covariance'] <= 0
                or not 0 <= s['initial_dither_probability'] <= 1
                or not 0 <= s['continuing_dither_probability'] <= 1):
            raise ValueError('finite positive timing and bounded dither required')
        self.theta = np.zeros(s['lags'] + 1)
        self.covariance = np.eye(len(self.theta)) * s['initial_covariance']
        self.rng = np.random.default_rng(random_seed)
        self.updates_enabled = bool(updates_enabled)
        self.updates = 0
        self.observations = 0
        self.elapsed_training_seconds = 0.
        self.preupdate_squared_error = 0.
        self.zero_model_squared_error = 0.
        self.prediction_count = 0
        self.command_counts = [0, 0, 0]
        self.dither_count = 0

    def predict(self, features):
        x = np.asarray(features, float)
        if x.shape != self.theta.shape or not np.isfinite(x).all():
            raise ValueError('finite command-history feature vector required')
        return float(self.theta @ x)

    def observe(self, features, displacement, elapsed_seconds):
        x = np.asarray(features, float)
        prediction = self.predict(x)
        if not math.isfinite(displacement) or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('observed displacement and positive elapsed time required')
        residual = float(displacement - prediction)
        # Prediction/error are recorded BEFORE fitting this newly observed target.
        self.observations += 1
        self.preupdate_squared_error += residual ** 2
        self.zero_model_squared_error += float(displacement) ** 2
        self.prediction_count += 1
        if self.updates_enabled:
            forgetting = math.exp(-elapsed_seconds / self.settings['forgetting_seconds'])
            px = self.covariance @ x
            gain = px / (forgetting + float(x @ px))
            self.theta += gain * residual
            self.covariance = (self.covariance - np.outer(gain, px)) / forgetting
            self.covariance = (self.covariance + self.covariance.T) / 2
            self.updates += 1
            self.elapsed_training_seconds += elapsed_seconds
            if not np.isfinite(self.theta).all() or not np.isfinite(self.covariance).all():
                raise FloatingPointError('nonfinite retained motor-response state')
        return dict(prediction=prediction, displacement=float(displacement),
            residual=residual, updated=self.updates_enabled, updates=self.updates)

    def future_features(self, commands, command):
        """Known requested history plus one constant future candidate command.

        This is a learned public-coordinate forecast. It does not query the game.
        Future camera intervals use the inherited nominal .02-second clock;
        actual fitting always uses the elapsed time returned by execution.
        """
        lags = self.settings['lags']
        previous = ([0.] * lags + list(commands))[-lags:]
        total = np.zeros(lags + 1)
        for _ in range(self.settings['horizon']):
            previous.append(float(command))
            total += np.r_[1., previous[-lags:][::-1]] * self.settings['nominal_seconds']
        return total

    def candidate_displacements(self, commands):
        return [self.predict(self.future_features(commands, drive)) for drive in (-1, 0, 1)]

    def ready(self, predictions):
        return (self.updates >= self.settings['minimum_updates']
            and abs(predictions[2] - predictions[0]) / 2 >= self.settings['minimum_horizon_gain']
            and max(map(abs, predictions)) <= self.settings['maximum_predicted_displacement'])

    def perturb(self, action, supported):
        if not supported:
            return int(action), False
        p = self.settings['initial_dither_probability'] if self.updates < self.settings['initial_dither_updates'] else self.settings['continuing_dither_probability']
        if self.rng.random() < p:
            chosen = int(self.rng.integers(3))
            self.dither_count += 1
            return chosen, True
        return int(action), False

    def state(self):
        values = {key: copy.deepcopy(value) for key, value in vars(self).items()
                  if key not in ('theta', 'covariance', 'rng')}
        return dict(schema=SCHEMA, **values, theta=self.theta.tolist(),
            covariance=self.covariance.tolist(), rng_state=copy.deepcopy(self.rng.bit_generator.state))

    @classmethod
    def from_state(cls, state):
        values = copy.deepcopy(state)
        if values.pop('schema', None) != SCHEMA:
            raise ValueError('unknown numeric motor checkpoint')
        other = cls(settings=values['settings'], updates_enabled=values['updates_enabled'])
        rng_state = values.pop('rng_state')
        if set(values) != set(vars(other)) - {'rng'}:
            raise ValueError('numeric checkpoint fields differ')
        for key, value in values.items():
            setattr(other, key, np.asarray(value, float) if key in ('theta', 'covariance') else value)
        n = other.settings['lags'] + 1
        if (other.theta.shape != (n,) or other.covariance.shape != (n, n)
                or not np.isfinite(other.theta).all() or not np.isfinite(other.covariance).all()):
            raise ValueError('invalid numeric checkpoint arrays')
        other.rng.bit_generator.state = rng_state
        return other

    def frozen_branch(self):
        other = self.from_state(self.state())
        other.updates_enabled = False
        return other


class VisualMotorController:
    """Inherited visual relation plus an online learned motor-response forecast."""
    def __init__(self, inherited_policy, learner):
        self.inherited = PersistentRelationalController(inherited_policy)
        self.learner = learner
        self.reset()

    def reset(self):
        self.inherited.reset()
        self.commands = deque(maxlen=self.learner.settings['lags'])
        self.features = deque(maxlen=self.learner.settings['window'])
        self.positions = deque(maxlen=self.learner.settings['window'] + 1)
        self.times = deque(maxlen=self.learner.settings['window'] + 1)
        self.target_positions = deque(maxlen=self.learner.settings['window'] + 1)
        self.motor_id = None
        self.target_id = None
        self.seconds = 0.
        self.pending_action = None
        self.latest_entities = None
        self.last_update = None
        self.numeric_actions = 0
        self.changed_from_inherited = 0

    def _lookup(self, entities):
        try:
            rows, metadata = public_rows(entities)
        except (ValueError, TypeError, KeyError, OverflowError):
            return {}
        return {m['id']: (r, m) for r, m in zip(rows, metadata)}

    def _position(self, lookup, identity):
        entry = lookup.get(identity)
        if entry is None or not entry[1]['observed']:
            return None
        return float(entry[0][self.inherited.policy.original.axis])

    def _bind(self, decision, lookup):
        if decision.get('bound_ids') is not None:
            motor, target = decision['bound_ids']
            if motor != self.motor_id:
                self.motor_id = motor
                self.positions.clear()
                self.features.clear()
                self.times.clear()
            if target != self.target_id:
                self.target_id = target
                self.target_positions.clear()
        if not self.positions:
            self.positions.append(self._position(lookup, self.motor_id))
            self.times.append(self.seconds)
        if not self.target_positions:
            self.target_positions.append((self.seconds, self._position(lookup, self.target_id)))

    def choose(self, public_entities, baseline_action):
        if self.pending_action is not None:
            raise ValueError('observe actual execution before next choice')
        inherited_action, inherited_decision = self.inherited.choose(public_entities, baseline_action)
        lookup = self._lookup(public_entities)
        self._bind(inherited_decision, lookup)
        own = self._position(lookup, self.motor_id)
        target = self._position(lookup, self.target_id)
        predictions = self.learner.candidate_displacements(self.commands)
        supported = (inherited_decision['supported'] and own is not None and target is not None)
        ready = supported and self.learner.ready(predictions)
        action = inherited_action
        lead = 0.
        costs = None
        if ready:
            # Estimate target motion from its own previously observed coordinates.
            valid = [(t, p) for t, p in self.target_positions if p is not None]
            if len(valid) >= 2 and valid[-1][0] > valid[0][0]:
                velocity = (valid[-1][1] - valid[0][1]) / (valid[-1][0] - valid[0][0])
                horizon_seconds = self.learner.settings['horizon'] * self.learner.settings['nominal_seconds']
                lead = float(np.clip(velocity * horizon_seconds,
                    -self.learner.settings['target_lead_limit'], self.learner.settings['target_lead_limit']))
            costs = [(target + lead - own - prediction) ** 2
                + self.learner.settings['command_cost'] * drive ** 2
                for drive, prediction in zip((-1, 0, 1), predictions)]
            # Neutral wins exact ties; there is no test-phase or delay input.
            action = min((1, 0, 2), key=lambda a: costs[a])
            self.numeric_actions += 1
            self.changed_from_inherited += int(action != inherited_action)
        action, dither = self.learner.perturb(action, supported)
        self.pending_action = action
        # The inherited runtime history follows the actual requested command.
        self.inherited.pending_action = action
        self.latest_entities = public_entities
        return action, dict(mode='numeric_public_fir' if ready else 'inherited_relation',
            inherited_action=int(inherited_action), inherited_decision=inherited_decision,
            motor_id=self.motor_id, target_id=self.target_id, own=own, target=target,
            predictions=predictions, costs=costs, target_lead=lead,
            model_updates=self.learner.updates, ready=bool(ready), dither=bool(dither), action=int(action))

    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds, public_entities):
        if self.pending_action is None:
            raise ValueError('a chosen actual command is required')
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('positive actual public elapsed time required')
        self.inherited.observe(own_hits, terminated)
        self.learner.command_counts[self.pending_action] += 1
        self.commands.append(self.pending_action - 1)
        history = ([0.] * self.learner.settings['lags'] + list(self.commands))[-self.learner.settings['lags']:]
        feature = np.r_[1., history[::-1]] * elapsed_seconds
        self.features.append(feature)
        self.seconds += elapsed_seconds
        lookup = self._lookup(public_entities)
        self.positions.append(self._position(lookup, self.motor_id))
        self.times.append(self.seconds)
        self.target_positions.append((self.seconds, self._position(lookup, self.target_id)))
        self.last_update = None
        if (len(self.features) == self.learner.settings['window']
                and len(self.positions) == self.learner.settings['window'] + 1
                and self.positions[0] is not None and self.positions[-1] is not None):
            # Forgetting advances once per fresh observed transition, not once
            # per overlapping four-interval target duration.
            self.last_update = self.learner.observe(np.sum(self.features, axis=0),
                self.positions[-1] - self.positions[0], elapsed_seconds)
        self.pending_action = None
        self.latest_entities = public_entities
        return self.last_update
