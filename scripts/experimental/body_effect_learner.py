"""Learn bounded motor commands from public image-space displacements."""

import numpy as np


_PRIOR_SCHEMA = 'visual-body-effects-v1'
_SPATIAL_VARIANCE = np.array([10., 1.5, 1.5, .5, .5, .5])
_MOTION_UNIT = .05
_FORGETTING = .998


def _vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return value


class BodyLearner:
    """Estimate local command effects and image-space momentum by recursive least squares.

    All positions are in a common normalized camera frame. The checkpoint contains
    fitted forward-model coefficients. A receiving body starts with independent
    covariance and empty motion history, then updates from its own transitions.
    """

    def __init__(self, action_dim=2, seed=0, *, prior=None, updates_enabled=True,
                 observation_noise=.008):
        if not isinstance(action_dim, int) or action_dim < 1:
            raise ValueError('action_dim must be a positive integer')
        if not np.isfinite(observation_noise) or observation_noise <= 0:
            raise ValueError('observation_noise must be positive and finite')
        self.action_dim = action_dim
        self.updates_enabled = updates_enabled
        self.observation_noise = observation_noise
        self.rng = np.random.default_rng(seed)

        self._motor_features = len(_SPATIAL_VARIANCE) * action_dim
        variance = np.r_[np.repeat(_SPATIAL_VARIANCE, action_dim), .5, .5]
        self._initial_covariance = np.diag(variance)
        self._covariance = self._initial_covariance.copy()
        self._weights = np.zeros((len(variance), 2))
        if prior is not None:
            self._load_prior(prior)

        self.updates = 0
        self.recalibrations = 0
        self._surprises = 0
        self._last_recalibration = 0
        self._action_axes = np.eye(action_dim)
        self._image_axes = np.eye(2)
        self.reset_episode()

    def _load_prior(self, prior):
        if (not isinstance(prior, dict) or prior.get('schema') != _PRIOR_SCHEMA
                or prior.get('action_dim') != self.action_dim):
            raise ValueError('incompatible visual body effects checkpoint')
        weights = np.asarray(prior.get('weights'), dtype=float)
        if weights.shape != self._weights.shape or not np.isfinite(weights).all():
            raise ValueError('checkpoint must contain finite visual effect weights')
        self._weights = weights.copy()

    def export_prior(self):
        """Export the learned visual forward model as JSON-compatible values."""
        return {'schema': _PRIOR_SCHEMA, 'action_dim': self.action_dim,
                'weights': self._weights.tolist()}

    def reset_episode(self):
        """Clear the previous image displacement when the scene is reset."""
        self._last_position = None
        self._last_motion = np.zeros(2)
        self._surprises = 0

    @staticmethod
    def _spatial_features(position):
        x, y = position
        return np.array([1., x, y, x * x, x * y, y * y])

    def _motion(self, position):
        if self._last_position is not None and np.array_equal(position, self._last_position):
            return self._last_motion / _MOTION_UNIT
        return np.zeros(2)

    def _command_model(self, position):
        spatial = self._spatial_features(position)
        weights = self._weights[:self._motor_features].reshape(-1, self.action_dim, 2)
        effects = np.einsum('s,sao->oa', spatial, weights)
        drift = self._motion(position) @ self._weights[self._motor_features:]
        return spatial, effects, drift

    def _command(self, value):
        command = _vector(value, self.action_dim, 'action')
        if np.any(np.abs(command) > 1):
            raise ValueError('action components must lie in [-1, 1]')
        return command

    def predict_displacement(self, position, action):
        """Predict the next image displacement for a candidate bounded action."""
        position = _vector(position, 2, 'position')
        _, effects, drift = self._command_model(position)
        return effects @ self._command(action) + drift

    def act(self, position, target):
        position = _vector(position, 2, 'position')
        target = _vector(target, 2, 'target')
        spatial, effects, drift = self._command_model(position)
        error = target - position
        model = effects @ effects.T + self.observation_noise**2 * self._image_axes
        action = effects.T @ np.linalg.solve(model, error - drift)
        action /= max(1., np.max(np.abs(action)))
        if self.updates_enabled:
            action += self._probe(spatial, effects, error)
        return np.clip(action, -1., 1.)

    def _probe(self, spatial, effects, error):
        # Estimate one-step uncertainty separately in each direction of command space.
        motor_covariance = self._covariance[:self._motor_features, :self._motor_features]
        command = (spatial[:, None, None] * self._action_axes).reshape(
            self._motor_features, self.action_dim)
        uncertainty = np.sqrt(np.maximum(0., np.sum(command * (motor_covariance @ command), axis=0)))
        displacement = np.linalg.norm(effects) / np.sqrt(self.action_dim)
        scale = self.observation_noise / max(displacement, self.observation_noise)
        margin = min(1., np.linalg.norm(error) / (2 * displacement + self.observation_noise))
        amplitude = np.clip(.4 * scale * uncertainty, .02, .8) * margin
        return amplitude * self.rng.uniform(-1., 1., self.action_dim)

    def observe(self, previous_position, actual_requested_action, next_position):
        """Fit a single observed transition and retain its displacement for the next tick."""
        before = _vector(previous_position, 2, 'previous position')
        after = _vector(next_position, 2, 'next position')
        command = self._command(actual_requested_action)
        displacement = after - before
        if self.updates_enabled:
            spatial = self._spatial_features(before)
            motor = (spatial[:, None] * command).ravel()
            features = np.r_[motor, self._motion(before)]
            self._update(features, displacement)
        self._last_position = after.copy()
        self._last_motion = displacement

    def _update(self, features, displacement):
        cross = self._covariance @ features
        projected_variance = features @ cross
        error = displacement - features @ self._weights
        self._weights += np.outer(cross, error) / (_FORGETTING + projected_variance)

        # Relax confidence gradually so repeated source behavior cannot prevent adaptation.
        self._covariance -= np.outer(cross, cross) / (_FORGETTING + projected_variance)
        self._covariance = (_FORGETTING * self._covariance
                            + (1. - _FORGETTING) * self._initial_covariance)
        self.updates += 1
        self._check_surprise(error, projected_variance)

    def _check_surprise(self, error, projected_variance):
        # A run of errors beyond the expected visual noise can signal changed controls.
        surprise = error @ error > 12 * self.observation_noise**2 * (1 + projected_variance)
        self._surprises = self._surprises + 1 if surprise else 0
        if self._surprises >= 2 and self.updates - self._last_recalibration >= 12:
            self._covariance = self._initial_covariance.copy()
            self._surprises = 0
            self._last_recalibration = self.updates
            self.recalibrations += 1
