"""Align a frozen visual action-effect field to an unseen command interface."""

import numpy as np

from body_effect_learner import BodyLearner


_AXES = np.eye(2)
_ZERO = np.zeros(2)
_IDENTITY_ADAPTER = _AXES.ravel()
_INITIAL_COVARIANCE = np.eye(4)
_DECAY = .998


def visual_effects(model, position):
    """Recover the image-space drift and two linear command responses."""
    drift = model.predict_displacement(position, _ZERO)
    effects = np.column_stack([model.predict_displacement(position, axis) - drift for axis in _AXES])
    return effects, drift


def startup_commands(rng):
    """Probe each command axis in both directions with a randomized, balanced order."""
    order = rng.permutation(2)
    signs = rng.choice([-1., 1.], 2)
    outward = .9 * signs[:, None] * _AXES[order]
    return np.concatenate((outward, -outward))


def tracking_command(position, target, effects, drift, rng, *, observation_noise=.008):
    """Run the same damped image-space controller for all field estimators."""
    error = target - position
    regularized = effects @ effects.T + observation_noise**2 * _AXES
    command = effects.T @ np.linalg.solve(regularized, error - drift)
    command /= max(1., np.max(np.abs(command)))
    exploration = min(1., np.linalg.norm(error) / .08)
    return np.clip(command + exploration * rng.uniform(-.08, .08, 2), -1., 1.)


class BodyMotorAlignment:
    """Fit a linear command mixer while preserving the source's learned effect field.

    The unconstrained mixer lives inside the prediction, where it maps an actual
    requested command into virtual source coordinates. The physical command comes
    from the bounded tracking policy. Both the fitter and predictor use only
    decoded marker positions, the outward commands, and the source checkpoint.
    """

    def __init__(self, prior, *, spatial=True, observation_noise=.008):
        self.source = BodyLearner(prior=prior, updates_enabled=False,
                                  observation_noise=observation_noise)
        self.observation_noise = observation_noise
        self.spatial = spatial
        self.adapter = _IDENTITY_ADAPTER.reshape(2, 2).copy()
        self._covariance = _INITIAL_COVARIANCE.copy()
        self.updates = 0
        self.reset_episode()

    def reset_episode(self):
        self._anchor = None
        self.source.reset_episode()

    def _source_effects(self, position):
        effects, drift = visual_effects(self.source, position)
        if not self.spatial and self._anchor is not None:
            effects = self._anchor
        return effects, drift

    def visual_effects(self, position):
        effects, drift = self._source_effects(position)
        return effects @ self.adapter, drift

    def predict_displacement(self, position, action):
        effects, drift = self.visual_effects(position)
        return effects @ action + drift

    def observe(self, previous_position, actual_requested_action, next_position):
        previous_position = np.asarray(previous_position, dtype=float)
        next_position = np.asarray(next_position, dtype=float)
        command = np.asarray(actual_requested_action, dtype=float)
        effects, drift = self._source_effects(previous_position)
        self.source.observe(previous_position, command, next_position)
        if not self.spatial and self._anchor is None:
            self._anchor = effects.copy()
        # Both image components constrain the same action mixer on every tick.
        design = (effects[:, :, None] * command).reshape(2, 4) / self.observation_noise
        response = (next_position - previous_position - drift) / self.observation_noise
        cross = self._covariance @ design.T
        gain = np.linalg.solve(_AXES + design @ cross, cross.T).T
        adapter = self.adapter.ravel()
        adapter += gain @ (response - design @ adapter)
        posterior = self._covariance - gain @ cross.T
        self._covariance = _DECAY * posterior + (1. - _DECAY) * _INITIAL_COVARIANCE
        self.updates += 1
