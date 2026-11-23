"""Read a local SO-101 action-effect fit acquired during physical bouncing."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tt19_so101 import AXES, RATES
from .tt24_bounce import CONTROL_INTERVAL


OUTPUT = Path('artifacts/tt26')
CHECKPOINT = OUTPUT / 'bounce-effects.npz'
ORDINARY = OUTPUT / 'bounce-effects-ordinary.npz'
SCHEMA = 'tt26-bounce-effects-1'
JOINTS = AXES[:5]
STEP = np.asarray(RATES[:5]) * CONTROL_INTERVAL
AXIS_COUNT = len(JOINTS)


@dataclass(frozen=True)
class JointEffects:
    """Predict encoder changes from the body state and a normalized joint rate.

    Positions and requested servo targets are in radians. Velocities are in
    rad/s. The model expects the servo target before applying the new action.
    Each model step is 1/60 s. Leading array dimensions support batch inputs.
    """

    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    @staticmethod
    def _features(q, qvel, target, action, center, scale):
        q, qvel, target, action = np.broadcast_arrays(q, qvel, target, action)
        pose = (q - center) / scale
        bounded_pose = np.clip(pose, -2., 2.)
        # Fade the pose correction back to the shared response outside the
        # support of the source angles. The shared terms remain available.
        fade = 1. / (1. + np.maximum(np.abs(pose) - 2., 0.) ** 2)
        pose = bounded_pose * fade
        command = np.concatenate((qvel / 3., (target - q) / .09, action), axis=-1)
        interactions = (pose[..., :, None] * command[..., None, :]).reshape(
            *q.shape[:-1], AXIS_COUNT * command.shape[-1]
        )
        return np.concatenate((np.ones_like(q[..., :1]), pose, command, interactions), axis=-1)

    @classmethod
    def fit(cls, q, qvel, target, action, next_q, next_qvel):
        """Fit a ridge transition using issued targets and encoder samples only."""
        center = np.mean(q, axis=0)
        scale = np.maximum(np.std(q, axis=0), .075)
        features = cls._features(q, qvel, target, action, center, scale)
        observed = np.column_stack(((next_q - q) / CONTROL_INTERVAL, next_qvel))
        penalty = np.full(features.shape[1], .005)
        penalty[:1 + 4 * AXIS_COUNT] = .00004
        penalty[0] = .000001
        covariance = features.T @ features
        covariance.flat[::len(penalty) + 1] += len(q) * penalty
        coefficients = np.linalg.solve(covariance, features.T @ observed)
        return cls(center, scale, coefficients)

    def save(self, path, **metadata):
        """Save fitted tensors with axis order and the native control interval."""
        np.savez_compressed(path, schema=SCHEMA, axes=JOINTS, dt=CONTROL_INTERVAL,
                            center=self.center, scale=self.scale, coefficients=self.coefficients, **metadata)

    @classmethod
    def load(cls, path=CHECKPOINT):
        """Load a frozen stock-axis predictor without source simulation state."""
        with np.load(path, allow_pickle=False) as archive:
            if (archive['schema'].item() != SCHEMA or tuple(archive['axes']) != JOINTS
                    or archive['dt'].item() != CONTROL_INTERVAL):
                raise ValueError('incompatible SO-101 bounce-effect checkpoint')
            return cls(archive['center'], archive['scale'], archive['coefficients'])

    def advance(self, q, qvel, target, action, *, steps=1):
        """Predict position and velocity after holding the action for `steps`."""
        if steps < 1:
            raise ValueError('expected at least one 60 Hz model interval')
        q, qvel, target, action = map(np.asarray, (q, qvel, target, action))
        for _ in range(steps):
            outcome = self._features(q, qvel, target, action, self.center, self.scale) @ self.coefficients
            q = q + CONTROL_INTERVAL * outcome[..., :AXIS_COUNT]
            qvel = outcome[..., AXIS_COUNT:]
            target = target + STEP * action
        return q, qvel

    def predict(self, q, qvel, target, action, *, steps=1):
        """Predict attained angular displacement in radians."""
        return self.advance(q, qvel, target, action, steps=steps)[0] - q

    def linearize(self, q, qvel, target, *, steps=1):
        """Read displacement and its action Jacobian at zero action."""
        zero = np.zeros(AXIS_COUNT)
        perturbations = .01 * np.eye(AXIS_COUNT)
        baseline = self.predict(q, qvel, target, zero, steps=steps)
        positive = self.predict(q, qvel, target, perturbations, steps=steps)
        negative = self.predict(q, qvel, target, -perturbations, steps=steps)
        return baseline, ((positive - negative) / .02).T
