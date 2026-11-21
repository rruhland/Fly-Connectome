"""Control blade velocity during the predicted collision window."""
import numpy as np

from .tt6_continuous import PredictiveServo
from .tt7_flight import FlightPlayer


class SwingServo(PredictiveServo):
    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        if self.intercept is None or not -.5 < self.intercept < self.horizon + 1.:
            return
        # The collision may fall on either side of the estimated video frame.
        # Joint motion already in flight contributes to both differential poses.
        poses = self.fit.pose(states)
        displacement = np.diff(poses[:, 1:3], axis=0)
        response = np.einsum('an,tmn->tamn', jac[1:3], np.diff(effect, axis=0))
        times = np.arange(2, self.horizon + 1)
        weights = 3.5 * np.exp(-.5 * ((times - self.intercept - .5) / 1.25)**2)
        motion = (response * weights[:, None, None, None]).reshape(-1, rhs.size)
        target = np.array([self.stroke, .010])
        residual = ((target - displacement) * weights[:, None]).ravel()
        normal += motion.T @ motion
        rhs += motion.T @ residual


class SwingPlayer(FlightPlayer):
    def _make_servo(self):
        return SwingServo(self.adapter, self.timing_memory, learn=self.adjust)
