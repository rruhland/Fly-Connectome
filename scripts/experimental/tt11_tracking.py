"""Use an arm's learned motor directions under a change in visual coordinates."""
import numpy as np

from .tt6_body_control import MOTOR_BUDGET
from .tt10_target import PULSE, SIZE, SCALE, TargetArena, read_markers


class TrackingArena(TargetArena):
    """Two orthographic cursor displays viewed along a rotated set of axes."""
    def __init__(self, **kwargs):
        self.cursor_axes = np.eye(3)
        super().__init__(**kwargs)

    def frame(self):
        rgb = np.zeros((2, SIZE, SIZE, 3), dtype=np.uint8)
        for channel, obj in enumerate((self.paddle, self.target)):
            displacement = self.cursor_axes @ (obj - self.origin)
            x, y, z = np.clip(np.rint(displacement * SCALE).astype(int),
                              -SIZE//2+2, SIZE//2-3) + SIZE//2
            for i, row in enumerate((y, z)):
                rgb[i, row-1:row+2, x-1:x+2, channel] = 240
        return rgb


def orientation(cross):
    # Wahba's proper rotation fit. The last sign rules out a mirror transform.
    left, _, right = np.linalg.svd(cross)
    left[:, -1] *= np.linalg.det(left @ right)
    return left @ right


class PursuitPlayer:
    """Estimate the shared visual rotation and a local correction from camera motion."""
    def __init__(self, prior, *, align=True, probe_axes=None, lead=True, adapt=True, support=2.5):
        self.prior = prior.copy()
        self.align = align
        self.lead = lead
        self.support = support
        self.adapt = adapt
        self.previous = None
        self.previous_goal = None
        self.command = np.zeros(prior.shape[1])
        self.requested = self.command.copy()
        self.gram = np.zeros((len(self.command), len(self.command)))
        self.cross = np.zeros((3, len(self.command)))
        self.view_evidence = np.eye(3) * .00035
        self.probe_axes = list(probe_axes) if probe_axes is not None else []
        self.epoch = 0

    def act(self, rgb):
        pos, goal = read_markers(rgb)
        error = goal - pos
        if self.previous is not None:
            motion = pos - self.previous
            if self.lead:
                error += .8 * (goal - self.previous_goal)
            prediction = self.prior @ self.command
            if self.adapt:
                self.view_evidence += np.outer(motion, prediction)
                self.cross += np.outer(motion, self.command)
                self.gram += np.outer(self.command, self.command)
        prior = orientation(self.view_evidence) @ self.prior if self.align else self.prior
        jac = np.linalg.solve(self.gram + self.support * np.eye(len(self.command)), (self.cross + self.support * prior).T).T
        command = jac.T @ np.linalg.solve(jac @ jac.T + .014**2 * np.eye(3), error)
        if self.epoch < len(self.probe_axes):
            command[:] = 0.
            command[self.probe_axes[self.epoch]] = 1.6
        self.command = np.clip(command, np.maximum(-PULSE, -MOTOR_BUDGET-self.requested),
                               np.minimum(PULSE, MOTOR_BUDGET-self.requested))
        self.requested += self.command
        self.previous = pos
        self.previous_goal = goal
        self.epoch += 1
        return self.command / PULSE


def starting_probes(prior):
    first = int(np.argmax(np.linalg.norm(prior, axis=0)))
    second = int(np.argmax(np.linalg.norm(np.cross(prior[:, first], prior.T), axis=1)))
    return [first, second]
