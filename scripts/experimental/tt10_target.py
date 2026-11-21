"""A two-view cursor task driven by the physical arm from the table-tennis hall."""
import numpy as np

from .tt4_arena import SOLVER_SUBSTEPS, mujoco
from .tt6_body_arena import TT6Arena
from .tt6_body_control import MOTOR_BUDGET


SIZE = 192
SCALE = 100.
BLOCK = 8
PULSE = 2
TOLERANCE = .045
BLOCKS_PER_TARGET = 3


class TargetArena(TT6Arena):
    """Run the same joint servos and contacts with the game ball parked off court."""

    def reset(self, *, seed=None, render=True):
        super().reset(seed=seed, render=False)
        self.ball[:] = (1.8, 1.8, .05)
        self.velocity[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.origin = self.paddle.copy()
        self.target = self.origin.copy()
        self.streak = 0
        self.scored = False
        return self.frame() if render else None

    def set_target(self, point):
        self.target = np.asarray(point)
        self.streak = 0
        self.scored = False

    def step(self, action, *, render=True):
        increments = np.zeros(self.model.nu)
        ids = self._arm_ids[1][2]
        increments[ids] = np.clip(action, -1., 1.) * self._joint_increments[ids]
        for _ in range(self.frameskip * SOLVER_SUBSTEPS):
            np.clip(self.data.ctrl + increments, self._joint_low, self._joint_high, out=self.data.ctrl)
            mujoco.mj_step(self.model, self.data)
        distance = np.linalg.norm(self.paddle - self.target)
        self.streak = self.streak + 1 if distance <= TOLERANCE else 0
        reward = int(self.streak >= 2 and not self.scored)
        self.scored |= bool(reward)
        return self.frame() if render else None, reward, False, False, {}

    def frame(self):
        # The game displays orthographic XY and XZ cursors. Each color channel
        # is painted independently so that coincidence preserves both markers.
        rgb = np.zeros((2, SIZE, SIZE, 3), dtype=np.uint8)
        for channel, obj in enumerate((self.paddle, self.target)):
            pos = np.rint((obj - self.origin) * SCALE).astype(int)
            x, y, z = np.clip(pos, -SIZE//2+2, SIZE//2-3) + SIZE//2
            for i, row in enumerate((y, z)):
                rgb[i, row-1:row+2, x-1:x+2, channel] = 240
        return rgb


def read_markers(rgb):
    result = []
    for channel in (0, 1):
        xy = np.nonzero(rgb[0, :, :, channel])
        xz = np.nonzero(rgb[1, :, :, channel])
        result.append((np.array([xy[1].mean(), xy[0].mean(), xz[0].mean()]) - SIZE//2) / SCALE)
    return result


class CursorPlayer:
    """Learn a local action-to-cursor map from motor pulses and RGB motion."""
    def __init__(self, prior, *, update=True, exploration='scheduled'):
        self.channels = prior.shape[1]
        self.penalty = 1.5
        self.covariance = np.eye(self.channels) * self.penalty
        self.cross = self.penalty * prior.T.copy()
        self.update = update
        self.exploration = exploration
        self.prior = prior.copy()
        self.probes = 0
        self.conflicts = 0
        self.discarded = False
        self.count = 0
        self.command = np.zeros(self.channels)
        self.requested = self.command.copy()
        self.previous = None

    def act(self, rgb):
        pos, goal = read_markers(rgb)
        if self.previous is not None:
            movement = pos - self.previous
            predicted = self.prior @ self.command
            if self.update and self.exploration == 'evidence' and not self.discarded and np.linalg.norm(predicted) > .02:
                conflict = np.linalg.norm(predicted - movement) > .025 + np.linalg.norm(movement)
                self.conflicts += int(conflict)
                if self.conflicts >= 2:
                    self.cross -= self.penalty * self.prior.T
                    self.discarded = True
            self.covariance += np.outer(self.command, self.command)
            if self.update:
                self.cross += np.outer(self.command, movement)
        matrix = np.linalg.solve(self.covariance, self.cross).T if self.update else self.prior
        # Each signed command unit moves the target actuator for one frame.
        # Work in that same unit with the source Jacobian and target observations.
        command = matrix.T @ np.linalg.solve(matrix @ matrix.T + .014**2 * np.eye(3), goal - pos)
        if self.exploration == 'evidence':
            offset = goal - pos
            feasible = np.clip(command, -PULSE, PULSE)
            residual = np.linalg.norm(offset - matrix @ feasible)
            if residual > max(TOLERANCE, .68 * np.linalg.norm(offset)):
                axis = np.argmax(np.diag(np.linalg.inv(self.covariance)))
                command = .35 * feasible
                direction = 1. if matrix[:, axis] @ offset >= 0. else -1.
                command[axis] = -direction * 1.6 if direction * self.requested[axis] > 2.5 else direction * 1.6
                self.probes += 1
        elif self.count < self.channels:
            probe = np.zeros(self.channels)
            probe[self.count] = 1.6
            command = probe if self.exploration == 'probe' else .65 * command + probe
            self.probes += 1
        lower = np.maximum(-PULSE, -MOTOR_BUDGET - self.requested)
        upper = np.minimum(PULSE, MOTOR_BUDGET - self.requested)
        self.command = np.clip(command, lower, upper)
        self.requested += self.command
        self.previous = pos
        self.count += 1
        return self.command / PULSE
