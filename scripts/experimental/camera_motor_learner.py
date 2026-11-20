"""Online identification of delayed image motion from issued motor commands."""

from dataclasses import dataclass

import numpy as np

from camera_reaching_arena import MAX_COMMAND_DELAY


CALIBRATION_TICKS = 14
PIXEL_DEADBAND = 1.9
DECAY = .91
RIDGE = .14


@dataclass
class MotorModel:
    """Discounted least squares for one integer action delay."""

    delay: int
    gram: np.ndarray
    cross: np.ndarray
    displacement_energy: float = 0.

    @classmethod
    def empty(cls, dimension, delay):
        return cls(delay, np.zeros((dimension, dimension)), np.zeros((dimension, 2)))

    def observe(self, command, displacement, *, discount=1.):
        self.gram = discount * self.gram + np.outer(command, command)
        self.cross = discount * self.cross + np.outer(command, displacement)
        self.displacement_energy = discount * self.displacement_energy + displacement @ displacement

    def effects(self):
        return np.linalg.solve(self.gram + RIDGE * np.eye(len(self.gram)), self.cross).T

    def residual(self):
        return self.displacement_energy - np.sum(self.cross.T * self.effects())


class CameraMotorLearner:
    """Regress local image motion on prior commands, and invert it for reaching.

    Initial exploration selects the action delay whose regularized regression
    explains the same observed displacements best. Reaching updates one model
    on each visible transition. Repeated large prediction errors or stalled
    progress restart identification without knowing whether the hardware moved.
    """

    def __init__(self, dimension, *, model=None, frozen=False):
        self.dimension = dimension
        self.model = model
        self.last_model = model
        self.frozen = frozen

    def begin_trial(self, seed):
        self._rng = np.random.default_rng(seed)
        self.commands = []
        self.probe_ticks = []
        self.adaptations = []
        self.calibrations = []
        self._first_usable = 0
        self._surprises = 0
        self.begin_goal()
        if self.model is None:
            self._begin_calibration()

    def begin_goal(self):
        self._best_distance = np.inf
        self._progress_tick = len(self.commands)

    def _begin_calibration(self):
        self.model = None
        self._first_usable = len(self.commands)
        self._alternatives = [MotorModel.empty(self.dimension, delay) for delay in range(MAX_COMMAND_DELAY + 1)]
        # Mix signs across channels while balancing total commands over the burst.
        flips = np.repeat((-.68, .68), CALIBRATION_TICKS // 2)
        self._probes = np.column_stack([self._rng.permutation(flips) for _ in range(self.dimension)])
        self._surprises = 0

    def act(self, error):
        tick = len(self.commands)
        if error is None:
            command = np.zeros(self.dimension)
        else:
            distance = np.linalg.norm(error)
            if distance < self._best_distance - .45:
                self._best_distance = distance
                self._progress_tick = tick
            if self.model is not None and not self.frozen and distance > 4.0 and tick - self._progress_tick >= 12:
                self.adaptations.append({'tick': tick, 'trigger': 'stalled image progress'})
                self._begin_calibration()
            if self.model is None:
                index = tick - self._first_usable
                if index >= CALIBRATION_TICKS:
                    # Missing images can interrupt exploration. Start a fresh paid burst.
                    self._begin_calibration()
                    index = 0
                command = self._probes[index]
                self.probe_ticks.append(tick + 1)
            else:
                command = self._steer(error)
        self.commands.append(command)
        return command

    def _steer(self, error):
        effects = self.model.effects()
        if self.model.delay and self.commands:
            pending = np.sum(self.commands[-self.model.delay:], axis=0)
            error = error - effects @ pending
        if np.linalg.norm(error) <= PIXEL_DEADBAND:
            return np.zeros(self.dimension)
        normal = effects @ effects.T + .31 * np.eye(2)
        command = effects.T @ np.linalg.solve(normal, .75 * error)
        return command / max(1., np.max(np.abs(command)))

    def observe(self, displacement):
        if displacement is None or self.frozen:
            return
        command_tick = len(self.commands) - 1
        if self.model is None:
            calibration_tick = command_tick - self._first_usable
            if calibration_tick >= MAX_COMMAND_DELAY:
                for hypothesis in self._alternatives:
                    hypothesis.observe(self.commands[command_tick - hypothesis.delay], displacement)
            if calibration_tick >= CALIBRATION_TICKS - 1:
                self.model = min(self._alternatives, key=MotorModel.residual)
                self.last_model = self.model
                self.calibrations.append({'tick': command_tick + 1, 'delay': self.model.delay})
                self.begin_goal()
            return
        issued = command_tick - self.model.delay
        if issued < 0:
            return
        command = self.commands[issued]
        expected = self.model.effects() @ command
        motion = np.linalg.norm(expected)
        if motion >= 1.05:
            error = np.linalg.norm(displacement - expected)
            contradicted = error > 1.35 and expected @ displacement < .30 * motion * motion
            self._surprises = self._surprises + 1 if contradicted else 0
            if self._surprises >= 2:
                self.adaptations.append({'tick': command_tick + 1, 'trigger': 'image prediction errors'})
                self._begin_calibration()
                return
            if self._surprises:
                # Keep the predictor for one more command to test the discrepancy.
                return
        self.model.observe(command, displacement, discount=DECAY)
