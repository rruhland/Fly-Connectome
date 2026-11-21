"""Select which learned model to update from reward and visual motion discrepancies."""
from collections import deque

import numpy as np

from .tt6_body_control import MOTOR_BUDGET
from .tt11_pursuit_learning import LocalServo
from .tt14_signal_change import ChangePlayer
from .tt14_signal_search import marker


class MotorEvidence:
    """Compare short observed motions with motor geometries carried from table tennis."""
    def __init__(self, priors):
        self.models = np.asarray(priors)
        self.samples = deque(maxlen=32)
        self.suspect = 0

    def suggest(self, motor, position, active, tick):
        effective = motor.requested + (motor.curve - 1.) @ motor.history
        self.samples.append((effective.copy(), position))
        if tick % 4 or len(self.samples) < 24:
            return active
        commands, positions = map(np.asarray, zip(*self.samples))
        motion = positions[14:] - positions[6:-8]
        costs = []
        for lag in (0, 2, 4, 6):
            times = np.arange(14, len(positions)) - lag
            pulses = commands[times] - commands[times-8]
            predictions = np.einsum('moi,ti->mto', self.models, pulses)
            costs.append(np.linalg.norm(predictions-motion, axis=-1).mean(axis=1))
        errors = np.min(costs, axis=0)
        best = int(np.argmin(errors))
        supported = best != active and errors[best]+.009 < .72*errors[active]
        self.suspect = self.suspect + 1 if supported else 0
        if self.suspect >= 2:
            self.samples.clear()
            self.suspect = 0
            return best
        return active


class ContextPlayer(ChangePlayer):
    def __init__(self, priors, mode):
        super().__init__(priors[0], detect=True, probe=False)
        self.evidence = MotorEvidence(priors)
        self.active = 0
        self.mode = mode
        self.events = []

    def restart_motor(self, position=None):
        previous = self.motor
        self.motor = LocalServo(self.evidence.models[self.active], probe=False)
        # Motors remain live. Keep the budget and outstanding commands across a model switch.
        self.motor.requested[:] = previous.requested
        self.motor.history[:] = previous.history
        self.motor.tick = previous.tick
        if position is not None:
            # A different port basis changes the interpretation of accumulated
            # travel. Estimate occupied travel from the tool marker in the new basis.
            jac = self.evidence.models[self.active]
            self.motor.requested = np.clip(jac.T @ np.linalg.solve(jac @ jac.T + .008**2*np.eye(3), position),
                                            -MOTOR_BUDGET, MOTOR_BUDGET)

    def restart_surface(self):
        self.gram = np.eye(10)*.025
        self.cross = np.zeros(10)
        self.cross[4:7] = -.35*.025
        self.tick = 0
        self.suspect = 0
        self.errors.clear()

    def act(self, rgb, reward):
        tick = self.elapsed
        position = marker(rgb)
        predicted = self.evidence.suggest(self.motor, position, self.active, tick)
        motor_changed = self.mode != 'reward_only' and predicted != self.active
        if motor_changed:
            self.active = predicted
            self.restart_motor(position)
        count = len(self.switches)
        # The goal observer uses the strength conditional on the observed RGB position.
        action = super().act(rgb, reward)
        reward_changed = len(self.switches) != count
        if self.mode.startswith('reset_both'):
            if motor_changed and not reward_changed:
                self.restart_surface()
            if reward_changed and not motor_changed:
                self.restart_motor()
        if motor_changed or reward_changed:
            self.events.append(dict(tick=tick, motor=motor_changed, reward=reward_changed,
                                    selected=self.active))
        if self.mode.endswith('_probe'):
            # Balanced, small movements along a direction that separates the
            # recalled motor maps. Use only public commands and stored geometry.
            phase = tick % 59
            if phase < 6 and tick >= 59:
                disagreement = np.linalg.norm(self.evidence.models[0]-self.evidence.models[1],axis=0)
                axis = np.argsort(disagreement)[-(1+(tick//59)%2)]
                before = self.motor.requested - action
                pulse = np.zeros_like(action)
                pulse[axis] = .18 if phase < 3 else -.18
                new = np.clip(action + pulse, np.maximum(-.25,-MOTOR_BUDGET-before),
                              np.minimum(.25,MOTOR_BUDGET-before))
                self.motor.requested = before + new
                self.motor.history[0] = new
                action = new
        return action
