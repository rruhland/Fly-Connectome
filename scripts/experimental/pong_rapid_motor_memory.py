"""Carry the inferred public action effects between fresh visual trajectories."""

from dataclasses import dataclass, replace
import math

import numpy as np

from pong_rapid_motor_recovery import ORDERS, RapidMotorController


SCHEMA = 'public-visual-motor-order-memory-v1'


@dataclass(frozen=True)
class RapidMotorMemory:
    """Checkpoint the discrete action belief independently of the current paddle."""

    order: tuple[int, int, int]
    probabilities: tuple[float, ...]
    updates_enabled: bool = True

    def __post_init__(self):
        if (not isinstance(self.order, tuple) or len(self.order) != 3
                or any(type(value) is not int for value in self.order)
                or self.order not in ORDERS):
            raise ValueError('a confirmed ordering of three distinct requests is required')
        if (not isinstance(self.probabilities, tuple) or len(self.probabilities) != len(ORDERS)
                or any(type(value) not in (float, int) for value in self.probabilities)
                or not all(math.isfinite(value) and value >= 0 for value in self.probabilities)
                or not math.isclose(sum(self.probabilities), 1., abs_tol=1e-12)):
            raise ValueError('a normalized probability for each possible ordering is required')
        if type(self.updates_enabled) is not bool:
            raise ValueError('the action belief update flag must be Boolean')

    @classmethod
    def from_learner(cls, learner):
        order = tuple(learner.request(intent) for intent in range(3))
        return cls(order, tuple(float(mode == order) for mode in ORDERS), learner.updates_enabled)

    def request(self, intent):
        return self.order[intent]

    def new_controller(self, policy):
        controller = RapidMotorController(policy, self)
        controller.belief.probabilities = np.asarray(self.probabilities, float).copy()
        return controller

    def capture(self, controller):
        if controller.pending is not None:
            raise ValueError('observe the completed request before capturing its action belief')
        if controller.adaptation_enabled != self.updates_enabled:
            raise ValueError('controller and memory update modes differ')
        if not self.updates_enabled:
            return self
        return replace(self, order=controller.belief.order,
                       probabilities=tuple(controller.belief.probabilities.tolist()))

    def state(self):
        return dict(schema=SCHEMA, order=list(self.order), probabilities=list(self.probabilities),
                    updates_enabled=self.updates_enabled)

    @classmethod
    def from_state(cls, state):
        keys = {'schema', 'order', 'probabilities', 'updates_enabled'}
        if not isinstance(state, dict) or state.keys() != keys or state['schema'] != SCHEMA:
            raise ValueError('unknown visual motor order checkpoint')
        if not isinstance(state['order'], list) or not isinstance(state['probabilities'], list):
            raise ValueError('checkpoint order and probabilities must be lists')
        return cls(tuple(state['order']), tuple(state['probabilities']), state['updates_enabled'])

    def frozen_branch(self):
        return replace(self, updates_enabled=False)
