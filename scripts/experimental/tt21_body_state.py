"""Initialize reply rollouts from a command-driven, independent nominal arm."""

from dataclasses import dataclass

import numpy as np

from .tt4_arena import mujoco
from .tt19_catalog import Player
from .tt19_model_rollout import DrySimulator, RolloutPlayer
from .tt20_curriculum import model_arena


@dataclass(frozen=True)
class BodyState:
    physics: np.ndarray
    targets: np.ndarray


class CommandTwin:
    """Maintain only factory state and the player's own requested-rate history."""

    def __init__(self, nominal):
        self.arena = nominal
        self.model = nominal.model
        self.data = mujoco.MjData(self.model)
        self.flags = mujoco.mjtState.mjSTATE_INTEGRATION
        self.physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.axes = nominal._arm_ids[1][2]
        self.micro = int(round(nominal.control_dt / self.model.opt.timestep))
        self.inc = nominal._joint_increments[self.axes]
        self.low, self.high = nominal._joint_low[self.axes], nominal._joint_high[self.axes]
        self._reset_buffer = np.empty(mujoco.mj_stateSize(self.model, self.flags))
        self.park = np.array((1.70, 2.0, .6))
        self.reset()

    def reset(self):
        # The beginning of a fresh point is a factory initialization, not a
        # copy of an evaluator's arm, ball, servo or solver state.
        self.arena.reset(seed=84105173, server=-1, render=False)
        self.arena.ball[:] = self.park
        ball = self.arena._ball_dof
        self.arena.data.qvel[ball:ball + 6] = 0.
        mujoco.mj_forward(self.model, self.arena.data)
        mujoco.mj_getState(self.model, self.arena.data, self._reset_buffer, self.flags)
        mujoco.mj_setState(self.model, self.data, self._reset_buffer, self.flags)
        mujoco.mj_forward(self.model, self.data)

    def advance(self, command):
        increment = self.inc * np.clip(command, -1., 1.)
        for _ in range(self.micro):
            self.data.ctrl[self.axes] = np.clip(self.data.ctrl[self.axes] + increment, self.low, self.high)
            mujoco.mj_step(self.model, self.data)

    def state(self):
        physics = np.empty(mujoco.mj_stateSize(self.model, self.physics))
        mujoco.mj_getState(self.model, self.data, physics, self.physics)
        return BodyState(physics, self.data.ctrl.copy())


class StateInitializedDrySimulator(DrySimulator):
    """Change candidate start state and targets, preserving the exact rollout ranker."""

    def __init__(self, law):
        super().__init__(law, arena=model_arena())
        self.current = None
        self.twin = CommandTwin(self.arena)

    def predict(self, tape, duration, observation, offsets):
        nominal_state, nominal_ctrl = self.base, self.home
        if self.current is not None:
            self.base, self.home = self.current.physics, self.current.targets
        try:
            return super().predict(tape, duration, observation, offsets)
        finally:
            self.base, self.home = nominal_state, nominal_ctrl


class ReplyRanker(RolloutPlayer):
    """Run matching nominal candidate lists and durations with two body starts."""

    def __init__(self, catalog, law, response, *, count=32):
        Player.__init__(self, catalog, law, response, 'late')
        self.sim = StateInitializedDrySimulator(law)
        self.count = count
        self.evaluated = 0

    def rank_at(self, history, estimate, body_state=None):
        self.history = history
        self.sim.current = body_state
        previous = self.evaluated
        try:
            decision = self.choose(estimate)
        finally:
            self.sim.current = None
        return decision, self.evaluated - previous

    def advance(self, command):
        self.sim.twin.advance(command)

    def reset_body(self):
        self.sim.twin.reset()
