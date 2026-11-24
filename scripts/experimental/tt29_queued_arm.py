"""Forecast the robot's pose through outstanding commands before selecting its next movement."""

import mujoco
import numpy as np

from .tt29_vision_twin import OpticalPlanner


class QueuedArmPlanner(OpticalPlanner):
    def __init__(self, twin, history, *, seed):
        super().__init__(twin, seed=seed)
        self.history = history
        self.forecasts = 0

    def _refresh(self, step, flight, angles, rate, target):
        lag = self.history.lag
        history = list(self.history.targets)
        if not lag or len(history) <= lag:
            return super()._refresh(step, flight, angles, rate, target)
        station = self.twin
        model, data = station.model, station.data
        super()._refresh(step - lag, flight, angles, rate, history[-lag - 1])
        micro = round(station.control_dt / model.opt.timestep)
        # Each pending target was issued by the learner. The counts come from
        # encoders during prior practice; the live command queue is private.
        for next_target in history[-lag:]:
            previous = data.ctrl[self.actuators].copy()
            change = (next_target - previous) / micro
            for _ in range(micro):
                data.ctrl[self.actuators] += change
                station._aerodynamics()
                mujoco.mj_step(model, data)
                if station._motor_limits is not None:
                    model.actuator_forcerange[self.actuators] = station._motor_limits
                    station._motor_limits = None
        future = data.qpos[self.positions].copy()
        velocity = data.qvel[self.dofs].copy()
        super()._refresh(step, flight, future, velocity, target)
        self.forecasts += 1
