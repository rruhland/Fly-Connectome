"""Test model-based transfer on a separate plant initialized exclusively from optical and encoder readings."""

from copy import copy

import mujoco
import numpy as np

from .tt29_rally_physics import NOMINAL_NO_LOAD
from .tt29_vision_twin import OpticalPlanner
from .tt37_opponent import arena


class OpticalClinic(OpticalPlanner):
    def __init__(self, *, body=False, robust=False):
        # This independently constructed nominal arena never receives a serve, a card or a live MjData.
        super().__init__(arena(), seed=371840)
        self.predict = self.nominal.predict
        self.robust = robust
        self.body_model = copy(self.twin.model) if body else None
        if body:
            ball = self.body_model.geom("ball")
            ball.contype[0] = ball.conaffinity[0] = 0
            self.body_state = mujoco.MjData(self.body_model)
            self.limits = self.body_model.actuator_forcerange[self.actuators].copy()
        self.previous = None
        self.rate = None
        self.latest = None
        self.tracker = None

    def observe(self, tick, tracker, q, reference):
        if self.body_model is not None:
            self.propagate(tick, q, reference)
        if self.previous is None:
            self.rate = np.zeros_like(q)
        else:
            self.rate = (q - self.previous) / self.twin.control_dt
        self.previous = q.copy()
        self.tracker = tracker
        self.latest = tracker.latest
        self.tick = tick
        self.reference = reference
        if (
            self.latest is not None
            and abs(tick * self.twin.control_dt - self.latest.time) < 0.08
        ):
            p, v = self.latest.advance(tick * self.twin.control_dt)
            if v[1] > 0.9:
                self.bounced = False
            elif v[1] < -0.9 and -0.01 > p[1] and (self.latest.bounced or v[2] > 0.55):
                self.bounced = True

    def propagate(self, tick, q, reference):
        t = self.twin
        d = self.body_state
        if tick == 0:
            # Nominal calibrated starting pose plus actual joint encoder readings.
            t.reset(seed=0, render=False)
            mujoco.mj_copyData(d, self.body_model, t.data)
            d.qpos[self.positions] = q
            d.ctrl[self.actuators] = reference
            mujoco.mj_forward(self.body_model, d)
            return
        micro = self.predict.micro
        increment = (reference - d.ctrl[self.actuators]) / micro
        for _ in range(micro):
            d.ctrl[self.actuators] = np.clip(
                d.ctrl[self.actuators] + increment,
                t._joint_low[self.actuators],
                t._joint_high[self.actuators],
            )
            speed = d.qvel[self.dofs] / NOMINAL_NO_LOAD
            self.body_model.actuator_forcerange[self.actuators, 0] = self.limits[
                :, 0
            ] * np.clip(1 + speed, 0, 1)
            self.body_model.actuator_forcerange[self.actuators, 1] = self.limits[
                :, 1
            ] * np.clip(1 - speed, 0, 1)
            mujoco.mj_step(self.body_model, d)
        # Observed encoders tether the model through small unmeasured paddle impulses.
        err = q - d.qpos[self.positions]
        d.qpos[self.positions] += 0.40 * err
        d.qvel[self.dofs] += 0.12 * err / t.control_dt

    def sync(self):
        if self.latest is None:
            return False
        self._refresh(self.tick, self.latest, self.previous, self.rate, self.reference)
        if self.body_model is not None:
            a = self.twin
            b = self.body_state
            d = a.data
            d.qpos[self.positions] = b.qpos[self.positions]
            d.qvel[self.dofs] = b.qvel[self.dofs]
            d.qpos[self.paddle_q : self.paddle_q + 7] = b.qpos[
                self.paddle_q : self.paddle_q + 7
            ]
            d.qvel[self.paddle_v : self.paddle_v + 6] = b.qvel[
                self.paddle_v : self.paddle_v + 6
            ]
            mujoco.mj_forward(a.model, d)
        return True

    def select(self, good, measurement, fit, eta, dt, home, rng):
        from .tt37_motor_clinic import AIM, landing_error, schedules

        # Refine locally around physically viable motor modes. Score them again under
        # alternative camera errors before issuing one movement to the real arm.
        n = 48
        good = good[np.argsort(measurement[:, 0])[:8]]
        proposals = (
            good[rng.integers(len(good), size=n)]
            + rng.normal(0, 1, (n, 10))
            * np.r_[[0.038, 0.040, 0.046, 0.048, 0.055], [0.23, 0.31, 0.27, 0.32, 0.39]]
        )
        actions = np.vstack([good, proposals])
        goals = schedules(*fit, eta, actions, dt, home, round((eta + 1.05) / dt))
        t = self.twin
        p, v = t.ball.copy(), t.velocity.copy()
        angular = t.data.qvel[t._ball_dof + 3 : t._ball_dof + 6]
        initial_spin = angular.copy()
        quality = []
        traces = []
        nominal = None
        contacts = 0
        legal = 0
        try:
            for dx, dz, dvy, dvz, spin in [
                (0, 0, 0, 0, 0),
                (0.001, 0.001, 0.03, 0.09, 23),
                (-0.001, -0.001, -0.03, -0.09, 38),
                (0.001, -0.001, 0.06, 0.17, 44),
                (-0.001, 0.001, -0.06, -0.17, 10),
            ]:
                t.ball[:] = p + [dx, 0, dz]
                t.velocity[:] = v + [0, dvy, dvz]
                angular[:] = initial_spin + [spin, 0, 0]
                f = np.zeros((len(actions), 9))
                _, y = self.predict.trajectories(goals, AIM, eta + 0.12, feedback=f)
                okay = np.isfinite(y[:, 0])
                if nominal is None:
                    nominal = y.copy()
                legal += int(okay.sum())
                contacts += int(f[:, 1].sum())
                score = np.where(
                    okay,
                    1.0 + 0.10 * np.exp(-0.45 * np.nan_to_num(y[:, 0], nan=100)),
                    0.16 * f[:, 1] * np.exp(-landing_error(f)),
                )
                quality.append(score)
                traces.append(okay)
        finally:
            t.ball[:] = p
            t.velocity[:] = v
            angular[:] = initial_spin
        quality = np.asarray(quality)
        allowed = np.flatnonzero(np.isfinite(nominal[:, 0]))
        winner = allowed[np.argmax(np.mean(quality[:, allowed], axis=0))]
        return (
            actions[winner],
            nominal[winner].tolist(),
            {
                "trials": len(actions) * len(quality),
                "contacts": contacts,
                "legal": legal,
                "robust_settings": int(np.asarray(traces)[:, winner].sum()),
            },
        )

    def trajectories(self, goals, target, cutoff, *, feedback=None, traces=None):
        return self.predict.trajectories(
            goals, target, cutoff, feedback=feedback, traces=traces
        )
