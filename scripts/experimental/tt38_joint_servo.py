"""Use fresh camera measurements to test and, when needed, revise the remaining actuator references."""

import numpy as np
from scipy.stats import qmc

from .tt37_motor_clinic import AIM, landing_error


class JointServo:
    def __init__(self):
        self.notes = []
        self.pending = []
        self.memory = []

    def begin(self):
        self.pending.clear()
        self.notes = []

    def repay(self):
        self.memory.extend(self.pending)
        self.pending.clear()

    def inspect(self, engine, tail, eta, dt, tick, q):
        if not len(tail):
            return None
        t = engine.twin
        x = np.r_[eta, t.ball[[0, 2]], t.velocity, q[:5]]
        pos_scale = np.array([0.12, 0.17, 0.18, 0.19, 0.26])
        slope_scale = np.array([0.075, 0.12, 0.10, 0.115, 0.18])
        scale = np.r_[pos_scale, slope_scale]
        full = np.concatenate([tail, np.broadcast_to(tail[-1], (round(1.15 / dt), 6))])
        candidate = np.zeros((1, 9))
        _, s = engine.trajectories(full[None], AIM, eta + 0.16, feedback=candidate)
        learned = getattr(engine, "motor_value", None)
        valid = np.isfinite(s[0, 0])
        if valid:
            chance = learned.predict(s)[0, 0] if learned else 0.75
            if chance >= 0.51 and s[0, 5] > 0.022 and abs(s[0, 3]) < 0.70:
                self.notes.append({"tick": tick, "status": "kept", "trials": 1})
                return None
        # A changed camera prediction calls for alternative joint controls relative to
        # the actual ongoing command sequence. No simulator state is read from the game.
        memory = []
        if self.memory:
            distance = [
                np.linalg.norm(
                    (y - x) / [0.05, 0.19, 0.09, 0.9, 1.0, 1.2, 0.6, 0.7, 0.8, 0.7, 0.6]
                )
                for y, _ in self.memory
            ]
            memory = [self.memory[i][1] for i in np.argsort(distance)[:12]]
        sobol = 2 * qmc.Sobol(10, scramble=True, seed=38189 + tick).random_base2(5) - 1
        params = np.vstack([np.zeros(10), *memory, sobol * scale])
        elapsed = (np.arange(len(full)) + 1) * dt
        first = np.clip(elapsed / max(0.025, 0.40 * eta), 0.0, 1.0)
        second = np.clip((elapsed - 0.65 * eta) / max(0.04, 0.8 * eta), 0.0, 1.0)
        rng = np.random.default_rng(38781 + tick)
        used = 1
        for _ in range(3):
            goals = np.broadcast_to(full, (len(params), *full.shape)).copy()
            goals[:, :, :5] += (
                params[:, None, :5] * first[None, :, None]
                + params[:, None, 5:] * second[None, :, None]
            )
            f = np.zeros((len(goals), 9))
            _, r = engine.trajectories(goals, AIM, eta + 0.16, feedback=f)
            used += len(params)
            ok = np.flatnonzero(np.isfinite(r[:, 0]))
            if len(ok):
                c = learned.cost(r[ok]) if learned is not None else r[ok, 0]
                c += 0.24 * np.mean((params[ok] / scale) ** 2, axis=1)
                i = ok[np.argmin(c)]
                if np.linalg.norm(params[i]) > 1e-5:
                    self.pending.append((x, params[i].copy()))
                    self.notes.append(
                        {
                            "tick": tick,
                            "status": "corrected",
                            "trials": used,
                            "nominal": bool(valid),
                            "alternatives": len(ok),
                        }
                    )
                    return goals[i, : len(tail)]
                self.notes.append({"tick": tick, "status": "kept", "trials": used})
                return None
            loss = np.where(f[:, 1] > 0, landing_error(f), 10.0 + 20 * f[:, 0])
            elite = params[np.argsort(loss)[:6]]
            params = (
                elite[rng.integers(len(elite), size=32)]
                + rng.normal(0, 0.55, (32, 10)) * scale
            )
        self.notes.append({"tick": tick, "status": "no_solution", "trials": used})
        return None
