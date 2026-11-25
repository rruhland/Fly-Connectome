"""Closed-loop execution of direct joint choices using new optical information between planner decisions."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write

mpc = course.mpc


def corrected(goal, x, q, prediction, jac, remaining):
    # Screen-time innovation, not a predicted table-plane intercept. Correct
    # only while the sensed and predicted ball are on the same flight segment.
    if not np.isfinite(prediction).all() or x[4] > -0.9 or prediction[4] > -0.9:
        return goal
    if (x[5] > 0.5 and prediction[5] < -0.5) or (x[5] < -0.5 and prediction[5] > 0.5):
        return goal
    lag = np.clip(remaining, 0.0, 0.10)
    observed = x[:3] - prediction[:3]
    if np.linalg.norm(observed) > 0.075:
        return goal
    hand = jac @ (q[:5] - prediction[6:11])
    error = observed + lag * np.clip(x[3:6] - prediction[3:6], -0.65, 0.65) - hand
    shift = jac.T @ np.linalg.solve(jac @ jac.T + 0.030**2 * np.eye(3), error)
    if remaining < 0.035 or np.linalg.norm(error) < 0.0015:
        return goal
    cmd = goal.copy()
    cmd[:5] += 0.78 * np.clip(shift, -0.09, 0.09)
    return cmd


class Tracking(servo.Temporal):
    def begin(self):
        super().begin()
        self.prediction = None
        self.last_camera_time = -100.0

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        self.prediction = None
        if self.tick != tick or self.best is None or not self.notes:
            return
        note = self.notes[-1]
        if note.get("robust", 0) < 1 or note.get("contact_delay_s", 0.0) < 0.065:
            return
        # Nominal prediction belongs to the exact accepted action, before
        # beginning execution. Later observations cannot change that record.
        clip = self.best[: mpc.LEASE + 1].copy()
        trace = np.full((1, len(clip), 17), np.nan)
        self.engine.trajectories(
            clip[None, ...], mpc.AIM, len(clip) * mpc.DT, traces=trace
        )
        self.prediction = trace[0], note["contact_delay_s"]
        self.last_camera_time = self.engine.latest.time
        self.diag["tracked_actions"] += 1

    def step(self, tick, eye, x, q, reference, base):
        original = super().step(tick, eye, x, q, reference, base)
        age = tick - self.tick
        if (
            self.prediction is None
            or x is None
            or not 1 <= age < mpc.LEASE
            or self.engine.latest is None
        ):
            return original
        if self.engine.latest.time <= self.last_camera_time + 1e-7:
            return original
        self.last_camera_time = self.engine.latest.time
        trace, delay = self.prediction
        _, _, jac, _ = self.geometry.pose(q[:5], jacobian=True)
        actual = corrected(original, x, q, trace[age], jac, delay - age * mpc.DT)
        self.diag["visual_corrections"] += int(np.max(abs(actual - original)) > 1e-7)
        return actual


def run(a):
    with patch.object(mpc, "MPC", Tracking):
        course.run(a)
    record = _read(a.out)
    record["tracking_sha256"] = _hash(Path(__file__))
    _write(a.out, record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="assessment",
        mode="frozen",
        cap=240,
        carry=course.ROOT / "learning-12.json.gz",
        resume=False,
    )
    run(p.parse_args())
