"""Update early direct joint commands with task-space errors observed on physically missed and scored attempts."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from . import tt44_rally_course as fresh
from . import tt45_context_feedback as feedback
from .tt4_transfer_trial import _hash, _read, _write

SCALE = np.array(
    [
        0.20,
        0.4,
        0.18,
        0.9,
        1.5,
        1.5,
        0.60,
        0.58,
        0.55,
        0.65,
        0.65,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
    ]
)


def dis(context, prior):
    scaled = (np.asarray(prior) - context) / SCALE
    return np.sqrt(np.mean(np.minimum(scaled**2, 16.0), axis=-1))


def sense(archive, start, geometry):
    seq = [(t, x, q) for t, x, q, *_ in archive if t >= start and x is not None]
    nears = [
        (t, x, q, geometry.pose(q[:5])[0])
        for t, x, q in seq
        if x[4] < -0.9 and x[1] < -0.66
    ]
    where = None
    for a, b in zip(nears, nears[1:]):
        if b[0] - a[0] > 2:
            continue
        before, after = a[1][:3] - a[3], b[1][:3] - b[3]
        change = after - before
        u = np.clip(-before @ change / (1e-8 + change @ change), 0.0, 1.0)
        residual = before + u * change
        magnitude = np.linalg.norm(residual)
        if where is None or magnitude < where[0]:
            where = (
                magnitude,
                a[0] + u * (b[0] - a[0]),
                a[2][:5] + u * (b[2][:5] - a[2][:5]),
                residual,
            )
    if where is None or where[0] > 0.40:
        return None
    _, time, joints, difference = where
    incoming = [x[3:6] for t, x, _ in seq if time - 8 <= t <= time and x[4] < -0.9]
    outgoing = [
        x[3:6] + np.array([0.0, 0.0, 9.81]) * (t - time) / 60.0
        for t, x, _ in seq
        if time + 1 <= t <= time + 12 and x[4] > 0.9
    ]
    impulse = None
    if incoming and len(outgoing) >= 2:
        diff = np.median(outgoing, axis=0) - np.median(incoming, axis=0)
        if np.linalg.norm(diff) > 1.0:
            impulse = diff / np.linalg.norm(diff)
    return joints, difference, impulse


class Learning:
    def __init__(self):
        self.success = deque(maxlen=300)
        self.lessons = deque(maxlen=120)

    def apply(self, key):
        if not self.lessons:
            return np.zeros(5)
        d = dis(key, [r[0] for r in self.lessons])
        ids = np.argsort(d)[:5]
        weights = np.exp(-2.8 * d[ids] ** 2)
        return np.clip(
            (weights @ np.array([self.lessons[i][1] for i in ids]))
            / (1.0 + weights.sum()),
            -0.16,
            0.16,
        )

    def update(self, key, achieved, shift, won, geometry):
        q, gap, impulse = achieved
        if won:
            self.success.append(
                (key.copy(), gap.copy(), None if impulse is None else impulse.copy())
            )
            if np.linalg.norm(shift) > 0.015:
                self.lessons.append((key.copy(), shift.copy()))
            return "scored"
        if not self.success:
            return "no_scored_example"
        distances = dis(key, [r[0] for r in self.success])
        near = np.argsort(distances)[:8]
        near = near[distances[near] < 1.25]
        if not len(near):
            return "unfamiliar"
        _, normal, linear, angular = geometry.pose(q, jacobian=True)
        target = np.median([self.success[i][1] for i in near], axis=0)
        jac = linear
        residual = np.clip(gap - target, -0.13, 0.13)
        comparison = [
            self.success[i][2] for i in near if self.success[i][2] is not None
        ]
        if impulse is not None and comparison:
            wanted = np.median(comparison, axis=0)
            direction = wanted / max(0.001, np.linalg.norm(wanted)) - impulse
            sign = 1.0 if normal @ impulse >= 0.0 else -1.0
            jac = np.vstack((linear, 0.11 * sign * angular))
            residual = np.r_[0.45 * residual, 0.11 * direction]
        change = np.linalg.solve(jac.T @ jac + 0.045**2 * np.eye(5), jac.T @ residual)
        self.lessons.append(
            (
                key.copy(),
                np.clip(shift + 0.80 * np.clip(change, -0.25, 0.25), -0.25, 0.25),
            )
        )
        return (
            "departure_error"
            if impulse is not None and comparison
            else "proximity_error"
        )


class Errors(servo.Temporal):
    def __init__(self, *args, learning, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning = learning
        self.priors.on_score = self.record

    def begin(self):
        super().begin()
        self.sensor = feedback.Approach()
        self.active = None
        self.credits = []

    def step(self, tick, eye, x, q, reference, base):
        key = self.sensor.observe(x, q)
        if key is not None:
            self.active = (tick, key, self.learning.apply(key))
        desired = super().step(tick, eye, x, q, reference, base)
        if (
            self.active is not None
            and x is not None
            and x[4] < -0.9
            and -0.46 < x[1] < 1.35
        ):
            desired = desired.copy()
            desired[:5] += self.active[2]
            self.diag["error_corrected_ticks"] += int(
                np.linalg.norm(self.active[2]) > 0.015
            )
        return desired

    def record(self, tape, won):
        if self.active is None:
            return
        start, key, shift = self.active
        self.active = None
        achieved = sense(tape, start, self.geometry)
        kind = (
            "unobserved"
            if achieved is None
            else self.learning.update(key, achieved, shift, won, self.geometry)
        )
        self.credits.append(
            dict(
                start=start,
                kind=kind,
                won=bool(won),
                shift_rad=float(np.linalg.norm(shift)),
                distance_m=None
                if achieved is None
                else float(np.linalg.norm(achieved[1])),
            )
        )

    def missed(self, tape):
        self.record(tape, False)


class Memory(course.Carried):
    def keep(self, archive, leg):
        saved = super().keep(archive, leg)
        self.on_score(archive, True)
        return saved


def run(a):
    learning = Learning()
    ordinary = course.mpc.play

    def play(*args, **kwargs):
        row = ordinary(*args, **kwargs)
        row["outcome_errors"] = args[10].credits
        return row

    with (
        patch.object(course, "PLAN", fresh.PLAN),
        patch.object(course, "Carried", Memory),
        patch.object(
            course.mpc,
            "MPC",
            lambda *args, **kw: Errors(*args, learning=learning, **kw),
        ),
        patch.object(course.mpc, "play", play),
    ):
        course.run(a)
    result = _read(a.out)
    result.update(
        error_learning_sha256=_hash(Path(__file__)),
        correction_memory=[(k.tolist(), v.tolist()) for k, v in learning.lessons],
        credited_effects=[
            (k.tolist(), v.tolist(), None if w is None else w.tolist())
            for k, v, w in learning.success
        ],
    )
    _write(a.out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="assessment", mode="live", cap=240, carry=fresh.CARRY, resume=False
    )
    run(p.parse_args())
