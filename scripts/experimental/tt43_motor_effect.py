"""Recover a weak motor decision using local actuator responses and optically observed successful flight effects."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt38_joint_mpc import AIM, DT, HORIZON


def optical_pairs(path):
    with np.load(path) as bank:
        states, marks = bank["states"], bank["marks"]
    examples = []
    start = 0
    for episode in _read(path.with_suffix(".json"))["practice"]:
        end = start + episode["frames"]
        frames = marks[start:end]
        stamps = np.rint((frames[:, 0] * DT - frames[:, 1]) * 480).astype(int)
        for stamp in np.unique(stamps):
            mask = stamps == stamp
            effect = (
                mask
                & (frames[:, 1] > 0.09)
                & (frames[:, 1] < 0.20)
                & (states[start:end, 4] > 0.95)
            )
            before = mask & (frames[:, 1] < -0.015) & (frames[:, 1] > -0.10)
            if not effect.any() or not before.any():
                continue
            near = states[start:end][before][-1, :6]
            vel = states[start:end][effect, 3:6].copy()
            vel[:, 2] += 9.81 * frames[effect, 1]
            examples.append((near.copy(), np.median(vel, axis=0)))
        start = end
    return tuple(np.stack(x) for x in zip(*examples, strict=True))


def control_basis():
    t = np.arange(1, HORIZON + 1)
    shape = np.minimum(1, t / 5.0)
    ramp = shape * np.clip((t - 6.0) / 7.0, -1.0, 1.0)
    basis = np.zeros((8, HORIZON, 6))
    for i, scale in enumerate([0.18, 0.21, 0.17, 0.21, 0.26]):
        basis[i, :, i] = shape * scale
    for i, joint in enumerate([0, 2, 3]):
        basis[i + 5, :, joint] = 0.20 * ramp
    return basis


def residual(a, f, velocity):
    # Approach is a prediction by the controller's own forward model.
    normal = a[:, 7:10].copy()
    normal *= np.where(normal[:, 1:2] >= 0, 1.0, -1.0)
    gap = a[:, 1:4] - a[:, 4:7]
    tangential = gap - (gap * normal).sum(axis=1)[:, None] * normal
    reached = f[:, 1] > 0.5
    # After separation, match the cameras' observed outgoing velocity for nearby
    # rewarded strokes. For misses only the end-effector tracking error is used.
    flight = np.where(reached[:, None], f[:, 2:5] - velocity, 0.0)
    return np.c_[tangential / 0.060, flight / [1.4, 1.0, 1.1]], reached


class Response(course.mpc.MPC):
    def __init__(self, file, raw, **kwargs):
        super().__init__(file, raw, **kwargs)
        self.context, self.effects = optical_pairs(file)
        self.basis = control_basis()

    def trial(self, goals, *, long=False, traces=False):
        result = super().trial(goals, long=long, traces=traces)
        if getattr(self, "collecting", False) and not long and len(goals) >= 7:
            quality = result[0]
            for i in np.argsort(quality)[:3]:
                self.available.append((float(quality[i]), goals[i]))
        return result

    def model_effect(self, gs, target):
        a = np.empty((len(gs), 19))
        f = np.empty((len(gs), 9))
        self.engine.predict.trajectories(
            np.asarray(gs), AIM, HORIZON * DT, feedback=f, approaches=a
        )
        self.diag["effect_rollouts"] += len(gs)
        return residual(a, f, target)

    def search(self, tick, x, q, reference, base):
        self.collecting = True
        self.available = []
        try:
            super().search(tick, x, q, reference, base)
        finally:
            self.collecting = False
        if not self.available or not self.notes or self.notes[-1]["tick"] != tick:
            return
        note = self.notes[-1]
        if (
            note.get("robust", 0) >= 1
            or not self.engine.bounced
            or not -1.48 < x[1] < -0.66
        ):
            return
        scale = np.array([0.15, 0.45, 0.14, 0.75, 1.2, 1.2])
        neighbors = np.argsort(np.linalg.norm((self.context - x) / scale, axis=1))[:11]
        target = np.median(self.effects[neighbors], axis=0)
        incumbent = self.best if self.tick == tick and self.best is not None else None
        anchors = ([incumbent] if incumbent is not None else []) + [
            g for _, g in sorted(self.available, key=lambda m: m[0])[:2]
        ]
        candidates = []
        for anchor in anchors:
            step = 0.55
            moves = np.concatenate(
                ([anchor], anchor + self.basis * step, anchor - self.basis * step)
            )
            difference, hit = self.model_effect(moves, target)
            if not np.isfinite(difference[0]).all():
                continue
            jac = (difference[1:9] - difference[9:]) / (2 * step)
            if hit[0]:
                jac[~(hit[1:9] & hit[9:]), 3:] = 0.0
            else:
                jac[:, 3:] = 0.0
            jac = np.nan_to_num(jac, nan=0.0)
            update = np.linalg.solve(
                jac @ jac.T + 0.22 * np.eye(8), -jac @ difference[0]
            )
            shift = np.tensordot(np.clip(update, -1.7, 1.7), self.basis, axes=1)
            candidates.extend([anchor + amount * shift for amount in (0.55, 1.0, 1.45)])
            # Physical forks that improved their effect can also supply a
            # movement when two sides of a local linearization straddle contact.
            quality = np.linalg.norm(difference, axis=1) + 4.0 * (~hit)
            candidates.extend(moves[j] for j in np.argsort(quality)[:2])
        if not candidates:
            return
        loss, good, _, scores, _ = self.trial(candidates, long=True)
        valid = np.flatnonzero(good)
        if not len(valid):
            return
        short = valid[np.argsort(loss[valid])[:5]]
        count, risk = self.uncertainty([candidates[i] for i in short], loss[short])
        eligible = np.flatnonzero(count == 2)
        if len(eligible):
            j = eligible[np.argmin(risk[eligible])]
            selected = short[j]
            self.best = candidates[selected]
            self.tick, self.lease = tick, course.mpc.LEASE
            self.diag["effect_recoveries"] += 1
            note["effect_recovery"] = {
                "original": note.get("chosen"),
                "risk": float(risk[j]),
                "radius": float(scores[selected, 12]),
                "sensor_goal": target.tolist(),
            }


def run(a):
    with patch.object(course.mpc, "MPC", Response):
        course.run(a)
    d = _read(a.out)
    d["effect_method_sha256"] = _hash(Path(__file__))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--carry", type=Path, default=course.ROOT / "learning-12.json.gz")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(mode="frozen", deck="assessment", cap=240, resume=False)
    run(p.parse_args())
