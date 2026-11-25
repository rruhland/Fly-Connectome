"""Acquire approximate body reflection from sampled kinematics and reuse scored joint controls on the other wing."""

import argparse
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares

from . import tt42_learning_course as course
from . import tt45_composed_feedback as composed
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt47_symmetry")
FIT = ROOT / "calibration.json"
mpc = course.mpc
FLIP = np.array([-1.0, 1.0, 1.0])


def affine(q, sign, offset):
    return np.asarray(q) * sign + offset


def fit(out):
    with np.load(mpc.BANK) as bank:
        data = bank["states"][:, 6:11]
    indices = np.random.default_rng(470205).permutation(len(data))[:128]
    q = data[indices]
    a = mpc.arena()
    arm = mpc.BladeIK(a)
    try:
        poses = np.array([arm.pose(v) for v in q])
        want = poses * FLIP

        def residual(sign, offset, indices):
            ideal = affine(q[indices], sign, offset)
            qs = np.clip(ideal, arm.low, arm.high)
            out = np.array([arm.pose(v) for v in qs]) - want[indices]
            return np.concatenate(
                [
                    out[:, 0].ravel(),
                    0.10 * out[:, 1].ravel(),
                    0.1 * (ideal - qs).ravel(),
                ]
            )

        # A reflected serial arm reverses some joint axes; discover the signs
        # and angular offsets from geometry, without assuming which joints.
        ready = np.array(_read(mpc.BANK.with_suffix(".json"))["ready"])[:5]
        signs = [np.array(s) for s in product((-1.0, 1.0), repeat=5)]
        rank = sorted(
            range(len(signs)),
            key=lambda i: np.linalg.norm(
                residual(signs[i], ready * (1 - signs[i]), np.arange(24))
            ),
        )
        fits = []
        for i in rank[:4]:
            sign = signs[i]
            start = ready * (1 - sign)
            ans = least_squares(
                lambda off: residual(sign, off, np.arange(64)),
                start,
                max_nfev=24,
                xtol=1e-5,
            )
            # A reflection twice should return the original state. An
            # unsigned offset on a non-reversing joint violates this.
            offset = ans.x * (sign < 0)
            fits.append(
                (np.linalg.norm(residual(sign, offset, np.arange(64))), sign, offset)
            )
        _, sg, off = min(fits, key=lambda r: r[0])
        qs = np.clip(affine(q[64:], sg, off), arm.low, arm.high)
        got = np.array([arm.pose(v) for v in qs])
        distance = np.linalg.norm(got[:, 0] - want[64:, 0], axis=1)
        angle = np.arccos(np.clip((got[:, 1] * want[64:, 1]).sum(axis=1), -1, 1))
        out.parent.mkdir(exist_ok=True, parents=True)
        _write(
            out,
            dict(
                sign=sg.tolist(),
                offset=off.tolist(),
                sampled=128,
                calibration=64,
                held_position_quantiles_m=np.quantile(
                    distance, [0.5, 0.9, 0.99]
                ).tolist(),
                held_normal_quantiles_rad=np.quantile(angle, [0.5, 0.9, 0.99]).tolist(),
                bank_sha=_hash(mpc.BANK),
            ),
        )
    finally:
        a.close()


class Reflected(composed.Composed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        info = _read(FIT)
        self.sign = np.array(info["sign"])
        self.offset = np.array(info["offset"])

    def search(self, tick, x, q, ref, base):
        super().search(tick, x, q, ref, base)
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        note = self.notes[-1]
        if note.get("composed_issued") or note.get("robust", 0) >= 2:
            return
        xx = x.copy()
        xx[[0, 3]] *= -1
        rq = affine(q[:5], self.sign, self.offset)
        other = ref.copy()
        other[:5] = affine(other[:5], self.sign, self.offset)
        goals = []
        for seq, _ in self.priors.priors(xx, rq, other, 10):
            mapped = seq.copy()
            mapped[:, :5] = affine(seq[:, :5], self.sign, self.offset)
            goals.append(mapped)
        if not goals:
            return
        _, _, traces, _, _ = self.trial(goals, traces=True)
        for seq, tr in zip(goals.copy(), traces):
            correction = self.transport(tr, face=True)
            if correction is not None:
                goals.append(seq + correction)
        cost, leg, *_ = self.trial(goals, long=True)
        ids = np.flatnonzero(leg)
        note["mirrored_candidates"] = len(goals)
        note["mirrored_legal"] = len(ids)
        if not len(ids):
            return
        ids = ids[np.argsort(cost[ids])[:4]]
        votes, rank = self.uncertainty([goals[j] for j in ids], cost[ids])
        valid = np.flatnonzero(votes > note.get("robust", 0))
        if len(valid):
            k = valid[np.argmin(rank[valid] - 1.4 * votes[valid])]
            self.best = goals[ids[k]]
            self.tick, self.lease = tick, mpc.LEASE
            note["reflected_issued"] = int(votes[k])
            self.diag["reflected_issued"] += 1


def run(a):
    with patch.object(mpc, "MPC", Reflected):
        course.run(a)
    d = _read(a.out)
    d.update(body_reflection=_hash(FIT))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "play"))
    p.add_argument("out", type=Path)
    p.add_argument("--n", default=8, type=int)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="frozen",
        deck="assessment",
        carry=course.ROOT / "learning-12.json.gz",
        cap=240,
        resume=False,
    )
    a = p.parse_args()
    fit(a.out) if a.phase == "fit" else run(a)
