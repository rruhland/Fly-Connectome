"""Select older or recently earned motor candidates using matched practice rewards."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_reuse
from . import tt41_practice_curriculum as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_motor_clinic import SCALE
from .tt42_motor_portfolio import EXPERTS, choices

MODEL = Path("artifacts/tt42/portfolio-practice.npz")
MOTOR_SCALE = np.array([0.20, 0.26, 0.32, 0.32, 0.42, 1.8, 2.4, 2.5, 2.5, 3.0])


def basis(state, action, feedback, nominal):
    legal = np.isfinite(nominal[:, 0])
    v = feedback[:, 2:5]
    p = feedback[:, 5:8]
    dt = np.clip(
        0.94
        * (v[:, 2] + np.sqrt(np.maximum(0, v[:, 2] ** 2 + 19.62 * (p[:, 2] - 0.78))))
        / 9.81,
        0,
        1.8,
    )
    land = p[:, :2] + v[:, :2] * dt[:, None]
    tnet = -p[:, 1] / np.maximum(0.6, 0.94 * v[:, 1])
    net = p[:, 2] + v[:, 2] * tnet - 4.905 * tnet**2 - 0.933
    land[legal] = nominal[legal, 3:5]
    net[legal] = nominal[legal, 5]
    e = np.clip(
        np.c_[
            legal,
            feedback[:, 1],
            feedback[:, 0] / 0.045,
            land[:, 0] / 0.45,
            (land[:, 1] - 0.68) / 0.55,
            net / 0.14,
            v[:, 0] / 1.5,
            (v[:, 1] - 3.5) / 1.4,
            (v[:, 2] - 2.0) / 1.5,
            feedback[:, 8] / 0.04,
        ],
        -4,
        4,
    )
    ac = np.clip(action / MOTOR_SCALE, -4, 4)
    context = (
        state - np.r_[0.12, 0, 1.05, 0, -3.6, 0, 0, -1.55, -1.5, 1.5, -0.7]
    ) / np.r_[0.065, SCALE]
    z = np.c_[e, ac]
    cross = np.clip(context, -3, 3)[:, None] * z[:, None, :]
    return np.c_[
        z,
        abs(e[:, 3:6]),
        np.maximum(e[:, 9] - 0.9, 0),
        0.23 * cross.reshape(len(z), -1),
    ]


class Selector:
    def __init__(self, file, exclude=None):
        with np.load(file) as z:
            groups = z["release"]
            self.ids = set(groups)
            xs = np.stack(
                [
                    basis(s, a, f, y)
                    for s, a, f, y in zip(
                        z["states"],
                        z["moves"],
                        z["feedback"],
                        z["nominal"],
                        strict=True,
                    )
                ]
            )
            y = z["reward"].astype(float)
        use = groups != exclude if exclude is not None else np.ones(len(groups), bool)
        phi = xs[use] - xs[use].mean(axis=1, keepdims=True)
        target = y[use] - y[use].mean(axis=1, keepdims=True)
        centered = phi.reshape(-1, phi.shape[-1])
        self.w = np.linalg.solve(
            centered.T @ centered + 28 * np.eye(centered.shape[-1]),
            centered.T @ target.ravel(),
        )

    def select(self, state, actions, picks, f, nominal):
        prior = picks[1]
        score = basis(state, actions, f, nominal) @ self.w
        valid = np.flatnonzero(np.isfinite(nominal[:, 0]))
        if not len(valid):
            return prior
        best = valid[np.argmax(score[valid])]
        # Require a sizable conditional advantage before changing a familiar motor response.
        return int(best) if score[best] > score[prior] + 0.12 else int(prior)


def validate(file, out):
    docs = []
    with np.load(file) as z:
        for g in sorted(set(z["release"])):
            m = Selector(file, g)
            for i in np.flatnonzero(z["release"] == g):
                r = z["reward"][i]
                js = z["picks"][i]
                new = m.select(
                    z["states"][i], z["moves"][i], js, z["feedback"][i], z["nominal"][i]
                )
                docs.append(
                    {
                        "release": int(g),
                        "tick": int(z["tick"][i]),
                        "previous": int(r[js[1]]),
                        "recent": int(r[js[2]]),
                        "original": int(r[js[0]]),
                        "picked": int(r[new]),
                        "changed": bool(new != js[1]),
                        "best_expert": int(r[js].any()),
                        "any": int(r.any()),
                    }
                )
    _write(
        out,
        {
            "source_sha256": _hash(file),
            "rows": docs,
            "totals": {
                k: sum(r[k] for r in docs)
                for k in [
                    "previous",
                    "recent",
                    "original",
                    "picked",
                    "changed",
                    "best_expert",
                    "any",
                ]
            },
        },
    )


def play(a):
    sel = Selector(a.file)
    cards = _read(course.PLAN)["assessment"][a.start : a.start + a.n]
    if sel.ids.intersection(c["id"] for c in cards):
        raise ValueError("Matched physical practice and assessment overlap")
    banks = [_read(p)["bank"] for p in EXPERTS]

    def propose(engine, bank, x, fit, eta, dt, home):
        u, _, idx, ok, f, y = choices(engine, banks, x, fit, dt, eta, home)
        j = sel.select(np.r_[eta, x], u, idx, f, y)
        receipt = {
            "trials": len(u),
            "legal": int(ok.sum()),
            "contacts": int((f[:, 1] > 0).sum()),
            "memory": len(bank),
            "changed": j != idx[1],
        }
        receipt["changed"] = bool(receipt["changed"])
        if not ok[j]:
            return None, None, receipt
        return u[j], y[j].tolist(), receipt

    with patch.object(tt37_reuse, "propose", propose):
        course.clinic("optical-fixed", a.out, "assessment", EXPERTS[1], a.start, a.n)
    d = _read(a.out)
    d["paired_motor_practice"] = _hash(a.file)
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("validate", "play"))
    p.add_argument("out", type=Path)
    p.add_argument("--file", type=Path, default=MODEL)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=24)
    a = p.parse_args()
    if a.mode == "validate":
        validate(a.file, a.out)
    else:
        play(a)
