"""Use real paired practice to learn how well the camera-initialized dynamics predict an action."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _write
from .tt37_motor_clinic import AIM
from .tt40_paired_practice import describe, motions, practice


def outcome_features(engine, cmds, eta):
    fb = np.zeros((len(cmds), 9))
    _, s = engine.trajectories(cmds, AIM, eta + 0.12, feedback=fb)
    legal = np.isfinite(s[:, 0])
    vel = fb[:, 2:5]
    p = fb[:, 5:8]
    dt = (
        0.94
        * (
            vel[:, 2]
            + np.sqrt(np.maximum(0.0, vel[:, 2] ** 2 + 19.62 * (p[:, 2] - 0.779)))
        )
        / 9.81
    )
    land = p[:, :2] + vel[:, :2] * dt[:, None]
    tn = -p[:, 1] / np.maximum(0.4, 0.94 * vel[:, 1])
    net = p[:, 2] + tn * vel[:, 2] - 4.905 * tn**2 - 0.933
    land[legal] = s[legal, 3:5]
    net[legal] = s[legal, 5]
    return np.c_[
        legal,
        fb[:, 1],
        np.clip(fb[:, 0], 0, 0.2),
        np.clip(land, -2.0, 2.0),
        np.clip(net, -0.4, 0.6),
        np.clip(vel, -9.0, 9.0),
        fb[:, 8],
    ]


def embedding(x, fx, changes):
    action = fx / np.array([1.1, 1.0, 0.065, 0.46, 0.45, 0.12, 1.8, 1.4, 1.2, 0.033])
    sense = x[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] / np.array(
        [0.11, 0.22, 0.057, 1.7, 1.6, 1.4, 1.4, 1.2, 1.3, 1.3, 1.3]
    )
    return np.c_[
        action,
        np.broadcast_to(0.65 * sense, (len(fx), len(sense))),
        0.4 * changes / 0.2,
    ]


class Ranker:
    def __init__(self, file, exclude=None):
        with np.load(file) as b:
            use = (
                np.ones(len(b["x"]), bool)
                if exclude is None
                else b["release"] != exclude
            )
            self.moves = b["moves"]
            self.x = np.concatenate(
                [
                    embedding(x, f, self.moves)
                    for x, f in zip(b["x"][use], b["effects"][use], strict=True)
                ]
            )
            self.y = b["y"][use].reshape(-1)
            self.groups = np.repeat(np.arange(use.sum()), len(self.moves))
        self.tree = cKDTree(self.x)
        self.history = []
        self.pending = None

    def select(self, x, fx):
        z = embedding(x, fx, self.moves)
        dist, ix = self.tree.query(z, min(96, len(self.x)))
        # Correlated proposals from one training situation must not outvote
        # several independent incoming balls at similar predicted effects.
        w = np.exp(-0.55 * dist**2)
        for i, neighbors in enumerate(ix):
            _, inv, counts = np.unique(
                self.groups[neighbors], return_inverse=True, return_counts=True
            )
            w[i] /= np.sqrt(counts[inv])
        p = (2.0 * 0.55 + np.sum(w * self.y[ix], axis=1)) / (2 + w.sum(axis=1))
        score = p - 0.015 * np.linalg.norm(self.moves / 0.22, axis=1)
        j = int(np.argmax(score))
        if w[j].sum() < 1.5 or score[j] < score[0] + 0.15:
            j = 0
        return j, p

    def begin(self):
        self.pending = None

    def cancel(self):
        self.pending = None

    def feedback(self, legal, terminal):
        if self.pending is not None and (legal or terminal):
            state, choice = self.pending
            self.history.append((state, choice, bool(legal)))
            self.pending = None

    def plan(self, engine, surf, obs, selected, score, fit, eta):
        result = {
            "tape": None,
            "score": None,
            "eta": None,
            "trace": [],
            "note": {"issued": False},
        }
        self.pending = None
        if obs[2] > 1.075 or not 0.03 < eta < 0.19:
            return result
        x = describe(
            engine,
            obs,
            engine.previous,
            fit,
            selected,
            eta,
            engine.twin.control_dt,
            surf.ready,
        )
        cmds = motions(
            fit, selected, eta, engine.twin.control_dt, surf.ready, self.moves
        )
        fx = outcome_features(engine, cmds, eta)
        j, p = self.select(x, fx)
        result["note"].update(mode=j, p_stock=float(p[0]), p_selected=float(p[j]))
        if j:
            self.pending = (x.copy(), j)
            result.update(
                tape=cmds[j, : max(1, round((eta + 0.075) / engine.twin.control_dt))],
                eta=eta,
                commit=True,
            )
            result["note"]["issued"] = True
        return result


def validate(file, out):
    with np.load(file) as z:
        xs = z["x"]
        ys = z["y"]
        groups = z["release"]
        fs = z["effects"]
    rows = []
    for group in sorted(set(groups)):
        model = Ranker(file, group)
        for i in np.flatnonzero(groups == group):
            j, p = model.select(xs[i], fs[i])
            rows.append(
                {
                    "group": int(group),
                    "i": int(i),
                    "mode": j,
                    "stock": bool(ys[i, 0]),
                    "new": bool(ys[i, j]),
                    "stock_p": float(p[0]),
                    "new_p": float(p[j]),
                }
            )
    _write(
        out,
        {
            "source_sha256": _hash(file),
            "n": len(rows),
            "stock": sum(r["stock"] for r in rows),
            "selected": sum(r["new"] for r in rows),
            "changed": sum(r["mode"] != 0 for r in rows),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("practice", "validate"))
    p.add_argument("out", type=Path)
    p.add_argument(
        "--known", type=Path, default=Path("artifacts/tt40/shared-practice.npz")
    )
    p.add_argument(
        "--source", type=Path, default=Path("artifacts/tt37/mix-rollin.json.gz")
    )
    p.add_argument("--model", type=Path, default=Path("artifacts/tt41/effects.npz"))
    a = p.parse_args()
    if a.mode == "practice":
        practice([a.source], a.out, known=a.known)
    else:
        validate(a.model, a.out)
