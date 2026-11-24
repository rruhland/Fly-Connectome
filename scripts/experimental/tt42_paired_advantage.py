"""Learn how changing a motor command changes landing reward within matched training shots."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt41_effect_rank import Ranker


class Advantages(Ranker):
    """A linear readout on fixed nonlinear features, learned by paired least squares."""

    def __init__(self, path, exclude=None):
        self.history = []
        self.pending = None
        with np.load(path) as z:
            self.moves = z["moves"]
            self.ids = z["release"]
            x, e, y = z["x"], z["effects"], z["y"]
        use = self.ids != exclude if exclude is not None else np.ones(len(x), bool)
        input_width = self.features(x[0], e[0]).shape[1]
        rng = np.random.default_rng(420249)
        self.projections = rng.normal(size=(input_width, 80)) / np.sqrt(input_width)
        self.phases = rng.uniform(-np.pi, np.pi, size=80)
        phi = np.stack(
            [self.basis(state, effect) for state, effect in zip(x, e, strict=True)]
        )
        self.fit(phi[use], y[use])

    def features(self, x, eff):
        # Observed receive height, travel and encoders describe the current
        # situation. All candidate effects come from the camera-initialized twin.
        e = np.clip(
            eff / np.array([1.0, 1.0, 0.05, 0.45, 0.55, 0.15, 2.0, 2.0, 1.8, 0.045]),
            -5,
            5,
        )
        x0 = np.clip(
            (x[[0, 1, 2, 3, 4, 5]] - [0.12, 0, 1.045, 0, -3.6, 0])
            / [0.06, 0.19, 0.07, 1.4, 1.25, 1.8],
            -4,
            4,
        )
        m = self.moves / np.array(
            [0.10, 0.13, 0.13, 0.20, 0.22, 0.12, 0.14, 0.17, 0.032]
        )
        return np.c_[e, m, np.broadcast_to(x0, (len(m), len(x0)))]

    def basis(self, x, eff):
        z = self.features(x, eff)
        e = z[:, :10]
        m = z[:, 10:19]
        state = z[0, 19:]
        curves = np.c_[
            np.abs(e[:, 3]),
            np.abs(e[:, 4] - 1.2),
            np.maximum(0, 0.30 - e[:, 5]),
            np.maximum(0, e[:, 9] - 0.95),
        ]
        effects = np.c_[e, curves, m]
        interactions = (effects[:, :, None] * state).reshape(len(m), -1)
        fixed = np.sqrt(2.0) * np.cos(z @ self.projections + self.phases)
        return np.c_[effects, 0.36 * interactions, 0.48 * fixed]

    def fit(self, phi, reward):
        # The same incoming and pre-action limb state are held fixed across
        # alternatives. A difficult feed alone cannot teach an action penalty.
        p = phi - phi.mean(axis=1, keepdims=True)
        y = reward - reward.mean(axis=1, keepdims=True)
        x = p.reshape(-1, p.shape[-1])
        alpha = 20.0
        self.weights = np.linalg.solve(
            x.T @ x + alpha * np.eye(x.shape[-1]), x.T @ y.ravel()
        )

    def select(self, x, fx):
        basis = self.basis(x, fx)
        values = (basis - basis[0]) @ self.weights
        values -= 0.010 * np.linalg.norm(self.moves / 0.20, axis=1)
        j = int(np.argmax(values))
        return (j if values[j] > 0.12 else 0), values


def validate(file, out):
    with np.load(file) as z:
        states, effects, y, groups = z["x"], z["effects"], z["y"], z["release"]
    rows = []
    for g in sorted(set(groups)):
        model = Advantages(file, g)
        for i in np.flatnonzero(groups == g):
            j, v = model.select(states[i], effects[i])
            rows.append(
                {
                    "index": int(i),
                    "release": int(g),
                    "selected": int(j),
                    "stock": int(y[i, 0]),
                    "reward": int(y[i, j]),
                    "estimated_gain": float(v[j]),
                }
            )
    _write(
        out,
        {
            "training_sha256": _hash(file),
            "stock": sum(r["stock"] for r in rows),
            "selected": sum(r["reward"] for r in rows),
            "ceiling": int(y.any(axis=1).sum()),
            "changed": sum(r["selected"] != 0 for r in rows),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--play", action="store_true")
    a = p.parse_args()
    if not a.play:
        validate(a.file, a.out)
    else:
        from . import tt40_paired_practice as paired
        from .tt41_practice_curriculum import clinic

        with patch.object(paired, "Paired", Advantages):
            clinic(
                "optical-fixed",
                a.out,
                "assessment",
                Path("artifacts/tt37/independent-cold-value.json.gz"),
                0,
                12,
                options=("--paired-practice", str(a.file)),
            )
