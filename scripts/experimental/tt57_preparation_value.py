"""Select short direct joint movements by which subsequent observed situations yielded returns."""

from pathlib import Path

import numpy as np
from scipy.special import expit

from . import tt56_scored_motor_atlas as atlas
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt57_preparation_value")
MODEL = ROOT / "state-values.npz"
BALL = np.array([0, 1, 2, 3, 4, 5, 16, 17])


class Value:
    def __init__(self, X, Y, seed=571107):
        self.mu = X.mean(axis=0)
        self.scale = np.maximum(0.18, X.std(axis=0))
        self.projection = np.random.default_rng(seed).normal(
            size=(X.shape[1], 112)
        ) / np.sqrt(X.shape[1])
        X = self.features(X)
        # A regularized logistic readout of fixed features. The lower weight
        # of well-classified samples reduces the influence of routine high balls.
        w = np.zeros(X.shape[1])
        for _ in range(8):
            p = expit(X @ w)
            hessian = X.T @ ((p * (1 - p))[:, None] * X) + 4 * np.eye(len(w))
            delta = np.linalg.solve(hessian, X.T @ (Y - p) - 4 * w)
            w += delta
            if np.max(abs(delta)) < 1e-5:
                break
        self.w = w

    def features(self, X):
        z = np.clip((np.atleast_2d(X) - self.mu) / self.scale, -4, 4)
        f = z @ self.projection
        return np.c_[np.ones(len(z)), z, np.sin(f), np.cos(f)]

    def predict(self, x):
        return expit(self.features(x) @ self.w)


def fit():
    with np.load(atlas.MODEL) as d:
        x, y, g = d["keys"].astype(float), d["reward"].astype(float), d["release"]
    groups = np.random.default_rng(57312).permutation(np.unique(g))
    score = {}
    for name, col in [("optical", BALL), ("optical_and_motor", np.arange(x.shape[1]))]:
        pred = np.empty_like(y)
        for hold in np.array_split(groups, 5):
            tr = ~np.isin(g, hold)
            v = Value(x[tr][:, col], y[tr])
            pred[~tr] = v.predict(x[~tr][:, col])
        score[name] = float(np.mean((pred - y) ** 2))
    final = Value(x, y)
    ROOT.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        MODEL, mu=final.mu, scale=final.scale, projection=final.projection, w=final.w
    )
    _write(
        ROOT / "fit.json",
        dict(
            samples=len(y),
            releases=len(groups),
            release_held_brier=score,
            model=_hash(MODEL),
            training=_hash(atlas.MODEL),
        ),
    )


if __name__ == "__main__":
    fit()
