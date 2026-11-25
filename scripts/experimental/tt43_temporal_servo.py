"""Distill early direct joint feedback with fixed temporal features and a closed-form motor readout."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_deep import SCALES

ROOT = Path("artifacts/tt43_servo")
SOURCES = (course.mpc.BANK, course.PRACTICE)
D = 11
LAGS = (1, 3, 6)


def feature(sequence):
    sequence = np.asarray(sequence)
    # Joint differences encode actuator velocity, which identical current
    # positions can conceal. Ball differences capture a change of flight.
    return np.concatenate(
        [
            sequence[-1] / SCALES["local"],
            *[
                (sequence[-1] - sequence[-1 - i]) / np.sqrt(i) / SCALES["local"]
                for i in LAGS
            ],
        ]
    )


def packets(paths):
    X, Y, groups = [], [], []
    for path in paths:
        receipt = _read(path.with_suffix(".json"))
        if _hash(path) != receipt["bank_sha256"]:
            raise ValueError("Changed optical practice")
        with np.load(path) as p:
            states, marks, now = (p[k] for k in ("states", "marks", "targets_now"))
        start = 0
        for point in receipt["practice"]:
            end = start + point["frames"]
            for i in range(start + 6, end):
                if (
                    states[i, 4] >= -0.9
                    or states[i, 1] < -0.57
                    or marks[i, 0] - marks[i - 6, 0] != 6
                    or np.any(states[i - 6 : i + 1, 4] > -0.9)
                ):
                    continue
                X.append(feature(states[i - 6 : i + 1]))
                Y.append(now[i, :5] - states[i, 6:11])
                groups.append(point["card"]["id"])
            start = end
    return np.stack(X), np.stack(Y), np.array(groups)


class Readout:
    def __init__(self, X, Y, history):
        self.history = history
        X = X if history else X[:, :D]
        self.mean = X.mean(0)
        self.std = np.maximum(X.std(0), 0.065)
        rng = np.random.default_rng(43876)
        self.filters = rng.normal(size=(X.shape[1], 192)) / np.sqrt(X.shape[1])
        self.gates = rng.normal(size=(X.shape[1], 192)) / np.sqrt(X.shape[1])
        self.bias = rng.uniform(-1, 1, size=192)
        F = self.encode(X)
        self.weights = np.linalg.solve(F.T @ F + 0.04 * np.eye(F.shape[1]), F.T @ Y)

    def encode(self, X):
        X = np.atleast_2d(X)
        u = np.clip(
            ((X if self.history else X[:, :D]) - self.mean) / self.std, -5.0, 5.0
        )
        h = np.tanh(u @ self.filters + self.bias) / (
            1 + np.exp(-np.clip(u @ self.gates, -12, 12))
        )
        return np.c_[np.ones(len(u)), u, h]

    def predict(self, X):
        return self.encode(X) @ self.weights


def fit(out):
    X, Y, group = packets(SOURCES)
    splits = np.array_split(
        np.random.default_rng(4301).permutation(np.unique(group)), 5
    )
    held = []
    for ids in splits:
        test = np.isin(group, ids)
        plain = Readout(X[~test], Y[~test], False).predict(X[test])
        temporal = Readout(X[~test], Y[~test], True).predict(X[test])
        for i in ids:
            mask = group[test] == i
            held.append(
                {
                    "id": int(i),
                    "n": int(mask.sum()),
                    "instant": float(
                        np.linalg.norm(plain[mask] - Y[test][mask], axis=1).mean()
                    ),
                    "history": float(
                        np.linalg.norm(temporal[mask] - Y[test][mask], axis=1).mean()
                    ),
                }
            )
    obj = Readout(X, Y, True)
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out, **obj.__dict__)
    _write(
        out.with_suffix(".json"),
        {
            "inputs": {p.name: _hash(p) for p in SOURCES},
            "model": _hash(out),
            "rows": held,
            "examples": len(X),
            "start": -0.57,
            "lags": LAGS,
        },
    )
    return held


def reload(path):
    if _hash(path) != _read(path.with_suffix(".json"))["model"]:
        raise ValueError("Readout checkpoint changed")
    obj = Readout.__new__(Readout)
    with np.load(path) as f:
        obj.__dict__.update({k: f[k] for k in f.files})
    return obj


class Temporal(course.mpc.MPC):
    def begin(self):
        super().begin()
        self.samples = deque(maxlen=7)
        self.motor = reload(ROOT / "prior.npz")

    def step(self, tick, eye, x, q, reference, base):
        if x is None or x[4] >= -0.9:
            self.samples.clear()
        else:
            self.samples.append(np.r_[x, q[:5]])
            if len(self.samples) == 7 and -0.57 < x[1] < 1.35:
                inferred = q[:5] + self.motor.predict(feature(self.samples))[0]
                # Respect the existing range when outside the practice manifold.
                candidate = base.copy()
                candidate[:5] += 0.65 * np.clip(inferred - base[:5], -0.25, 0.25)
                base = candidate
                self.diag["temporal_inbound_steps"] += 1
        return super().step(tick, eye, x, q, reference, base)


def run(a):
    held = {c["id"] for c in _read(course.PLAN)[a.deck][: a.n]}
    trained = {r["id"] for r in _read(ROOT / "prior.json")["rows"]}
    if held & trained:
        raise ValueError("Release overlap")
    with patch.object(course.mpc, "MPC", Temporal):
        course.run(a)
    d = _read(a.out)
    d["temporal_sha256"] = _hash(ROOT / "prior.npz")
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "play"))
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--carry", type=Path, default=course.ROOT / "learning-12.json.gz")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(mode="frozen", deck="assessment", cap=240, resume=False)
    a = p.parse_args()
    if a.phase == "fit":
        fit(a.out)
    else:
        run(a)
