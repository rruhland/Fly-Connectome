"""Forecast observed incoming ball motion before the opponent strikes, from public video."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from scipy.ndimage import label as connected
from scipy.linalg import solve

from .tt4_transfer_trial import _write
from .tt50_opponent_vision import ROOT, crop

GATES = (-0.45, 0.0, 0.45, 0.85)
TARGET = [0, 2, 3, 4, 5]


def fit_background(paths):
    frames = []
    for path in paths:
        with np.load(path) as data:
            p = data["rgb"]
            frames.extend(p[np.linspace(0, len(p) - 1, 16, dtype=int)])
    return np.median(frames, axis=0).astype(np.uint8)


def appearance(frame, background):
    # Identify the compact low-chroma moving region in each far-court window.
    # Shading and silhouette carry orientation information at sub-pixel motion.
    sides = []
    for im, bg in zip(frame.astype(np.int16), background.astype(np.int16), strict=True):
        delta = np.abs(im - bg).max(axis=-1)
        r, g, b = np.moveaxis(im, -1, 0)
        mask = (delta >= 5) & (r < 97) & (b >= r) & (b - r < 19) & (g - r < 10)
        labels, n = connected(mask, structure=np.ones((3, 3)))
        which = np.bincount(labels.ravel(), minlength=2)
        which[0] = 0
        ident = int(np.argmax(which))
        valid = n and which[ident] >= 7
        if not valid:
            sides.append(np.full(14, np.nan))
            continue
        y, x = np.nonzero(labels == ident)
        cx, cy = x.mean(), y.mean()
        a, b = x - cx, y - cy
        color = im[y, x].mean(axis=1)
        sides.append(
            np.array(
                [
                    cx,
                    cy,
                    len(x),
                    np.mean(a * a),
                    np.mean(a * b),
                    np.mean(b * b),
                    np.mean(a**3),
                    np.mean(b**3),
                    color.mean(),
                    np.percentile(color, 75),
                    np.mean((color - color.mean()) * a),
                    np.mean((color - color.mean()) * b),
                    np.mean(a * a * b),
                    np.mean(b * b * a),
                ]
            )
        )
    return np.concatenate(sides)


def descriptor(history, x):
    if len(history) < 4:
        return None
    stack = np.array(history)
    recent = stack[-4:]
    if not np.isfinite(recent[-1]).all() or np.sum(np.isfinite(recent[:, 0])) < 2:
        return None
    filled = np.where(np.isfinite(recent), recent, recent[-1])
    current = filled[-1]
    # Position, shape, reflectance and recent velocity of the visible opponent.
    return np.r_[x, current, (filled[-1] - filled[0]) / 3, filled.mean(axis=0)]


class Predictor:
    """Call on real stereo exposures and the existing ball track. Returns once per crossed gate."""

    def __init__(self, path=ROOT / "model.npz"):
        with np.load(path) as data:
            self.background = data["background"]
            self.params = {
                g: [data[f"{j}_{k}"] for k in ("mean", "scale", "coef", "bias")]
                for j, g in enumerate(GATES)
                if f"{j}_coef" in data
            }
        self.reset()

    def reset(self):
        self.history = deque(maxlen=6)
        self.seen = set()
        self.last_out = None
        self.latest = None

    def update(self, frames, x, tick):
        if frames is None:
            return None
        if x is not None and x[4] < -0.9:
            self.reset()
            return None
        if self.last_out is not None and tick - self.last_out > 8:
            self.reset()
        self.history.append(appearance(crop(frames), self.background))
        if x is None or x[4] < 0.9:
            return None
        self.last_out = tick
        for gate in GATES:
            if gate not in self.seen and gate in self.params and x[1] >= gate:
                self.seen.add(gate)
                f = descriptor(self.history, x)
                if f is not None:
                    m, s, c, b = self.params[gate]
                    self.latest = (gate, (f - m) / s @ c + b)
                    return self.latest
        return None


def examples(path, bg):
    with np.load(path) as z:
        rgb = z["rgb"]
        seen = np.array([appearance(frame, bg) for frame in rgb])
        rows = []
        for b, y in enumerate(z["label"][:, TARGET]):
            if not np.isfinite(y).all():
                continue
            hist = deque(maxlen=6)
            used = set()
            idx = np.flatnonzero(z["stream"] == b)
            for k in idx:
                hist.append(seen[k])
                x = z["ball"][k]
                if not np.isfinite(x).all() or x[4] < 0.9:
                    continue
                for gate in GATES:
                    if x[1] >= gate and gate not in used:
                        used.add(gate)
                        f = descriptor(hist, x)
                        if f is not None:
                            rows.append(
                                (
                                    gate,
                                    b,
                                    int(z["id"]),
                                    f,
                                    y,
                                    (z["contact"][b] - z["tick"][k]) / 60,
                                )
                            )
        return rows


def fit(X, Y, group, alpha):
    mean = X.mean(axis=0)
    # Common scales for within-release supervision and cross-release examples.
    scale = np.maximum(X.std(axis=0), 1e-3)
    A = (X - mean) / scale
    bias = Y.mean(axis=0)
    B = Y - bias
    da = np.empty_like(A)
    db = np.empty_like(B)
    for i in np.unique(group):
        idx = group == i
        da[idx] = A[idx] - A[idx].mean(axis=0)
        db[idx] = B[idx] - B[idx].mean(axis=0)
    design = np.vstack((A, 2 * da))
    labels = np.vstack((B, 2 * db))
    coef = solve(
        design.T @ design + alpha * np.eye(design.shape[1]),
        design.T @ labels,
        assume_a="pos",
    )
    return mean, scale, coef, bias


def run():
    train = sorted((ROOT / "observations").glob("case-*.npz"))
    tr = [p for p in train if int(p.stem.split("-")[-1]) < 16]
    held = [p for p in train if p not in tr]
    if len(tr) < 8 or not held:
        raise ValueError("Need practice and held RGB files")
    bg = fit_background(tr)
    A = [r for p in tr for r in examples(p, bg)]
    H = [r for p in held for r in examples(p, bg)]
    model = dict(background=bg)
    comparator = dict(background=bg)
    report = dict(train_files=len(tr), held_files=len(held), gates={})
    for j, gate in enumerate(GATES):

        def matrices(rows):
            sub = [r for r in rows if r[0] == gate]
            return [np.array([r[k] for r in sub]) for k in (3, 4, 2, 5)]

        x, y, g, _ = matrices(A)
        h, t, v, lead = matrices(H)
        if not len(h):
            continue
        info = dict(
            held=len(h),
            training=len(x),
            lead_s=np.percentile(lead, [10, 50, 90]).tolist(),
        )
        for name, d in (("ball", 6), ("vision", x.shape[1])):
            m, s, c, b = fit(x[:, :d], y, g, 12.0)
            pred = (h[:, :d] - m) / s @ c + b
            diff = pred - t
            info[name] = dict(
                mae=np.mean(abs(diff), axis=0).tolist(),
                centered_mae=np.mean(
                    abs(
                        np.concatenate(
                            [
                                diff[v == group] - diff[v == group].mean(axis=0)
                                for group in np.unique(v)
                            ]
                        )
                    ),
                    axis=0,
                ).tolist(),
            )
            if name == "ball":
                extra = x.shape[1] - 6
                parts = (
                    np.r_[m, np.zeros(extra)],
                    np.r_[s, np.ones(extra)],
                    np.vstack((c, np.zeros((extra, len(b))))),
                    b,
                )
                for k, z in zip(("mean", "scale", "coef", "bias"), parts):
                    comparator[f"{j}_{k}"] = z
            if name == "vision":
                for k, z in zip(("mean", "scale", "coef", "bias"), (m, s, c, b)):
                    model[f"{j}_{k}"] = z
                wrong = h.copy()
                for group in np.unique(v):
                    select = v == group
                    wrong[select, 6:] = np.roll(h[select, 6:], 1, axis=0)
                info["wrong_stream_mae"] = np.mean(
                    abs((wrong - m) / s @ c + b - t), axis=0
                ).tolist()
        report["gates"][str(gate)] = info
    np.savez_compressed(ROOT / "model.npz", **model)
    np.savez_compressed(ROOT / "ball-only.npz", **comparator)
    _write(ROOT / "evaluation.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    run()
