"""Choose joint corrections by their measured success on neighboring visual approaches."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt37_motor_clinic import SCALE, landing_error

ACTION_SCALE = np.array([0.23, 0.33, 0.38, 0.36, 0.5, 2.2, 3.0, 2.8, 3.0, 3.6])


class Response:
    def __init__(self, paths, *, adaptive=False):
        table = np.concatenate([self.load(p) for p in paths])
        states, inverse = np.unique(table[:, :10], return_inverse=True, axis=0)
        self.states = states
        self.index = cKDTree(states / SCALE)
        self.maps = []
        self.degree = []
        self.good = []
        for j in range(len(states)):
            part = table[inverse == j]
            a = part[:, 10:20]
            sense = part[:, 20:29]
            legal = part[:, 29] >= 0
            valid = sense[:, 1] > 0
            value = 0.09 * np.exp(-6.0 * sense[:, 0])
            value[valid] = 0.16 + 0.48 * np.exp(
                -2.5 * np.minimum(landing_error(sense[valid]), 8.0)
            )
            value[legal] = 0.86 + 0.14 * np.exp(-0.25 * part[legal, 29])
            self.maps.append((cKDTree(a / ACTION_SCALE), value))
            self.good.append(a[np.argsort(value)[-6:]].copy())
            self.degree.append(int(legal.sum()))
        self.adaptive = adaptive
        self.experience = []
        self.last = None
        self.nextout = 0

    @staticmethod
    def load(path):
        with np.load(Path(path)) as archive:
            return archive["rows"].astype(float)

    def observe(self, won):
        if self.last is not None and self.adaptive:
            x, act = self.last
            self.experience.append((x, act, float(won)))
        self.last = None

    def __call__(self, bank, x):
        dist, idx = self.index.query(x / SCALE, k=min(12, len(self.maps)))
        idx = np.atleast_1d(idx)
        dist = np.atleast_1d(dist)
        weight = np.exp(-0.5 * (dist / max(0.25, dist[min(2, len(dist) - 1)])) ** 2) / (
            dist + 0.15
        )
        weight /= weight.sum()
        popular = np.vstack([self.good[j] for j in idx[:8]])
        early = self.good[idx[0]][-1]
        candidates = np.vstack(
            [popular, 0.5 * (popular + early), 0.5 * popular, np.zeros(10)]
        )
        pred = np.zeros(len(candidates))
        for w, j in zip(weight, idx):
            tree, values = self.maps[j]
            d, k = tree.query(candidates / ACTION_SCALE, k=min(7, len(values)))
            aw = 1 / (0.16 + d) ** 2
            pred += w * np.sum(aw * values[k], axis=1) / aw.sum(axis=1)
        if self.experience:
            contexts, acts, outcomes = map(
                np.asarray, zip(*self.experience, strict=True)
            )
            cd = np.linalg.norm((contexts - x) / SCALE, axis=1)
            local = cd < 2.0
            if local.any():
                ad = np.linalg.norm(
                    (acts[local, None, :] - candidates) / ACTION_SCALE, axis=2
                )
                w = np.exp(-0.5 * (cd[local, None] / 0.8) ** 2 - 0.5 * (ad / 0.9) ** 2)
                pred = (pred + 0.65 * w.T @ outcomes[local]) / (
                    1.0 + 0.65 * w.sum(axis=0)
                )
        chosen = candidates[np.argmax(pred)].copy()
        self.last = x.copy(), chosen.copy()
        return chosen
