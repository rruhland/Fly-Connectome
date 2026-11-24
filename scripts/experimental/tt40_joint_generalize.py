"""Adapt successful joint sequences to a new measured approach through local motor regression."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_joint_credit import Credited

STOCK = mpc.Fragments


class Generalized(Credited):
    def __init__(self, bank, engine, *, extras, generalize):
        super().__init__(bank, engine)
        self.generalize = generalize
        for file in extras:
            extra = STOCK(file, engine)
            self.rows.extend(
                (x, g, ("practice", file.name, *tag)) for x, g, tag in extra.rows
            )
        self.tree = cKDTree(np.stack([r[0] for r in self.rows]))
        self.online = deque(maxlen=512)
        self.online_tree = None
        self.last = None

    def priors(self, x, q, reference, limit):
        key = mpc.query(x, q, reference)
        d, i = self.tree.query(key, min(155, len(self.rows)))
        raw = [
            (float(dist), *self.rows[j], False)
            for dist, j in zip(np.atleast_1d(d), np.atleast_1d(i), strict=True)
        ]
        if self.online:
            if self.last is not self.online[-1]:
                self.last = self.online[-1]
                self.online_tree = cKDTree(np.stack([r[0] for r in self.online]))
                self.indexed = list(self.online)
            d, i = self.online_tree.query(key, min(100, len(self.indexed)))
            raw.extend(
                (max(0.0, float(dist) - 0.1), *self.indexed[j], True)
                for dist, j in zip(np.atleast_1d(d), np.atleast_1d(i), strict=True)
            )
        distinct = []
        used = set()
        for row in sorted(raw, key=lambda r: r[0]):
            if row[3] not in used:
                distinct.append(row)
                used.add(row[3])
            if len(distinct) == 36:
                break
        take = [r for j, r in enumerate(distinct[:limit]) if j < 2 or r[0] <= 3.1]
        if not self.generalize or len(distinct) < 8 or len(take) < 4:
            return [(r[2].copy(), r[4]) for r in take]
        states = np.stack([r[1] for r in distinct])
        moves = np.stack([r[2] for r in distinct])
        dist = np.array([r[0] for r in distinct])
        reference_pose = (
            states[:, 6:11] * mpc.OPEN_NORM[6:11] + states[:, 11:] * mpc.OPEN_NORM[11:]
        )
        # Express a trajectory by its short physical displacement relative to the
        # command already held when the example was recorded. This distinguishes
        # opposite motor modes at similar visual positions.
        shape = (moves[:, [3, 8, 15], :5] - reference_pose[:, None, :]) / [
            0.22,
            0.19,
            0.23,
            0.23,
            0.28,
        ]
        output = [(r[2].copy(), r[4]) for r in take[: max(2, len(take) - 3)]]
        for anchor in range(len(take) - len(output)):
            similarity = np.mean((shape - shape[anchor]) ** 2, axis=(1, 2))
            weights = np.exp(
                -0.5
                * np.minimum(
                    (dist / max(0.40, dist[min(8, len(dist) - 1)])) ** 2
                    + 2.1 * similarity,
                    80,
                )
            )
            weights /= weights.sum()
            center = weights @ states
            dx = states - center
            y = moves[:, :, :5].reshape(len(moves), -1)
            mean = weights @ y
            slope = np.linalg.solve(
                dx.T @ (dx * weights[:, None]) + 0.12 * np.eye(len(key)),
                dx.T @ ((y - mean) * weights[:, None]),
            )
            # Preserve the actually rewarded mode and adjust only its estimated
            # displacement for this new approach. Every proposal still faces the
            # same independent physical checks as an unmodified fragment.
            offset = ((key - states[anchor]) @ slope).reshape(mpc.HORIZON, 5)
            pred = moves[anchor].copy()
            pred[:, :5] += 0.9 * np.clip(offset, -0.32, 0.32)
            output.append((pred, distinct[anchor][4]))
        return output


def run(a):
    ids = {v["id"] for v in _read(a.plan)[a.deck][: a.limit]}
    for src in [a.bank, *a.extras]:
        if ids & {v["card"]["id"] for v in _read(src.with_suffix(".json"))["practice"]}:
            raise ValueError("Controller practiced a test release")
    instances = []

    def setup(bank, engine):
        v = Generalized(bank, engine, extras=a.extras, generalize=a.mode == "adapt")
        instances.append(v)
        return v

    with patch.object(mpc, "Fragments", setup):
        mpc.run(
            a.feeds,
            a.bank,
            a.plan,
            a.deck,
            "live",
            a.out,
            optic=mpc.ROOT / "tt37/rebound.npz",
            limit=a.limit,
            variant="elastic",
            cap=200,
        )
    model = instances[0]
    ck = a.out.with_suffix("").with_suffix(".npz")
    np.savez_compressed(
        ck,
        states=[v[0] for v in model.online],
        goals=[v[1] for v in model.online],
        tags=[repr(v[2]) for v in model.online],
    )
    doc = _read(a.out)
    doc.update(
        consolidated={str(f): _hash(f) for f in a.extras},
        rule=a.mode,
        online_checkpoint=_hash(ck),
    )
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("retrieve", "adapt"))
    p.add_argument("out", type=Path)
    p.add_argument(
        "--plan", type=Path, default=Path("artifacts/tt37/memory-course.json")
    )
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument(
        "--bank", type=Path, default=Path("artifacts/tt37/self-combined.npz")
    )
    p.add_argument(
        "--extras",
        type=Path,
        nargs="*",
        default=[
            Path("artifacts/tt37/camera-time-school.npz"),
            Path("artifacts/tt37/camera-mix-rollin.npz"),
        ],
    )
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
