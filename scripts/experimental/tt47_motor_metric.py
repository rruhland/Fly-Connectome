"""Learn which differences between visual and motor states permit reusing a previously scored joint movement."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt47_metric")
WEIGHTS = ROOT / "weights.json"
mpc = course.mpc
GROUPS = (
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (5,),
    (6, 10),
    (7, 8, 9),
    (11, 15),
    (12, 13, 14),
)


def samples():
    examples = []
    for file in servo.SOURCES:
        receipt = _read(file.with_suffix(".json"))
        if _hash(file) != receipt["bank_sha256"]:
            raise ValueError("Changed motor demonstrations")
        with np.load(file) as bank:
            q, m, t = bank["states"], bank["marks"], bank["targets_now"]
        start = 0
        for point in receipt["practice"]:
            end = start + point["frames"]
            for k in range(start + 1, end - 13, 3):
                tick = int(m[k, 0])
                if (
                    not -0.46 < m[k, 1] < -0.025
                    or q[k, 4] > -0.9
                    or m[k + 12, 0] - m[k - 1, 0] != 13
                ):
                    continue
                key = mpc.query(q[k, :6], q[k, 6:11], t[k - 1])
                # Relative multi-tick reference movement is independent of
                # trivial similarity between absolute starting joint angles.
                motor = (t[k + np.array([1, 5, 10]), :5] - t[k - 1, :5]).ravel()
                group = point["card"]["id"]
                hit = round((tick * mpc.DT - m[k, 1]) * 480)
                examples.append((key, motor, group, (group, hit)))
            start = end
    return examples


def neighbours(values, release, tags):
    tree = cKDTree(values)
    _, ids = tree.query(values, k=min(360, len(values)))
    pairs = []
    for k, group in enumerate(release):
        candidates = ids[k][release[ids[k]] != group]
        if len(candidates) < 65:
            continue
        # Keep nearest frames from up to 60 separate scored motions.
        unique = []
        for i in candidates:
            if all(tags[i] != tags[j] for j in unique):
                unique.append(i)
            if len(unique) == 60:
                break
        if len(unique) >= 16:
            pairs.append((k, np.array(unique[:16])))
    return pairs


def train(out):
    ex = samples()
    x, m, groups, tags = zip(*ex, strict=True)
    x, m, groups = map(np.asarray, (x, m, groups))
    pairs = neighbours(x, groups, tags)
    k = np.array([q for q, ids in pairs])
    ids = np.stack([ids for _, ids in pairs])
    dif = (x[k, None] - x[ids]) ** 2
    shape = np.mean((m[k, None] - m[ids]) ** 2, axis=2) ** 0.5
    distance = np.stack([dif[:, :, list(c)].sum(axis=2) for c in GROUPS], axis=-1)

    def loss(w, mask):
        ds = distance[mask] @ w
        err = shape[mask]
        order = np.argsort(ds, axis=1)[:, :5]
        ranked = np.take_along_axis(err, order, axis=1)
        return float(
            0.45 * np.mean(np.min(ranked, axis=1)) + 0.55 * np.mean(ranked[:, 0])
        )

    def optimize(mask):
        weights = np.ones(len(GROUPS))
        for _ in range(2):
            for i in range(len(weights)):
                options = []
                for v in (0.25, 0.5, 1.0, 2.0, 4.0):
                    tmp = weights.copy()
                    tmp[i] = v
                    options.append(
                        (loss(tmp, mask) + 0.0005 * np.mean(np.log2(tmp) ** 2), v)
                    )
                weights[i] = min(options)[1]
        return weights

    held = []
    for fold in np.array_split(
        np.random.default_rng(4709).permutation(np.unique(groups)), 4
    ):
        test = np.isin(groups[k], fold)
        w = optimize(~test)
        held.append(
            dict(
                queries=int(test.sum()),
                stock=loss(np.ones(len(w)), test),
                fit=loss(w, test),
            )
        )
    w = optimize(np.ones(len(k), dtype=bool))
    expand = np.ones(16)
    for group, v in zip(GROUPS, w, strict=True):
        expand[list(group)] = np.sqrt(v)
    out.parent.mkdir(exist_ok=True, parents=True)
    _write(
        out,
        dict(
            scale=expand.tolist(),
            held=held,
            query_count=len(k),
            release=np.unique(groups).tolist(),
            source={p.name: _hash(p) for p in servo.SOURCES},
        ),
    )


class Metric(course.Carried):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = np.array(_read(WEIGHTS)["scale"])
        self.fitted = cKDTree(np.array([r[0] for r in self.rows]) * self.scale)

    def priors(self, x, q, ref, limit):
        original = super().priors(x, q, ref, limit)
        if len(original) < 4:
            return original
        key = mpc.query(x, q, ref) * self.scale
        distances, ids = self.fitted.query(key, min(len(self.rows), 150))
        choices = [
            (ds, self.rows[i][1], self.rows[i][2], False)
            for ds, i in zip(distances, ids)
        ]
        choices.extend(
            (max(0.0, np.linalg.norm(state * self.scale - key) - 0.10), t, leg, True)
            for state, t, leg in self.online
        )
        out = original[:3]
        used = {s.tobytes() for s, _ in out}
        tags = set()
        for _, seq, leg, earned in sorted(choices, key=lambda r: r[0]):
            signature = seq.tobytes()
            if leg in tags:
                continue
            tags.add(leg)
            if signature in used:
                continue
            used.add(signature)
            out.append((seq.copy(), earned))
            if len(out) >= len(original):
                break
        return out


def run(a):
    with patch.object(course, "Carried", Metric):
        season.run(a)
    d = _read(a.out)
    d.update(motor_metric=_hash(WEIGHTS))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "live", "frozen"))
    p.add_argument("out", type=Path)
    p.add_argument("--n", default=16, type=int)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(deck="practice", carry=season.CARRY, cap=240, resume=False)
    a = p.parse_args()
    train(a.out) if a.mode == "fit" else run(a)
