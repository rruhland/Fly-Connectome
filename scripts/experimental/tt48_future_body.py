"""Use rewarded future encoder displacements to condition a learned local inverse motor model."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt48_future_body")
MODEL = ROOT / "body.npz"
HISTORY, LEAD = 3, 10
SCALE = np.array(
    [
        0.16,
        0.35,
        0.10,
        0.62,
        0.90,
        1.0,
        0.42,
        0.45,
        0.45,
        0.45,
        0.52,
        0.04,
        0.04,
        0.035,
        0.05,
        0.06,
    ]
)


def key(early, now):
    return np.r_[now, now[6:11] - early[6:11]] / SCALE


def inverse_features(k, delta):
    # Proprioception and intended displacement train the inverse model.
    q = k[..., 6:11] * SCALE[6:11]
    v = k[..., 11:16]
    u = delta / 0.20
    return np.concatenate(
        [
            np.ones((*q.shape[:-1], 1)),
            q,
            v,
            u,
            u * np.abs(u),
            u * np.abs(v),
            u * np.sin(q),
            np.sin(q),
        ],
        axis=-1,
    )


def fit_inverse(keys, displacement, issued):
    a = inverse_features(keys, displacement)
    return np.linalg.solve(a.T @ a + 3.0 * np.eye(a.shape[1]), a.T @ issued)


class Episodes:
    def __init__(self, keys, movements):
        self.keys, self.movements = keys, movements
        self.tree = cKDTree(keys)
        self.earned = deque(maxlen=1024)
        self.failed = deque(maxlen=160)

    def desired(self, k):
        d, ix = self.tree.query(k, k=12)
        moves = self.movements[ix]
        if self.earned:
            earned_k, earned_d = map(np.asarray, zip(*self.earned))
            near = np.linalg.norm(earned_k - k, axis=1)
            i = np.argsort(near)[:8]
            d, moves = np.r_[d, near[i]], np.vstack([moves, earned_d[i]])
        keep = d <= min(4.2, d.min() + 1.1)
        if not keep.any():
            return np.zeros(5), 0.0
        d, moves = d[keep], moves[keep]
        weight = np.exp(-0.55 * (d**2 - d.min() ** 2))
        if self.failed:
            fk, fm = map(np.asarray, zip(*self.failed))
            dist = np.sum((fk - k) ** 2, axis=1)
            i = np.argsort(dist)[:6]
            closeness = np.sum((moves[:, None] - fm[i]) ** 2, axis=2) / 0.045**2
            weight /= 1.0 + np.exp(-0.6 * dist[i] - closeness).sum(axis=1)
        # An observed member of the neighborhood keeps joint coordination from
        # a single attainable movement, even if different successful modes occur nearby.
        separation = np.linalg.norm(moves[:, None] - moves, axis=2)
        pick = np.argmin(separation @ weight / weight.sum() + 0.012 * (d - d.min()))
        return moves[pick], float(np.clip((4.2 - d.min()) / 2.0, 0, 1))


def train():
    keys, delta, goals, releases = [], [], [], []
    for path in servo.SOURCES:
        doc = _read(path.with_suffix(".json"))
        if _hash(path) != doc["bank_sha256"]:
            raise ValueError("Changed motor practice")
        with np.load(path) as b:
            states, marks, issued = (b[k] for k in ("states", "marks", "targets_now"))
        s = 0
        for row in doc["practice"]:
            e = s + row["frames"]
            for i in range(s + HISTORY, e - LEAD):
                if not -0.57 < states[i, 1] < 1.35 or np.any(
                    states[i - HISTORY : i + LEAD + 1, 4] >= -0.9
                ):
                    continue
                if marks[i + LEAD, 0] - marks[i - HISTORY, 0] != LEAD + HISTORY:
                    continue
                keys.append(key(states[i - HISTORY], states[i]))
                delta.append(states[i + LEAD, 6:11] - states[i, 6:11])
                goals.append(issued[i, :5] - states[i, 6:11])
                releases.append(row["card"]["id"])
            s = e
    keys, delta, goals, releases = map(np.asarray, (keys, delta, goals, releases))
    results = []
    for fold in np.array_split(
        np.random.default_rng(480101).permutation(np.unique(releases)), 4
    ):
        held = np.isin(releases, fold)
        memory = Episodes(keys[~held], delta[~held])
        model = fit_inverse(keys[~held], delta[~held], goals[~held])
        observed = delta[held]
        predicted = np.array([memory.desired(k)[0] for k in keys[held]])
        target = inverse_features(keys[held], predicted) @ model
        results.append(
            dict(
                n=int(held.sum()),
                no_motion=float(np.linalg.norm(observed, axis=1).mean()),
                displacement_error=float(
                    np.linalg.norm(predicted - observed, axis=1).mean()
                ),
                command_error=float(
                    np.linalg.norm(target - goals[held], axis=1).mean()
                ),
            )
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL, keys=keys, delta=delta, inverse=fit_inverse(keys, delta, goals)
    )
    _write(
        MODEL.with_suffix(".json"),
        dict(
            source={str(p): _hash(p) for p in servo.SOURCES},
            model=_hash(MODEL),
            folds=results,
            samples=len(keys),
            release_ids=np.unique(releases).tolist(),
        ),
    )


class Memory(course.Carried):
    def keep(self, tape, leg):
        self.callback(tape, True)
        return super().keep(tape, leg)


class Player(season.Player):
    def __init__(self, *args, episodes, inverse, **kw):
        super().__init__(*args, **kw)
        self.episodes, self.inverse = episodes, inverse
        self.priors.callback = self.hear

    def begin(self):
        super().begin()
        self.observations = deque(maxlen=110)

    def step(self, tick, eye, x, q, ref, base):
        original = super().step(tick, eye, x, q, ref, base)
        if x is None or len(self.samples) < HISTORY + 1 or not -0.52 < x[1] < 1.35:
            return original
        k = key(self.samples[-1 - HISTORY], self.samples[-1])
        self.observations.append((tick, k))
        delta, confidence = self.episodes.desired(k)
        desired = q[:5] + inverse_features(k, delta) @ self.inverse
        goal = original.copy()
        goal[:5] += 0.44 * confidence * np.clip(desired - original[:5], -0.25, 0.25)
        self.diag["future_encoder_commands"] += int(confidence > 0.1)
        return goal

    def hear(self, tape, reward):
        if not self.observations:
            return
        rows = {t: (x, q) for t, x, q, *_ in tape}
        end = self.observations[-1][0]
        dst = self.episodes.earned if reward else self.episodes.failed
        for t, k in self.observations:
            if t < end - 48 or t % 4 or t not in rows or t + LEAD not in rows:
                continue
            a, q = rows[t]
            b, other = rows[t + LEAD]
            if a is None or b is None or max(a[4], b[4]) >= -0.9:
                continue
            dst.append((k, other[:5] - q[:5]))
        self.observations.clear()

    def missed(self, tape):
        self.hear(tape, False)


def run(args):
    model = _read(MODEL.with_suffix(".json"))
    if model["model"] != _hash(MODEL):
        raise ValueError("Body readout changed")
    if set(model["release_ids"]) & {c["id"] for c in _read(season.PLAN)[args.deck]}:
        raise ValueError("Repeated fitting release")
    with np.load(MODEL) as z:
        episodes, inverse = Episodes(z["keys"], z["delta"]), z["inverse"]

    def build(*a, **kw):
        return Player(*a, episodes=episodes, inverse=inverse, **kw)

    with patch.object(season, "Player", build), patch.object(course, "Carried", Memory):
        season.run(args)
    d = _read(args.out)
    d.update(
        model_sha256=_hash(MODEL),
        earned=[(k.tolist(), v.tolist()) for k, v in episodes.earned],
        failed=[(k.tolist(), v.tolist()) for k, v in episodes.failed],
    )
    _write(args.out, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "live"))
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    parser.set_defaults(
        deck="practice", mode="live", carry=season.CARRY, cap=240, resume=False
    )
    args = parser.parse_args()
    train() if args.phase == "train" else run(args)
