"""Learn which direct actuator references worked in similar observed states, including misses."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt37_joint_deep as deep
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_visual_inertia import LearnedOptics

SENSOR_SCALE = deep.SCALES["local"]
ACTION_SCALE = np.array([0.12, 0.14, 0.17, 0.20, 0.21])


class ActionValue:
    def __init__(self, base, observations, *, learn=True, future_mix=0.0):
        self.future_mix = future_mix
        self.base = base
        self.learn = learn
        self.samples, self.offsets, self.scores = observations
        self.tree = cKDTree(self.samples / SENSOR_SCALE)
        self.added = []
        self.begin_point()

    def begin_point(self):
        self.flight = []
        self.score = 0
        self.diagnostics = {"changed": 0, "scored_strokes": 0, "missed_strokes": 0}

    def record_command(self, tick, x, q, ref):
        if x is None or x[4] >= -0.9:
            return
        if self.flight and tick - self.flight[-1][0] > 22:
            self.flight = []
        self.flight.append((tick, np.r_[x, q[:5]], ref[:5] - q[:5]))

    def reward(self, scored):
        if len(self.flight) >= 7:
            self.diagnostics["scored_strokes" if scored else "missed_strokes"] += 1
            if self.learn:
                if self.future_mix:
                    times = np.array([r[0] for r in self.flight])
                    for tick, x, u in self.flight[::2]:
                        j = np.searchsorted(times, tick + 6)
                        if j < len(times) and times[j] <= tick + 8:
                            _, newer, offset = self.flight[j]
                            future = newer[6:11] + offset - x[6:11]
                            self.added.append(
                                (x, u + self.future_mix * (future - u), float(scored))
                            )
                else:
                    self.added.extend(
                        (x, u, float(scored)) for _, x, u in self.flight[::2]
                    )
                # Index the short on-policy history separately so the offline reference set stays fixed.
                self.added = self.added[-6000:]
                if self.added:
                    self.recent = cKDTree(
                        np.array([r[0] for r in self.added]) / SENSOR_SCALE
                    )
        self.flight = []

    def score_feedback(self, score):
        if score > self.score:
            self.score = score
            self.reward(True)

    def end_point(self, *, terminal):
        if terminal:
            self.reward(False)
        self.diagnostics["stored_frames"] = len(self.added)

    def get(self, optical, q, *, reference=None):
        original, confidence = self.base.get(optical, q, reference=reference)
        if optical[4] >= -0.9:
            return original, confidence
        state = np.r_[optical, q[:5]] / SENSOR_SCALE
        distances, ix = self.tree.query(state, min(96, len(self.samples)))
        distances, ix = np.atleast_1d(distances), np.atleast_1d(ix)
        u = self.offsets[ix]
        score = self.scores[ix]
        if self.added:
            dr, ir = self.recent.query(state, min(40, len(self.added)))
            more = [self.added[j] for j in np.atleast_1d(ir)]
            distances = np.r_[distances, np.atleast_1d(dr)]
            u = np.vstack([u, [r[1] for r in more]])
            score = np.r_[score, [r[2] for r in more]]
        if distances.min() > 2.2 or not np.any(score):
            return original, confidence
        w = np.exp(
            -0.5
            * (
                distances
                / max(
                    0.30,
                    np.partition(distances, min(10, len(distances) - 1))[
                        min(10, len(distances) - 1)
                    ],
                )
            )
            ** 2
        )
        baseline = (original[:5] - q[:5]) if original is not None else np.zeros(5)
        pos = np.flatnonzero(score > 0)
        pos = pos[np.argsort(distances[pos])[:8]]
        movement = np.vstack([baseline, u[pos], 0.55 * u[pos] + 0.45 * baseline])
        # Compare candidate actions at the same visual/proprioceptive state. A broad prior shrinks
        # unmatched actions toward the region's observed success probability.
        kernel = np.exp(
            -0.5
            * np.mean(
                ((movement[:, None, :] - u[None, :, :]) / ACTION_SCALE) ** 2, axis=2
            )
        )
        weight = w * kernel
        marginal = (2 + w @ score) / (4 + w.sum())
        value = (1.5 * marginal + weight @ score) / (1.5 + weight.sum(axis=1))
        gain = (
            value
            - value[0]
            - 0.010 * np.mean(((movement - baseline) / ACTION_SCALE) ** 2, axis=1)
        )
        best = int(np.argmax(gain))
        if gain[best] <= 0.022:
            return original, confidence
        self.diagnostics["changed"] += 1
        # Bounded local residual learned entirely from outcomes of issued commands.
        adjusted = baseline + 0.88 * np.clip(movement[best] - baseline, -0.48, 0.48)
        return np.r_[q[:5] + adjusted, -0.1], max(
            confidence, min(1.0, (2.3 - distances.min()) / 0.8)
        )


def load_data(paths, *, future_mix=0.0):
    blocks = []
    seen = set()
    for path in paths:
        h = _read(path.with_suffix(".json"))
        if h["data_sha256"] != _hash(path):
            raise ValueError("Changed sensor dataset")
        seen.update(h["practice_ids"])
        with np.load(path) as z:
            x = z["states"]
            refs = z["references"]
            u = refs[:, :5] + future_mix * (refs[:, 6:11] - refs[:, :5]) - x[:, 6:11]
            y = z["reward"]
            blocks.append((x, u, y))
    return tuple(np.concatenate(b) for b in zip(*blocks, strict=True)), seen


def run(a):
    dataset, practiced = load_data(a.data, future_mix=a.lead)
    cards = _read(a.plan)[a.deck][: a.limit]
    practiced.update(
        r["card"]["id"] for r in _read(a.bank.with_suffix(".json"))["practice"]
    )
    if practiced & {c["id"] for c in cards}:
        raise ValueError("Training and assessment release IDs overlap")
    original = deep.Vision

    class CalibratedVision(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eye = LearnedOptics(self.eye, a.optics)

    players = []

    def wrap(field):
        if a.mode == "baseline":
            return field
        players.append(
            ActionValue(field, dataset, learn=a.mode == "learn", future_mix=a.lead)
        )
        return players[-1]

    with patch.object(deep, "Vision", CalibratedVision):
        deep.run(
            a.feeds,
            a.bank,
            a.plan,
            a.deck,
            "local",
            a.output,
            cap=200,
            limit=a.limit,
            field_hook=wrap,
            details={
                "source_sha256": {p.name: _hash(p) for p in a.data},
                "mode": a.mode,
                "future_fraction": a.lead,
                "optical_sha256": _hash(a.optics),
                "practice_ids": sorted(practiced),
            },
        )
    if players:
        ck = a.output.with_suffix("").with_suffix(".npz")
        examples = players[0].added
        np.savez_compressed(
            ck,
            states=np.asarray([r[0] for r in examples]).reshape(-1, 11),
            offsets=np.asarray([r[1] for r in examples]).reshape(-1, 5),
            reward=np.asarray([r[2] for r in examples]),
        )
        doc = _read(a.output)
        doc["online_checkpoint_sha256"] = _hash(ck)
        _write(a.output, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["baseline", "frozen", "learn"])
    p.add_argument("output", type=Path)
    p.add_argument("data", type=Path, nargs="+")
    p.add_argument(
        "--plan", type=Path, default=Path("artifacts/tt37/memory-course.json")
    )
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", default=24, type=int)
    p.add_argument(
        "--bank", type=Path, default=Path("artifacts/tt37/self-combined.npz")
    )
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--optics", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    p.add_argument("--lead", type=float, default=0.0)
    run(p.parse_args())
