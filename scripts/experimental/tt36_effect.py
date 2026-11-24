"""Value a visually observed outgoing ball by its subsequent reward, then learn local joint changes."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, resources
from .tt34_latency import PUPIL
from .tt36_course import OUT
from .tt36_hit import activation

PRACTICE = OUT / "hit-practice.json.gz"


def before_change(optical):
    return optical is not None and optical[1] < -0.50


def contexts(feeds, doc):
    model = _read(PUPIL)
    launches = measured_launches(feeds)
    observations = []
    with centered(model["ready"]):
        arena = replay_arena([])
        stereo, rig, stride, lo, hi = resources(arena)
        try:
            for card in doc["cards"]:
                arena.launch(launches[card["id"]], card=card)
                arena.partner.reset(seed=card["seed"])
                eye = Vision(rig, arena.control_dt, raw=True)
                actor = JointPolicy(model, stride, lo, hi)
                for i in range(360):
                    q = arena.joints.copy()
                    x = eye.update(i, stereo.capture() if i % 2 == 0 else None, q)
                    if before_change(x):
                        observations.append(np.r_[x, q[:5]].tolist())
                        break
                    _, _, done, cut, _ = arena.exchange(
                        actor.command(x, q), FAR_COMMAND
                    )
                    if done or cut:
                        raise ValueError(
                            "No pre-intervention observation for a practice opening"
                        )
        finally:
            stereo.close()
            arena.close()
    return observations


def vision_samples(doc):
    states = []
    labels = []
    releases = []
    for c, group in zip(doc["cards"], doc["rounds"], strict=True):
        for row in group:
            # The launch reward precedes every near stroke. Only subsequent public score
            # pulses can label an observed outgoing segment. Do not reuse an earlier pulse.
            pulses = [
                e["time"]
                for e in row["events"]
                if e["kind"] == "legal_return" and e["side"] == 1
            ][1:]
            outcomes = row["optical_outgoing"]
            for i, obs in enumerate(outcomes):
                start = obs["reversal_time"]
                stop = min(
                    start + 1.25,
                    outcomes[i + 1]["reversal_time"]
                    if i + 1 < len(outcomes)
                    else np.inf,
                )
                target = any(start < t < stop for t in pulses)
                states.append(obs["velocity"])
                labels.append(target)
                releases.append(c["id"])
    return np.asarray(states), np.asarray(labels, dtype=float), np.asarray(releases)


def expectation(x, y, v):
    scale = np.maximum(x.std(axis=0), [0.45, 0.65, 0.75])
    w = np.exp(-0.5 * np.sum(((x - v) / scale) ** 2, axis=1))
    return float((0.5 * y.mean() + w @ y) / (0.5 + w.sum()))


def fit(feeds):
    doc = _read(PRACTICE)
    x, y, ids = vision_samples(doc)
    forecasts = [
        expectation(x[ids != sid], y[ids != sid], v)
        for sid, v in zip(ids, x, strict=True)
    ]
    prior = [y[ids != sid].mean() for sid in ids]
    state = {
        "observations": contexts(feeds, doc),
        "source_sha256": _hash(PRACTICE),
        "outgoing": {
            "samples": len(y),
            "rewarded": int(y.sum()),
            "releases": len(set(ids)),
            "brier": float(np.mean((y - forecasts) ** 2)),
            "prior_brier": float(np.mean((y - prior) ** 2)),
            "labels": y.tolist(),
            "predictions": forecasts,
        },
        "utility": [],
        "score_only": [],
        "actions": doc["offsets_rad"],
        "training_ids": sorted({c["id"] for c in doc["cards"]}),
    }
    for card, rows in zip(doc["cards"], doc["rounds"], strict=True):
        keep = ids != card["id"]
        points = np.array([r["score"]["learner"] for r in rows])
        reward = (points > 0).astype(float) + 0.45 * np.clip(points - 1, 0, 3)
        # Treat missing optical labels as unknown, and preserve direct reward.
        sensory = np.array(
            [
                expectation(x[keep], y[keep], r["optical_first_turn"]["velocity"])
                if r["optical_first_turn"]
                else y[keep].mean()
                for r in rows
            ]
        )
        state["utility"].append(
            (reward - reward[0] + 0.4 * (sensory - sensory[0])).tolist()
        )
        state["score_only"].append((reward - reward[0]).tolist())
    _write(OUT / "effect-checkpoint.json", state)
    print(
        state["outgoing"]["brier"], state["outgoing"]["prior_brier"], state["utility"]
    )


class Effect:
    def __init__(self, state, score_only=False):
        self.x = np.array(state["observations"])
        self.scale = np.maximum(
            self.x.std(axis=0),
            [0.12, 0.18, 0.13, 0.65, 0.9, 1.05, 0.24, 0.24, 0.24, 0.24, 0.24],
        )
        self.y = np.asarray(state["score_only" if score_only else "utility"])
        self.actions = np.asarray(state["actions"])
        self.choice = None
        self.blind = 0

    def bias(self, x, q):
        if x is None:
            self.blind += 1
            if self.blind >= 7:
                self.choice = None
            return None
        self.blind = 0
        if x[1] > -0.32:
            self.choice = None
        if self.choice is None and before_change(x):
            dist = np.mean(((self.x - np.r_[x, q[:5]]) / self.scale) ** 2, axis=1)
            w = np.exp(-3 * dist)
            advantage = w @ self.y / (0.35 + w.sum())
            self.choice = int(np.argmax(advantage - [0, 0.06, 0.06, 0.06, 0.06]))
        return (
            None if self.choice is None else activation(x) * self.actions[self.choice]
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    fit(p.parse_args().feeds)
