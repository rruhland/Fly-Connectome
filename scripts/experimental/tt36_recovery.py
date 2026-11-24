"""Learn how much arm posture to retain after an optically observed outgoing ball."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt35_replies import varied_opponent
from .tt36_course import OUT

BASE, EYE = joint.JointPolicy, joint.Vision
GAINS = (-0.6, 0.0, 0.6, 1.0)
PRACTICE = OUT.parent / "tt35/search-practice.json.gz"


class BothDirections(EYE):
    def from_flight(self, now, f):
        incoming = super().from_flight(now, f)
        if incoming is not None:
            return incoming
        if (
            f is not None
            and now - f.time < 0.105
            and f.velocity[1] > 0.9
            and f.rms < 2.4
            and f.depth_sigma < 0.16
        ):
            return np.r_[*f.advance(now)]
        return None


def trial(feeds, cards, seeds, choose, path):
    base = _read(PUPIL)
    shots = measured_launches(feeds)
    episodes = []

    class Recovery(BASE):
        def __init__(self, *args):
            super().__init__(*args)
            self.posture = self.home.copy()
            self.gap = 100
            self.hold = 0
            self.gain = None
            self.choices = []
            episodes.append(self.choices)

        def command(self, optical, q):
            if optical is not None and optical[4] < 0:
                self.posture = q.copy()
                self.gap = self.hold = 0
                return super().command(optical, q)
            self.gap += 1
            if optical is not None and self.gap <= 12:
                if self.gain is None:
                    state = np.r_[optical, self.posture[:5] - self.home[:5]]
                    self.gain = float(choose(state))
                    self.choices.append({"observed": state.tolist(), "gain": self.gain})
                self.hold = 18
            if self.hold and self.gain:
                self.hold -= 1
                desired = self.home + np.clip(
                    self.gain * (self.posture - self.home), -0.45, 0.45
                )
                return self.drive(desired, q, self.options["recovery"])
            return super().command(None, q)

    result = {
        "cards": cards,
        "ready": base["ready"],
        "opening_side": 1,
        "pupil_sha256": _hash(PUPIL),
        "feed_sha256": _hash(feeds),
        "reply_seeds": seeds,
        "rows": [],
    }
    with (
        centered(base["ready"]),
        patch.object(joint, "Vision", BothDirections),
        patch.object(joint, "JointPolicy", Recovery),
    ):
        arena = replay_arena([])
        stereo, rig, stride, lo, hi = joint.resources(arena)
        try:
            for card, seed in zip(cards, seeds, strict=True):
                context = (
                    varied_opponent(card["index"], seed)
                    if seed is not None
                    else nullcontext()
                )
                with context:
                    arena.cards = iter([card])
                    row = joint.point(
                        arena,
                        stereo,
                        rig,
                        base,
                        stride,
                        lo,
                        hi,
                        shots[card["id"]],
                        card,
                        cap=24,
                    )
                row["recovery"] = episodes[-1]
                result["rows"].append(row)
                _write(path, result)
        finally:
            stereo.close()
            arena.close()


def course():
    # This conditional study requires the arm to make a stroke before it can learn recovery.
    # Select only by previously archived nominal practice scores, before testing adjustments.
    cards = []
    for r in _read(PRACTICE)["rounds"]:
        cards.extend(
            c
            for c, p in zip(r["cards"], r["rows"][0], strict=True)
            if p["score"]["learner"]
        )
    return cards * 2, [None] * len(cards) + list(range(364090, 364090 + len(cards)))


def fit():
    docs = [_read(OUT / f"recovery-practice-{i}.json.gz") for i in range(len(GAINS))]
    contexts, rewards = [], []
    for rows in zip(*(d["rows"] for d in docs), strict=True):
        if not all(r["recovery"] for r in rows):
            continue
        x = np.array([r["recovery"][0]["observed"] for r in rows])
        if not np.allclose(x, x[0], atol=1e-7, rtol=0):
            raise ValueError("Intervention altered its starting optical state")
        contexts.append(x[0].tolist())
        rewards.append([max(0, r["score"]["learner"] - 1) for r in rows])
    _write(
        OUT / "recovery-checkpoint.json",
        {
            "gains": GAINS,
            "contexts": contexts,
            "later_returns": rewards,
            "sources": [
                _hash(OUT / f"recovery-practice-{i}.json.gz") for i in range(len(GAINS))
            ],
        },
    )
    print(np.sum(rewards, axis=0), len(contexts))


def selector(state):
    x = np.asarray(state["contexts"])
    y = np.asarray(state["later_returns"])
    scale = np.maximum(
        x.std(axis=0), [0.12, 0.22, 0.12, 0.85, 0.9, 0.9, 0.25, 0.25, 0.25, 0.25, 0.25]
    )

    def choose(current):
        w = np.exp(-0.5 * np.mean(((x - current) / scale) ** 2, axis=1))
        scores = w @ (y - y[:, 1, None])
        # Break ties in favor of original readiness. Penalize weakly supported changes.
        return GAINS[int(np.argmax(scores - 0.35 * np.abs(GAINS)))]

    return choose


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("practice", "fit", "evaluate"))
    p.add_argument("--gain-index", type=int, choices=range(4))
    p.add_argument("--cohort", choices=("new", "replies"))
    p.add_argument("--feeds", type=Path)
    a = p.parse_args()
    if a.mode == "practice":
        cards, seeds = course()
        trial(
            a.feeds,
            cards,
            seeds,
            lambda _: GAINS[a.gain_index],
            OUT / f"recovery-practice-{a.gain_index}.json.gz",
        )
    elif a.mode == "fit":
        fit()
    else:
        doc = _read(OUT / f"course-{a.cohort}.json.gz")
        trial(
            a.feeds,
            doc["cards"],
            doc.get("reply_seeds", [None] * len(doc["cards"])),
            selector(_read(OUT / "recovery-checkpoint.json")),
            OUT / f"recovery-{a.cohort}.json.gz",
        )
