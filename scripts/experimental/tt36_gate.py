"""Learn when to use the hit and miss joint updates from scored, matched practice."""

import argparse

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt34_results import paired
from .tt36_course import OUT

MODES = ("fixed", "miss", "hit", "both")


def plan():
    paths = [
        OUT.parent / p
        for p in ("tt34/body-novel-stock.json.gz", "tt35/confirm-nominal.json.gz")
    ]
    exclude = {
        c["id"]
        for name in ("new", "replies")
        for c in _read(OUT / f"course-{name}.json.gz")["cards"]
    }
    groups = [[], [], []]
    for p in paths:
        data = _read(p)
        for card, row in zip(data["cards"], data["rows"], strict=True):
            if card["id"] in exclude:
                continue
            incoming = row["placements"]["delivered"][0]
            group = (
                2
                if incoming["near_landed"]
                else int(incoming["near_contact"] is not None)
            )
            groups[group].append(card)
    rng = np.random.default_rng(361907)
    cards = [g[i] for g in groups for i in rng.permutation(len(g))[:6]]
    cards = [cards[i] for i in rng.permutation(len(cards))]
    _write(
        OUT / "course-practice.json.gz",
        {
            "cards": cards,
            "sources": [_hash(p) for p in paths],
            "selection": "Six previously missed, six contacted and faulted, six returned first offers. No held releases.",
        },
    )


def fit():
    paths = [OUT / f"paired-practice-{mode}.json.gz" for mode in MODES]
    docs = [_read(p) for p in paths]
    for d in docs:
        paired(docs[0], d)
    observations = []
    advantages = []
    for rows in zip(*(d["rows"] for d in docs), strict=True):
        if any(r["initial_observation"] is None for r in rows):
            continue
        x = np.asarray([r["initial_observation"] for r in rows])
        if not np.allclose(x, x[0], atol=1e-8, rtol=0):
            raise ValueError("A motor correction preceded the sensory context")
        # The opening and every following return each earn one reward. The capped
        # difference prevents a single long rally dominating neighboring contexts.
        reward = np.array([r["score"]["learner"] for r in rows])
        observations.append(x[0].tolist())
        advantages.append(np.clip(reward - reward[0], -2, 2).tolist())
    if not observations:
        raise ValueError("Missing incoming states")
    _write(
        OUT / "gate-checkpoint.json",
        {
            "observations": observations,
            "advantages": advantages,
            "modes": MODES,
            "scale": np.maximum(
                np.std(observations, axis=0),
                [0.15, 0.25, 0.15, 0.6, 0.8, 1.5, 0.3, 0.3, 0.3, 0.3, 0.3],
            ).tolist(),
            "sources": [_hash(p) for p in paths],
        },
    )
    print(np.sum(advantages, axis=0), len(observations))


class Gate:
    def __init__(self, checkpoint):
        self.x = np.asarray(checkpoint["observations"])
        self.scale = np.array(checkpoint["scale"])
        self.y = np.asarray(checkpoint["advantages"])
        self.mode = None

    def choose(self, optical, q):
        if self.mode is None and optical is not None:
            x = np.r_[optical, q[:5]]
            dist = np.mean(((self.x - x) / self.scale) ** 2, axis=1)
            weights = np.exp(-2 * dist)
            predicted = weights @ self.y / (1.0 + weights.sum())
            self.mode = MODES[int(np.argmax(predicted - [0, 0.025, 0.025, 0.05]))]
        return self.mode or "fixed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "fit"))
    args = p.parse_args()
    if args.mode == "plan":
        plan()
    else:
        fit()
