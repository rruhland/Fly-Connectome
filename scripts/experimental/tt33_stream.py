"""Train direct actuator offsets from legal-return reward on a stream of distinct physical games."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt31_curriculum import card
from .tt33_servo import READY, Station, distill

SIGMA = np.array([0.12, 0.10])


class Feedback:
    def __init__(self, seed, examples=None):
        self.rng = np.random.default_rng(seed)
        axes = self.rng.choice([-1.0, 1.0], size=(4, 5))
        self.offsets = np.vstack((np.zeros(5), 0.095 * axes, -0.095 * axes))
        self.examples = [] if examples is None else list(examples)

    def choose(self, x, z, learn):
        near = np.zeros(len(self.offsets))
        won = np.zeros_like(near)
        for a in self.examples:
            weight = float(
                np.exp(-0.5 * np.sum(np.square((np.array([x, z]) - a["obs"]) / SIGMA)))
            )
            near[a["action"]] += weight
            won[a["action"]] += weight * a["won"]
        if learn:
            scores = self.rng.beta(0.6 + won, 1.6 + near - won)
            i = int(np.argmax(scores))
        else:
            # An action must have earned actual reward before use in a new observation.
            scores = np.where(won > 0, (won + 0.6) / (near + 2.2), -1.0)
            i = int(np.argmax(scores))
        return self.offsets[i], i

    def update(self, row):
        # Credit the joint offset actually issued for the observed incoming flight.
        # Public own-return reward is identified after the far-side bounce.
        # Treat an approach truncated by the evaluation cap as censored.
        events = row["events"]
        choices = row["choices"]
        for i, c in enumerate(choices):
            end = choices[i + 1]["time"] if i + 1 < len(choices) else np.inf
            hit = next(
                (
                    e
                    for e in events
                    if e["kind"] == "contact"
                    and e["side"] == 1
                    and c["time"] < e["time"] < end
                ),
                None,
            )
            legal = hit is not None and any(
                e["kind"] == "legal_return"
                and e["side"] == 1
                and hit["time"] < e["time"] < end
                for e in events
            )
            if not legal and i == len(choices) - 1 and row["score"]["returns"] >= 12:
                continue
            self.examples.append(
                {"obs": [c["x"], c["z"]], "action": c["choice"], "won": int(legal)}
            )


def run(args):
    weights = distill()
    source = _read(args.course)
    if args.mode == "practice":
        cards = []
        # Every physical opponent seed is new. Both sides and the crossover
        # recur without replaying the same feed to compare alternative actions.
        for i in range(args.count):
            sid = (12330, 12451, 11909, 13140)[i % 4]
            mode = ("periodic", "full", "continuous")[(i // 4 + i) % 3]
            cards.append(card(sid, mode, i, 334910 + i))
    else:
        cards = source["cards"]
    prior = _read(args.carry)["examples"] if args.carry else []
    feedback = Feedback(33991, prior)
    ready = _read(READY)["ready"]
    station = Station(args.feeds, ready)
    result = {
        "mode": args.mode,
        "ready": ready,
        "opening_side": 1,
        "source_sha256": _hash(args.course),
        "carry_sha256": _hash(args.carry) if args.carry else None,
        "cards": cards,
        "rows": [],
        "offsets": feedback.offsets.tolist(),
        "distillation": weights,
    }
    try:
        for i, c in enumerate(cards):
            row = station.play(
                c,
                np.asarray(weights["weights"]),
                lambda x, z: feedback.choose(x, z, args.mode == "practice"),
                12,
            )
            if args.mode == "practice":
                feedback.update(row)
            result["rows"].append(row)
            result["examples"] = feedback.examples
            _write(args.output, result)
            print(
                i + 1,
                c["id"],
                row["score"]["learner"],
                [(q["choice"]) for q in row["choices"]],
                flush=True,
            )
    finally:
        station.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("practice", "evaluate"))
    for k in ("feeds", "course", "output"):
        p.add_argument(k, type=Path)
    p.add_argument("--count", type=int, default=48)
    p.add_argument("--carry", type=Path)
    run(p.parse_args())
