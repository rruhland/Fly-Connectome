"""Retain earned joint movements while a practice partner samples challenging regions."""

import argparse
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_opponent as opponent
from .tt4_transfer_trial import _read, _write

ROOT = Path("artifacts")
OUT = ROOT / "tt41"
PLAN = ROOT / "tt39/endurance-plan.json"
ORDINARY = opponent.draw


class PracticePartner:
    """The practice coach sees its requested aim and the eventual public score."""

    def __init__(self):
        self.past = []
        self.seen = set()
        self.bands = np.array(opponent.PROFILES["reach"])
        self.bounds = self.bands[:, 1] - self.bands[:, 0]

    def draw(self, card, leg, protocol, rng):
        standard = ORDINARY(card, leg, protocol, rng)
        if len(self.past) < 8:
            return standard
        randomness = np.random.default_rng(
            [531687, card["seed"], card["reply_seed"], leg]
        )
        if randomness.random() < 0.38:
            return standard
        targets = randomness.uniform(self.bands[:, 0], self.bands[:, 1], (24, 4))
        x, y = zip(*self.past[-300:], strict=True)
        dist = (targets[:, None, :] - np.asarray(x)[None, :, :]) / self.bounds
        weights = np.exp(
            -8.0 * np.sum(dist**2 * np.array([1.0, 0.25, 1.0, 0.4]), axis=2)
        )
        success = (2.2 + weights @ np.asarray(y)) / (3 + weights.sum(axis=1))
        # Moderate difficulty retains some positive motor outcomes to imitate.
        logits = -abs(success - 0.58) * 22 + 0.45 / np.sqrt(1 + weights.sum(axis=1))
        p = np.exp(logits - logits.max())
        return tuple(
            targets[randomness.choice(len(targets), p=p / p.sum())]
        ), "curriculum"

    def record(self, game, placements):
        if game.card is None:
            return
        key = (game.card["id"], game.card["seed"], game.card["reply_seed"])
        if key not in self.seen:
            self.seen.add(key)
            self.past.extend(
                (v["target"], int(v["near_landed"])) for v in placements["delivered"]
            )


def clinic(mode, out, group, memory, start, count, *, options=()):
    argv = [
        "tt37_motor_clinic",
        mode,
        str(out),
        "--plan",
        str(PLAN),
        "--group",
        group,
        "--start",
        str(start),
        "--n",
        str(count),
        "--samples",
        "35",
        "--rounds",
        "3",
        "--depth",
        "-1.56",
        "--cut",
        ".18",
        "--guided",
        "--refine",
        "--economy",
        "--body",
        "--cap",
        "240",
        "--visual",
        str(ROOT / "tt37/rebound.npz"),
        "--rollin",
        str(ROOT / "tt37/self-combined.npz"),
        "--motor-value",
        str(ROOT / "tt37/carried-judgement.json.gz"),
        "--memory",
        str(memory),
        *options,
    ]
    with patch.object(sys, "argv", argv):
        runpy.run_module("scripts.experimental.tt37_motor_clinic", run_name="__main__")


def run(a):
    coach = PracticePartner()
    prior = ROOT / "tt37/independent-cold-value.json.gz"
    train = OUT / f"curriculum-{a.mode}-practice.json.gz"
    evaluate = OUT / f"curriculum-{a.mode}-assessment.json.gz"
    if not train.exists() or len(_read(train)["rows"]) != a.n:
        record = opponent.LiveArena.point_placements

        def finish(game):
            p = record(game)
            coach.record(game, p)
            return p

        with (
            patch.object(
                opponent, "draw", coach.draw if a.mode == "adaptive" else ORDINARY
            ),
            patch.object(opponent.LiveArena, "point_placements", finish),
        ):
            clinic("optical-learn", train, "practice", prior, 0, a.n)
        d = _read(train)
        d.update(coach_mode=a.mode, exposures=len(coach.past))
        _write(train, d)
    if opponent.draw is not ORDINARY:
        raise ValueError("Practice distribution leaked into assessment")
    clinic("optical-fixed", evaluate, "assessment", train, 12, 12)
    d = _read(evaluate)
    for c, r in zip(d["cards"], d["rows"], strict=True):
        for i, g in enumerate(r["placements"]["requested"]):
            expected, _ = ORDINARY(c, i, "random", None)
            if list(expected) != g["target"]:
                raise ValueError(
                    "Assessment did not use the unmodified continuous opponent"
                )
    d["practice_mode"] = a.mode
    _write(evaluate, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("adaptive", "uniform"))
    p.add_argument("--n", type=int, default=24)
    run(p.parse_args())
