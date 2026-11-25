"""Explore direct motor adjustments after local failures, while protecting a previously successful response."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt45_error_learning as sensory
from . import tt46_context_search as school
from .tt4_transfer_trial import _hash, _read, _write

PRIOR = school.control.PRACTICE
BaseSearch = school.LocalSearch


class Selective(BaseSearch):
    def __init__(self, basis, learn=True):
        super().__init__(basis, learn)
        packet = _read(PRIOR)
        self.examples = [
            (np.array(row["context"]), bool(row["landed"]))
            for row in packet["rows"]
            if row["arm"] == 1
        ]

    def act(self, key):
        # No random movement on a familiar successful context unless the
        # arm has actually failed in a similarly observed situation.
        keys = [r[0] for r in self.examples] + [r[0] for r in self.rows]
        success = np.array(
            [r[1] for r in self.examples] + [r[2] > 0.9 for r in self.rows]
        )
        dist = sensory.dis(key, keys)
        ids = np.argsort(dist, kind="stable")[:18]
        w = np.exp(-4.0 * dist[ids] ** 2)
        loss = w @ (1.0 - success[ids])
        total = w.sum()
        hazard = (loss + 0.4) / (total + 2.0)
        if hazard < 0.16 or loss < 0.55:
            return np.zeros(3)
        # The current carried policy is zero perturbation. Test a departure
        # on at most alternate exposed attempts, leaving actual control data.
        if self.learn and self.rng.random() > 0.5:
            return np.zeros(3)
        action = super().act(key)
        if not self.learn:
            if not self.rows:
                return np.zeros(3)
            d = sensory.dis(key, [r[0] for r in self.rows])
            changed = np.array([np.linalg.norm(r[1]) > 0.035 for r in self.rows])
            close = (d < 0.70) & changed
            if np.count_nonzero(close) < 3:
                return np.zeros(3)
            rewards = np.array([r[2] for r in self.rows])
            if (
                np.average(rewards[close], weights=np.exp(-4 * d[close] ** 2))
                < 1.0 - hazard
            ):
                return np.zeros(3)
        return action


def run(a):
    packet = _read(PRIOR)
    if {r["release"] for r in packet["rows"]} & {
        c["id"] for c in _read(school.season.PLAN)[a.deck][: a.n]
    }:
        raise ValueError(
            "Validation releases overlap failures used to decide when to explore"
        )
    with patch.object(school, "LocalSearch", Selective):
        school.run(a)
    d = _read(a.out)
    d.update(failure_calibration=_hash(PRIOR), selectivity=_hash(Path(__file__)))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("live", "frozen"))
    p.add_argument("out", type=Path)
    p.add_argument("--deck", default="practice", choices=("practice", "assessment"))
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--policy", type=Path)
    p.add_argument("--carry", type=Path, default=school.season.CARRY)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(cap=240, resume=False)
    run(p.parse_args())
