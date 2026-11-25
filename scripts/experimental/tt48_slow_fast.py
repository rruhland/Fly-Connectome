"""Retain a stable motor repertoire and consult additional practice when it can rescue weak decisions."""

import argparse
from ast import literal_eval
from collections import deque
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt46_continuous_season as season
from . import tt47_season_retention as retention
from . import tt48_hazard as hazard
from .tt4_transfer_trial import _hash, _read, _write


class Player(season.Player):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        parent = _read(season.CARRY)
        p = season.CARRY.with_suffix("").with_suffix(".npz")
        if _hash(p) != parent["checkpoint_sha256"]:
            raise ValueError("Changed stable motor bank")
        with np.load(p) as z:
            self.stable = tuple(
                zip(z["states"], z["goals"], map(literal_eval, z["tags"]), strict=True)
            )
        self.hazard = tuple(map(np.array, _read(hazard.FIT)["coefficients"]))

    def search(self, tick, x, q, ref, base):
        recent = self.priors.online
        # New reward in the current point is immediately usable. Experience
        # from prior points persists in 'recent' and can supply an alternative.
        same_point = [r for r in recent if r[2][:2] == ("live", self.serial)]
        old = deque(self.stable, maxlen=recent.maxlen)
        old.extend(same_point)
        before = (self.best, self.tick, self.lease)
        random_before = deepcopy(self.rng.bit_generator.state)
        self.priors.online = old
        try:
            super().search(tick, x, q, ref, base)
        finally:
            self.priors.online = recent
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        first = self.notes[-1]
        if not hazard.eligible(first):
            return
        probability = float(
            hazard.predict(self.hazard, hazard.features(self.notes[-2:]))[0]
        )
        if probability < 0.62:
            return
        candidate = (self.best, self.tick, self.lease)
        state = deepcopy(self.rng.bit_generator.state)
        self.best, self.tick, self.lease = before
        self.rng.bit_generator.state = random_before
        try:
            super().search(tick, x, q, ref, base)
        finally:
            self.rng.bit_generator.state = state
        if self.notes[-1] is first:
            self.best, self.tick, self.lease = candidate
            return
        extra = self.notes.pop()
        if extra.get("robust", 0) > 0 and self.tick == tick:
            self.notes[-1] = dict(
                extra,
                stable_chosen=first.get("chosen"),
                stable_risk=probability,
                additional_practice_selected=True,
            )
            self.diag["additional_practice_selected"] += 1
        else:
            self.best, self.tick, self.lease = candidate
        self.diag["additional_practice_consulted"] += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("live", "frozen"))
    p.add_argument("carry", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    args = p.parse_args()
    args.parent = season.CARRY
    args.deck = "assessment" if args.mode == "frozen" else "practice"
    args.cap = 240
    args.resume = False
    args.player = "standard"
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with patch.object(season, "Player", Player):
        if args.mode == "frozen":
            retention.run(args)
        else:
            if args.carry != season.CARRY:
                raise ValueError("Live comparison starts with the common original bank")
            season.run(args)
    d = _read(args.out)
    d.update(
        stable=_hash(season.CARRY), hazard=_hash(hazard.FIT), code=_hash(Path(__file__))
    )
    _write(args.out, d)
