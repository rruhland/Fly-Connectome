"""Use each real practice outcome to choose the difficulty of subsequent physical feeds."""

import argparse
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_opponent as opponent
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

course = season.course
ROOT = Path("artifacts/tt53_responsive_school")
BOUNDS = np.asarray(opponent.PROFILES["reach"])
AXES = (0, 2, 3)
BINS = np.array([5, 2, 2])
CELLS = np.array(list(product(*(range(n) for n in BINS))))
DRAW = opponent.draw


def cell_of(goal):
    fraction = (np.asarray(goal) - BOUNDS[:, 0]) / np.ptp(BOUNDS, axis=1)
    bins = np.clip(np.floor(fraction[list(AXES)] * BINS).astype(int), 0, BINS - 1)
    return np.ravel_multi_index(bins, tuple(BINS))


class School:
    def __init__(self, adaptive, budget, earlier):
        self.adaptive = adaptive
        self.budget = budget
        self.offers = 0
        self.record = np.zeros((len(CELLS), 2))
        self.draws = {}
        for r in earlier:
            self.observe(r)

    def observe(self, row):
        for d in row["placements"]["delivered"]:
            if d.get("near_bounce"):
                self.record[cell_of(d["target"]), int(bool(d["near_landed"]))] += 1

    def probabilities(self):
        n = self.record.sum(axis=1)
        prior = (2 + self.record[:, 1].sum()) / (4 + n.sum())
        rate = (self.record[:, 1] + 6 * prior) / (n + 6)
        # Seek attainable mistakes, then harder offers. Always retain ordinary
        # full-range play and support for little-practiced regions.
        target = 0.82 - 0.15 * min(1.0, self.offers / (0.65 * self.budget))
        weight = np.exp(-0.5 * ((rate - target) / 0.09) ** 2) / np.sqrt(n + 8)
        return 0.3 / len(n) + 0.7 * weight / weight.sum()

    def draw(self, card, leg, protocol, rng):
        key = (card["seed"], card["reply_seed"], leg)
        if key in self.draws:
            return self.draws[key]
        raw, name = DRAW(card, leg, protocol, rng)
        if not self.adaptive:
            return raw, name
        fraction = (np.array(raw) - BOUNDS[:, 0]) / np.ptp(BOUNDS, axis=1)
        picker = np.random.default_rng([*key, 532091])
        cell = CELLS[picker.choice(len(CELLS), p=self.probabilities())]
        fraction[list(AXES)] = (cell + fraction[list(AXES)]) / BINS
        reply = tuple(BOUNDS[:, 0] + fraction * np.ptp(BOUNDS, axis=1))
        self.draws[key] = (reply, "responsive_practice")
        return self.draws[key]


class BudgetReached(Exception):
    pass


def run(a):
    ROOT.mkdir(parents=True, exist_ok=True)
    a.out = ROOT / f"{a.kind}-{a.mode}.json.gz"
    parent = _read(season.CARRY)
    school = School(a.kind == "responsive", a.budget, parent["rows"])
    original_exchange = opponent.LiveArena.exchange
    original_play = course.mpc.play
    coverage = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    history = []

    def make(*args, **kw):
        actor = season.Player(*args, coverage=coverage, **kw)
        ordinary_keep = actor.priors.keep

        def keep(*args):
            return ordinary_keep(*args) if school.offers <= school.budget else 0

        actor.priors.keep = keep
        return actor

    def exchange(arena, *args):
        result = original_exchange(arena, *args)
        for event in result[-1]["events"]:
            if event["kind"] == "legal_return" and event["side"] == -1:
                school.offers += 1
        return result

    def play(*args):
        if school.offers >= school.budget:
            raise BudgetReached()
        before = school.offers
        row = original_play(*args)
        if school.offers - before != row["score"]["opponent"]:
            raise ValueError("Practice exposure disagrees with the referee")
        history.append(
            dict(
                offers_before=before,
                offers_after=school.offers,
                returns=row["score"]["learner"],
            )
        )
        school.observe(row)
        return row

    a.carry = season.CARRY
    if a.mode == "live":
        a.n, a.deck = 128, "practice"
        try:
            with (
                patch.object(course, "PLAN", season.PLAN),
                patch.object(course.mpc, "MPC", make),
                patch.object(course.mpc, "play", play),
                patch.object(opponent.LiveArena, "exchange", exchange),
                patch.object(opponent, "draw", school.draw),
                patch.object(course, "draw", school.draw),
            ):
                course.run(a)
        except BudgetReached:
            pass
        if school.offers < a.budget:
            raise ValueError("Practice ended before reaching the exposure budget")
        d = _read(a.out)
        d["exposure"] = dict(
            budget=a.budget,
            total=school.offers,
            games=history,
            parent_serial=len(parent["rows"]),
            cells=school.record.tolist(),
        )
        _write(a.out, d)
    else:
        a.carry = ROOT / f"{a.kind}-live.json.gz"
        a.deck, a.n = "assessment", 24
        training = _read(a.carry)
        if training["exposure"]["budget"] != a.budget:
            raise ValueError("Practice budget mismatch")
        restore = course.resume

        def full_serial(actor, path, d):
            restore(actor, path, dict(d, rows=[None] * len(parent["rows"]) + d["rows"]))

        with (
            patch.object(course, "PLAN", season.PLAN),
            patch.object(course, "resume", full_serial),
            patch.object(course.mpc, "MPC", make),
        ):
            course.run(a)
        d = _read(a.out)
        d["practice_document"] = _hash(a.carry)
        _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("ordinary", "responsive"))
    p.add_argument("mode", choices=("live", "frozen"))
    p.add_argument("--budget", type=int, default=192)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(resume=False, cap=240)
    run(p.parse_args())
