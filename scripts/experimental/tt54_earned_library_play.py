"""Supply sensor-supported direct motor examples from physically earned low and central returns."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt46_continuous_season as season
from . import tt52_practice_dose as dose
from . import tt54_earned_low_flights as data
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

mpc = season.course.mpc
ROOT = data.ROOT
BANK = ROOT / "context-motors.npz"


def fit():
    doc = _read(ROOT / "central-templates.json")
    extra = _read(ROOT / "central-earned-replay.json.gz")
    if not doc["complete"] or not all(
        v["clearance"]["penetration_samples"] == 0 for v in extra["rows"]
    ):
        raise ValueError("Unverified earned motor examples")
    groups, states, goals = [], [], []

    def add(row, ident):
        logs = row["approach"]
        t0 = row["template"]["source_phase_tick"]
        for i, r in enumerate(logs):
            if (
                not t0 - 32 <= r["tick"] <= t0 - 6
                or (r["tick"] - t0) % 2
                or r["x"] is None
                or r["x"][4] >= -1
            ):
                continue
            states.append(
                mpc.query(np.array(r["x"]), np.array(r["q"]), np.array(r["ref"]))
            )
            goals.append(
                mpc.tail(
                    np.array([s["next_ref"] for s in logs[i : i + mpc.HORIZON]]),
                    mpc.HORIZON,
                )
            )
            groups.append(ident)

    for r in doc["rows"]:
        add(r, r["case"]["release"])
    for r in extra["templates"]:
        add(r, r["release"])
    group = np.array(groups)
    states = np.asarray(states)
    # Estimate reference support by matching each scored optical state with
    # examples from different original releases. Unknown bodily situations
    # fall through to the existing joint learner.
    diffs = np.linalg.norm(states[:, None] - states[None, :], axis=2)
    diffs[group[:, None] == group[None, :]] = np.inf
    limit = float(np.quantile(diffs.min(axis=1), 0.90))
    np.savez_compressed(BANK, states=states, goals=goals, groups=group, limit=limit)
    _write(
        ROOT / "context-motors.json",
        dict(
            checksum=_hash(BANK),
            samples=len(states),
            releases=np.unique(group).tolist(),
            distance_limit=limit,
            sources=[
                _hash(ROOT / "central-templates.json"),
                _hash(ROOT / "central-earned-replay.json.gz"),
            ],
        ),
    )


class Options(season.course.Carried):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        with np.load(BANK) as z:
            self.tree_extra = cKDTree(z["states"])
            self.goals = z["goals"]
            self.groups = z["groups"]
            self.radius = float(z["limit"])
        self.offered = 0
        self.current = set()

    def priors(self, x, q, reference, limit):
        original = super().priors(x, q, reference, limit)
        k = mpc.query(x, q, reference)
        d, ids = self.tree_extra.query(k, k=min(45, len(self.groups)))
        used = set()
        for dist, i in zip(d, ids, strict=True):
            if dist > self.radius or len(used) >= 2:
                break
            if self.groups[i] in used:
                continue
            used.add(self.groups[i])
            g = self.goals[i].copy()
            original.append((g, True))
            self.current.add(g.tobytes())
            self.offered += 1
        return original


class Player(season.Player):
    def search(self, tick, *args):
        self.priors.current.clear()
        previous = self.priors.offered
        super().search(tick, *args)
        options = self.priors.offered - previous
        self.diag["rare_direct_proposals"] += options
        if options and self.tick == tick and self.best is not None:
            self.diag["unmodified_rare_selected"] += int(
                self.best.tobytes() in self.priors.current
            )


def play(a):
    plan = _read(dose.PLAN)
    bank = _read(ROOT / "context-motors.json")
    if _hash(BANK) != bank["checksum"] or set(bank["releases"]) & {
        c["id"] for c in plan["practice"]
    }:
        raise ValueError("Qualification release overlaps rewarded examples")
    scorer = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    a.out = ROOT / "context-games.json.gz"
    with (
        enabled(),
        patch.object(season.course, "PLAN", dose.PLAN),
        patch.object(season.course, "Carried", Options),
        patch.object(
            mpc, "MPC", lambda *args, **kw: Player(*args, coverage=scorer, **kw)
        ),
    ):
        season.course.run(a)
    d = _read(a.out)
    d["extra_bank"] = _hash(BANK)
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode_name", choices=("fit", "play"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live", deck="practice", carry=season.CARRY, cap=240, resume=False
    )
    a = p.parse_args()
    fit() if a.mode_name == "fit" else play(a)
