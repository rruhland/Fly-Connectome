"""Prefer motor commands that still return the ball after small execution errors."""

import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np

from . import tt46_continuous_season as season
from .tt52_action_margin import variants
from .tt4_transfer_trial import _read, _write, _hash

mpc = season.course.mpc
ROOT = Path("artifacts/tt52_matched_movements")


class Player(season.Player):
    def __init__(self, *a, recover=False, **kw):
        super().__init__(*a, **kw)
        self.recover = recover
        self.rank_cache = None
        self.capture = False

    def uncertainty(self, goals, nominal):
        v, q = super().uncertainty(goals, nominal)
        if self.capture:
            self.rank_cache = (v.copy(), q.copy())
        return v, q

    def robust(self, goals, rank, nominal):
        self.capture = True
        try:
            winner, n, quality = super().robust(goals, rank, nominal)
        finally:
            self.capture = False
        if len(goals) < 2:
            return winner, n, quality
        votes, cost = self.rank_cache
        stock = int(np.flatnonzero(rank == winner)[0])
        eligible = np.flatnonzero((votes >= n) & (cost <= quality + 0.60))
        if len(eligible) < 2:
            return winner, n, quality
        perturb = [
            g for i in eligible for g in variants(goals[i], self.engine.reference)
        ]
        self.in_rank = True
        try:
            _, ok, *_ = self.trial(perturb, long=True)
        finally:
            self.in_rank = False
        counts = ok.reshape(len(eligible), 6).sum(axis=1)
        keep = int(np.flatnonzero(eligible == stock)[0])
        self.diag["motor_margin_comparisons"] += 1
        select = np.flatnonzero(counts >= counts[keep] + 2)
        if len(select):
            j = select[np.argmax(counts[select] - 0.16 * cost[eligible[select]])]
            i = eligible[j]
            self.diag["motor_margin_changes"] += 1
            return int(rank[i]), int(votes[i]), float(cost[i])
        return winner, n, quality

    def search(self, tick, x, q, ref, base):
        super().search(tick, x, q, ref, base)
        if not self.recover or not self.notes or self.notes[-1]["tick"] != tick:
            return
        note = self.notes[-1]
        if note.get("legal_trials", 0) or not -1.49 < x[1] < -0.56:
            return
        # Search the empirical timing/scale neighborhood of remembered commands.
        # These are the same direct joint commands offered by the ordinary actor.
        seeds = [s for s, _ in self.priors.priors(x, q, ref, 7)]
        if self.best is not None and 0 <= tick - self.tick < mpc.HORIZON - 2:
            seeds.insert(0, mpc.tail(self.best[tick - self.tick :], mpc.HORIZON))
        if not seeds:
            return
        options = [p for g in seeds for p in variants(g, ref)]
        dist, _, _, _, _ = self.trial(options)
        ix = np.argsort(dist)[: min(14, len(options))]
        loss, legal, _, effects, _ = self.trial([options[i] for i in ix], long=True)
        cand = np.flatnonzero(legal)
        self.diag["motor_margin_rescue_searches"] += 1
        if not len(cand):
            return
        use = cand[np.argsort(loss[cand])[:4]]
        agree, cost = self.uncertainty([options[ix[i]] for i in use], loss[use])
        if not np.any(agree):
            return
        j = int(np.argmin(np.where(agree > 0, cost, np.inf)))
        i = use[j]
        self.best, self.tick, self.lease = options[ix[i]], tick, mpc.LEASE
        self.diag["motor_margin_rescues_issued"] += 1
        note.update(
            chosen="motor_timing_neighbor",
            robust=int(agree[j]),
            score=float(cost[j]),
            contact_delay_s=float(effects[i, 1]),
            target_errors=effects[i, 3:10].tolist(),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--recover", action="store_true")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live", carry=season.CARRY, deck="practice", cap=240, resume=False
    )
    args = p.parse_args()
    with patch.object(
        season, "Player", lambda *a, **kw: Player(*a, recover=args.recover, **kw)
    ):
        season.run(args)
    doc = _read(args.out)
    doc.update(action_margin_code=_hash(Path(__file__)), recover=args.recover)
    _write(args.out, doc)
