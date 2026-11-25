"""Spend extra motor exploration only on incoming flights whose standard continuation usually fails."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

from . import tt42_learning_course as course
from . import tt46_continuous_season as season
from . import tt48_hazard as hazard
from .tt4_transfer_trial import _hash, _read, _write
from .tt42_commit_teacher import jitter, unique_indices

mpc = course.mpc
RANGES = np.array([0.28, 0.30, 0.27, 0.37, 0.49, 1.7, 2.2, 2.3, 2.7, 3.0, 0.039])


class Player(hazard.Player):
    def begin(self):
        super().begin()
        self.engineered = np.random.default_rng(480409)

    def recover(self, tick, x, q, reference, base, note):
        roots = [seq.copy() for seq, _ in self.priors.priors(x, q, reference, 9)]
        roots.append(np.tile(base, (mpc.HORIZON, 1)))
        if self.best is not None and 0 <= tick - self.tick < mpc.HORIZON:
            roots.insert(0, mpc.tail(self.best[tick - self.tick :], mpc.HORIZON))
        pool = list(roots)
        eta = mpc.HORIZON * mpc.DT * 0.72
        best_legal = []
        for stage in range(2):
            scale = 1.0 if not stage else 0.65
            z = qmc.Sobol(
                11, scramble=True, seed=int(self.engineered.integers(2**30))
            ).random_base2(6)
            for j, w in enumerate((2 * z - 1) * RANGES * scale):
                pool.append(jitter(roots[j % len(roots)], eta, w))
            loss, legal, _, _, feedback = self.trial(pool, long=True)
            ids = np.flatnonzero(legal)
            if len(ids):
                ids = unique_indices(np.array(pool), ids[np.argsort(loss[ids])], 6)
                votes, risk = self.uncertainty([pool[i] for i in ids], loss[ids])
                for i, v, c in zip(ids, votes, risk, strict=True):
                    if v > 0:
                        best_legal.append((int(v), float(c), pool[i].copy()))
                if best_legal:
                    break
            # Retain both the best failed moves and distinct contact outcomes.
            # A low point error alone can throw away a different promising departure.
            selected = unique_indices(np.array(pool), np.argsort(loss), 5)
            contacts = np.flatnonzero(feedback[:, 1] > 0)
            if len(contacts):
                effects = feedback[contacts, 2:5]
                for j in range(3):
                    for i in (np.argmin(effects[:, j]), np.argmax(effects[:, j])):
                        if contacts[i] not in selected:
                            selected.append(int(contacts[i]))
            roots = [pool[i] for i in selected]
            pool = list(roots)
        self.diag["difficult_offer_searches"] += 1
        if best_legal:
            v, c, seq = min(best_legal, key=lambda r: (-r[0], r[1]))
            self.best, self.tick, self.lease = seq, tick, mpc.LEASE
            note["difficult_offer_action"] = dict(
                stress_votes=v, modeled_cost=c, rounds=stage + 1
            )
            self.diag["difficult_offer_actions"] += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="practice", mode="live", carry=season.CARRY, cap=240, resume=False
    )
    args = p.parse_args()
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with patch.object(season, "Player", Player):
        season.run(args)
    receipt = _read(args.out)
    receipt.update(hazard=_hash(hazard.FIT), fallback=_hash(Path(__file__)))
    _write(args.out, receipt)
