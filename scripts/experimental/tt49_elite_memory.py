"""Carry multiple direct-joint alternatives between optical updates on the same approaching object."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

mpc = course.mpc


def diverse(rows, limit):
    chosen = []
    for loss, g in sorted(rows, key=lambda r: r[0]):
        if all(np.max(np.abs(g[::3, :5] - s[::3, :5])) > 0.065 for _, s in chosen):
            chosen.append((loss, g))
            if len(chosen) == limit:
                break
    return chosen


class Player(season.Player):
    def begin(self):
        super().begin()
        self.alternatives = []
        self.last_search = -99
        self.capture = False

    def trial(self, goals, *, long=False, traces=False):
        r = super().trial(goals, long=long, traces=traces)
        if self.capture and not self.in_rank and (long or len(goals) >= self.budget):
            self.candidates.extend(
                (
                    float(r[0][i]) + (0.0 if long else 1.0),
                    np.asarray(goals[i])[: mpc.HORIZON].copy(),
                )
                for i in np.argsort(r[0])[:8]
            )
        return r

    def search(self, tick, x, q, ref, base):
        previous = self.alternatives
        elapsed = tick - self.last_search
        self.candidates = []
        self.capture = True
        try:
            super().search(tick, x, q, ref, base)
        finally:
            self.capture = False
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        self.alternatives = diverse(self.candidates, 6)
        self.last_search = tick
        note = self.notes[-1]
        if note.get("robust", 0) > 0 or not 1 <= elapsed <= 8 or not previous:
            return
        # The nominal sensor state and body have advanced, while the old full
        # plans assumed different earlier commands. Shift each plan, then retest
        # from the actual new camera/encoder state before issuing any part of it.
        goals = []
        for _, plan in previous:
            seq = mpc.tail(plan[elapsed:], mpc.HORIZON)
            goals.append(seq)
            lag = ref[:5] - seq[0, :5]
            alternate = seq.copy()
            alternate[:, :5] += np.maximum(0.0, 1.0 - np.arange(mpc.HORIZON) / 9)[
                :, None
            ] * np.clip(lag, -0.24, 0.24)
            goals.append(alternate)
        loss, _, tr, *_ = self.trial(goals, traces=True)
        for i in np.argsort(loss)[:4]:
            adjustment = self.transport(tr[i], face=True)
            if adjustment is not None:
                goals.append(goals[i] + adjustment)
        loss, legal, *_ = self.trial(goals, long=True)
        self.diag["rechecked_alternative_motions"] += len(goals)
        self.alternatives = diverse(
            self.alternatives
            + [(float(loss[i]), goals[i]) for i in np.argsort(loss)[:5]],
            6,
        )
        ix = np.flatnonzero(legal)
        if not len(ix):
            return
        ix = ix[np.argsort(loss[ix])[:5]]
        votes, risk = self.uncertainty([goals[i] for i in ix], loss[ix])
        use = np.flatnonzero(votes > 0)
        if len(use):
            j = use[np.argmin(risk[use] - 1.4 * votes[use])]
            self.best, self.tick, self.lease = goals[ix[j]], tick, mpc.LEASE
            note["retained_alternative"] = dict(
                votes=int(votes[j]), cost=float(risk[j]), age=elapsed
            )
            self.diag["retained_alternative_actions"] += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="practice", mode="live", carry=season.CARRY, cap=240, resume=False
    )
    a = p.parse_args()
    a.out.parent.mkdir(exist_ok=True, parents=True)
    with patch.object(season, "Player", Player):
        season.run(a)
    d = _read(a.out)
    d["elite_method"] = _hash(Path(__file__))
    _write(a.out, d)
