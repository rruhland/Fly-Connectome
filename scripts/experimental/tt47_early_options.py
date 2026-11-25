"""Look for controllable joint prefixes while the inbound ball is still too far for the close search."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt45_composed_feedback as composed
from .tt4_transfer_trial import _hash, _read, _write

mpc = course.mpc


class Advance(composed.Composed):
    def begin(self):
        super().begin()
        self.next_preview = 0
        self.launches = []

    def step(self, tick, eye, x, q, reference, base):
        current = super().step(tick, eye, x, q, reference, base)
        if (
            x is None
            or x[4] > -0.9
            or x[1] < -0.48
            or x[1] > 0.44
            or tick < self.next_preview
        ):
            return current
        if self.engine.latest is None or tick * mpc.DT - self.engine.latest.time > 0.02:
            return current
        self.next_preview = tick + 7
        if not self.engine.sync():
            return current
        # Use only an observable crossing time to move the *second decision*
        # into the part of flight covered by the short motor memory. Actual
        # impact location, contact time and later opponent input are unspecified.
        span = int(np.clip(round((x[1] + 0.85) / max(1.9, -x[4]) / mpc.DT), 6, 17))
        hold = np.tile(current, (span + 1, 1))
        prefixes = [hold]
        for g, _ in self.priors.priors(x, q, reference, 6):
            if all(np.max(np.abs(g[:span] - row[:span])) > 0.04 for row in prefixes):
                prefixes.append(g[: span + 1].copy())
        tr = np.full((len(prefixes), span + 1, 17), np.nan)
        self.engine.trajectories(
            np.stack(prefixes), mpc.AIM, (span + 1) * mpc.DT, traces=tr
        )
        sets = [
            composed.compose(self.priors, [t], [g], span)
            for t, g in zip(tr, prefixes, strict=True)
        ]
        if not sets[0]:
            return current
        cost, legal = self.test(sets[0], landing=True)
        yes = np.flatnonzero(legal)
        if len(yes):
            ids = yes[np.argsort(cost[yes])[:2]]
            votes, _ = self.confirm([sets[0][i] for i in ids], cost[ids])
            if max(votes) >= 1:
                self.diag["early_standard_still_viable"] += 1
                return current
        candidates = [g for group in sets[1:] for g in group]
        if not candidates:
            return current
        cost, legal = self.test(candidates, landing=True)
        ok = np.flatnonzero(legal)
        self.diag["longer_trials"] += len(candidates)
        if not len(ok):
            return current
        ids = ok[np.argsort(cost[ok])[:5]]
        votes, rank = self.confirm([candidates[j] for j in ids], cost[ids])
        safe = np.flatnonzero(votes == 2)
        if len(safe):
            j = ids[safe[np.argmin(rank[safe])]]
            self.best = candidates[j]
            self.tick, self.lease = tick, min(span, 7)
            self.diag["early_physical_movements"] += 1
            self.launches.append(
                dict(tick=tick, observed=x.tolist(), span=span, choices=len(candidates))
            )
            return self.best[0]
        return current


def run(a):
    native = mpc.play

    def play(*args, **kw):
        r = native(*args, **kw)
        r["early_movements"] = args[10].launches
        return r

    with patch.object(mpc, "MPC", Advance), patch.object(mpc, "play", play):
        course.run(a)
    d = _read(a.out)
    d.update(early_options=_hash(Path(__file__)))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", default=8, type=int)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="frozen",
        deck="assessment",
        carry=course.ROOT / "learning-12.json.gz",
        cap=240,
        resume=False,
    )
    a = p.parse_args()
    a.out.parent.mkdir(exist_ok=True, parents=True)
    run(a)
