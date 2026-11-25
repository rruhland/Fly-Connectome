"""Search direct joint variations for movements followed by two successful physical returns."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri

from . import tt46_continuous_season as season
from . import tt49_sustain_value as value
from .tt4_transfer_trial import _hash, _read, _write

mpc = season.course.mpc


class Player(value.Player):
    def begin(self):
        super().begin()
        self.previous_attempt = -999
        self.directions = ndtri(
            qmc.Sobol(10, scramble=True, seed=490211).random_base2(4).clip(0.025, 0.975)
        )

    def search(self, tick, x, q, ref, base):
        super().search(tick, x, q, ref, base)
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        n = self.notes[-1]
        if (
            n.get("robust", 0) < 1
            or self.tick != tick
            or self.best is None
            or tick - self.previous_attempt < 45
            or n["contact_delay_s"] < 0.065
            or not -1.35 < x[1] < -0.60
        ):
            return
        pos = self.optical_position
        original = value.predict(
            self.value, np.r_[n["target_errors"], pos, n["robust"], n["score"]]
        )[0]
        if original[1] > 0.945 or original[0] < 0.87:
            return
        self.previous_attempt = tick
        parent = self.best.copy()
        goals = [mpc.structured(parent, noise, scale=0.42) for noise in self.directions]
        loss, legal, _, effects, _ = self.trial(goals, long=True)
        indices = np.flatnonzero(legal)
        self.diag["sustain_generation_attempts"] += 1
        if not len(indices):
            return
        e = effects[indices, 3:10]
        guess = value.predict(
            self.value,
            np.c_[
                e,
                np.tile(pos, (len(e), 1)),
                np.full(len(e), n["robust"]),
                loss[indices],
            ],
        )
        ids = indices[np.argsort(-guess[:, 1])[:8]]
        votes, quality = self.uncertainty([goals[i] for i in ids], loss[ids])
        e = effects[ids, 3:10]
        actual = value.predict(
            self.value, np.c_[e, np.tile(pos, (len(e), 1)), votes, quality]
        )
        supported = self.support.query(e / value.fresh.shot.SCALE, k=8)[0][:, -1] < 1.65
        allowed = (
            supported
            & (votes >= n["robust"])
            & (quality <= n["score"] + 0.65)
            & (actual[:, 0] >= original[0] - 0.008)
            & (actual[:, 1] > original[1] + 0.035)
        )
        if not allowed.any():
            return
        j = np.argmax(
            np.where(
                allowed,
                actual[:, 1] - 0.026 * np.maximum(0.0, quality - n["score"]),
                -np.inf,
            )
        )
        i = ids[j]
        self.best, self.tick, self.lease = goals[i], tick, mpc.LEASE
        n.update(
            sustain_generated=dict(
                previous=n["target_errors"],
                before=original.tolist(),
                after=actual[j].tolist(),
            ),
            chosen="sustain_generated",
            robust=int(votes[j]),
            score=float(quality[j]),
            contact_delay_s=float(effects[i, 1]),
            target_errors=effects[i, 3:10].tolist(),
        )
        self.diag["sustain_generated_choices"] += 1


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
    doc = _read(value.CHECKPOINT)
    if set(doc["source_ids"]) & {c["id"] for c in _read(season.PLAN)[a.deck]}:
        raise ValueError("Repeated training release")
    with patch.object(season, "Player", Player):
        season.run(a)
    result = _read(a.out)
    result.update(
        outcome_model=_hash(value.CHECKPOINT), generator=_hash(Path(__file__))
    )
    _write(a.out, result)
