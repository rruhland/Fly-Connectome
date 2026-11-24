"""Protect a first-pass movement repertoire during retrieval of later scored corrections."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_reuse as retrieval
from . import tt41_practice_curriculum as course
from .tt4_transfer_trial import _hash, _read, _write

PREVIOUS = course.ROOT / "tt39/endurance-04.json.gz"
RECENT = course.ROOT / "tt39/endurance-08.json.gz"
ORIGINAL = retrieval.propose


class Consolidation:
    def __init__(self, additional):
        self.additional = additional

    def propose(self, engine, bank, x, fit, eta, dt, home):
        incumbent, score, receipt = ORIGINAL(engine, bank, x, fit, eta, dt, home)
        model = engine.motor_value
        prior_probability = 0.0 if score is None else float(model.predict(score)[0, 0])
        # Reserve the earlier physiological solution when its private flight and
        # contact margins are comfortable. Learning in the subsequent pass can
        # supply responses where that repertoire remains unreliable.
        receipt["prior_probability"] = prior_probability
        receipt["later_selected"] = False
        if score is not None and prior_probability >= 0.80 and score[12] < 0.052:
            return incumbent, score, receipt
        extra, other, check = ORIGINAL(engine, self.additional, x, fit, eta, dt, home)
        for key in ("trials", "legal", "contacts"):
            receipt[key] += check[key]
        if other is None:
            return incumbent, score, receipt
        improved = score is None or (
            model.predict(other)[0, 0] > prior_probability + 0.10
            and model.cost(np.array([other]))[0] < model.cost(np.array([score]))[0]
        )
        if improved:
            receipt["later_selected"] = True
            return extra, other, receipt
        return incumbent, score, receipt


def run(a):
    past, current = (_read(p)["bank"] for p in (PREVIOUS, RECENT))
    if current[: len(past)] != past:
        raise ValueError(
            "Later practice no longer contains the first-pass motor experience"
        )
    if a.mode == "consolidated":
        policy = Consolidation(current[len(past) :])
        with patch.object(retrieval, "propose", policy.propose):
            course.clinic("optical-fixed", a.out, "assessment", PREVIOUS, a.start, a.n)
    else:
        source = PREVIOUS if a.mode == "previous" else RECENT
        course.clinic("optical-fixed", a.out, "assessment", source, a.start, a.n)
    d = _read(a.out)
    d["retrieval"] = {
        "mode": a.mode,
        "sources": {p.name: _hash(p) for p in (PREVIOUS, RECENT)},
    }
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("consolidated", "previous", "recent"))
    p.add_argument("out", type=Path)
    p.add_argument("--start", type=int, default=12)
    p.add_argument("--n", type=int, default=12)
    run(p.parse_args())
