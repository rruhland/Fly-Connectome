"""Grow the practice partner's continuously sampled range as an arm earns returns."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_opponent as opponent
from . import tt42_learning_course as course
from .tt4_transfer_trial import _read, _write

ORIGINAL = opponent.draw
NARROW = np.array([[0.12, 0.25], [-1.34, -1.22], [1.057, 1.105], [3.26, 3.75]])
WIDTH = (0.0, 0.33, 0.67, 1.0)


class Partner:
    def __init__(self, advance):
        self.advance = advance
        self.stage = 0
        self.history = []
        self.states = {}
        self.logs = []

    def goal(self, c, leg, protocol, rng):
        raw, name = ORIGINAL(c, leg, protocol, rng)
        key = (c["id"], c["seed"], c["reply_seed"])
        stage = self.states.setdefault(key, self.stage if self.advance else 3)
        wide = np.array(opponent.PROFILES["reach"])
        if stage == 3:
            return raw, name
        bounds = NARROW + (wide - NARROW) * WIDTH[stage]
        fraction = (raw - wide[:, 0]) / (wide[:, 1] - wide[:, 0])
        return tuple(
            bounds[:, 0] + fraction * (bounds[:, 1] - bounds[:, 0])
        ), f"practice_{stage}"

    def after(self, r):
        offers = [s for s in r["placements"]["delivered"] if s.get("near_bounce")]
        self.history.extend(int(s["near_landed"]) for s in offers)
        self.logs.append(
            {
                "stage": self.stage if self.advance else 3,
                "offers": len(offers),
                "returned": sum(int(s["near_landed"]) for s in offers),
            }
        )
        if (
            self.advance
            and self.stage < 3
            and len(self.history) >= 24
            and np.mean(self.history[-24:]) >= 0.8
        ):
            self.stage += 1
            self.history.clear()


def run(a):
    teacher = Partner(a.kind == "graduated")
    play = course.mpc.play

    def play_stage(*args, **kw):
        row = play(*args, **kw)
        teacher.after(row)
        return row

    with (
        patch.object(course.mpc, "play", play_stage),
        patch.object(opponent, "draw", teacher.goal),
        patch.object(course, "draw", teacher.goal),
    ):
        course.run(a)
    d = _read(a.out)
    d["practice_curriculum"] = {
        "kind": a.kind,
        "stages": teacher.logs,
        "narrow_bounds": NARROW.tolist(),
        "fractions": WIDTH,
    }
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("graduated", "uniform"))
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(mode="live", deck="practice", cap=40, resume=False, carry=None)
    run(p.parse_args())
