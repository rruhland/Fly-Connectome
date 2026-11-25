"""Exercise live hit/miss updates on the next three disjoint practice points."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt43_motor_modes as modes
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt39_carry_joints import Carried


class PracticingMotor(modes.MotorMPC):
    def missed(self, archive):
        # On a floor/out miss the tracker may continue tens of frames behind
        # the arm. It still supplies the subsequently issued command history,
        # but these frames do not anchor a negative *reachable* action window.
        eligible = [
            (t, x if x is None or x[4] >= -.9 or x[1] > -1.45 else None, q, ref, sent)
            for t, x, q, ref, sent in archive
        ]
        super().missed(eligible)


def run(a):
    plan = _read(course.PLAN)
    carry = _read(a.carry)
    practiced = [c["id"] for c in carry["cards"][:len(carry["rows"])]]
    cards = [c for c in plan["practice"] if c["id"] not in practiced][:3]
    ready = _read(mpc.BANK.with_suffix(".json"))["ready"]
    moves = mpc.measured_launches(a.feeds)
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "ready": ready, "cards": cards,
        "boundary": "Next three practice releases after the 24-game boundary, six alternating returns cap, intentionally short update exercise. Independent from the held study.",
        "carry_sha256": _hash(a.carry), "plan_sha256": _hash(course.PLAN),
        "mode": "live", "cap": 6, "rows": [],
    }
    with mpc.centered(ready):
        arena = mpc.arena()
        camera, rig, stride, lo, hi = mpc.resources(arena)
        with patch.object(mpc, "Fragments", lambda file, engine: Carried(
            file, engine, practice=[course.PRACTICE], held={c["id"] for c in cards}
        )):
            model = PracticingMotor(rescue=True, bank=mpc.BANK, raw=raw, variant="elastic")
        try:
            course.resume(model, modes.path_for(a.carry), carry)
            for card in cards:
                prior_misses = len(model.readout.misses)
                prior_launches = arena.launches
                new = mpc.play(arena, camera, rig, stride, lo, hi, ready, card, moves[card["id"]],
                               raw, model, course.OPTIC, "live", doc["cap"])
                if arena.launches - prior_launches != 1:
                    raise ValueError("Practice point did not launch exactly once")
                for leg, move in enumerate(new["placements"]["requested"]):
                    if list(draw(card, leg, "random", None)[0]) != move["target"]:
                        raise ValueError("Practice point used a different opponent draw")
                new["miss_sample_counts"] = [prior_misses, len(model.readout.misses)]
                doc["rows"].append(new)
                doc["checkpoint_sha256"] = modes.checkpoint(model, modes.path_for(a.out))
                doc["complete"] = len(doc["rows"]) == len(cards)
                _write(a.out, doc)
                print(card["id"], new["score"]["learner"],
                      new["planner"].get("credited_fragments", 0), new["miss_sample_counts"], flush=True)
        finally:
            model.close()
            camera.close()
            arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--carry", type=Path, default=modes.CARRY)
    p.add_argument("--feeds", type=Path, default=modes.FEEDS)
    run(p.parse_args())
