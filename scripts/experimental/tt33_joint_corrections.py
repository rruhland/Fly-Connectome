"""Ask the optical teacher to take over from live pupil states during practice only."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory as motor
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_vision_twin import OpticalPlanner
from .tt31_curriculum import OUTPUT as TT31, play_card
from .tt31_ready import centered
from .tt31_recovery import Recovery
from .tt33_joint import JointPolicy, Vision


def roll_in(feeds, reference, checkpoint, output, *, steps):
    source = _read(reference)
    model = _read(checkpoint)
    ready = model["ready"]
    if source["ready"] != ready:
        raise ValueError("Practice and deployed ready poses differ")
    retained = TT31 / "ready-staged-checkpoint.json.gz"
    original = OpticalPlanner.command
    started = []
    observations = []
    pupils = {}
    result = dict(ready=ready, cards=[], rows=[], opening_side=1, requested_steps=steps,
                  training_source_sha256=_hash(reference), source_student_sha256=_hash(checkpoint))

    def command(teacher, tick, flight, q, dq, target):
        if teacher not in pupils:
            a = teacher.twin
            axes = a._arm_ids[1][2]
            stride = a.control_dt * teacher.nominal.rates
            pupil = JointPolicy(model, stride, a._joint_low[axes], a._joint_high[axes])
            vision = Vision(None, a.control_dt, raw=pupil.raw)
            pupils[teacher] = pupil, vision
        pupil, vision = pupils[teacher]
        pupil.reference = target.copy()
        feature = vision.from_flight(tick * teacher.twin.control_dt, flight)
        if len(started) < steps:
            candidate = pupil.command(feature, q)
            if feature is not None and np.max(abs(candidate)) > .025:
                started.append(tick)
                return candidate
            if feature is None or not len(started) or tick <= started[0] + steps + 4:
                return np.zeros_like(q)
        action = original(teacher, tick, flight, q, dq, target)
        if len(started) and np.max(abs(action)) > .025:
            observations.append(tick)
        return action

    with centered(ready), patch.object(motor, "RallyObjective", Recovery), TemporaryDirectory() as tmp:
        for entry, previous in zip(source["cards"], source["rows"], strict=True):
            if not previous["score"]["learner"]:
                continue
            pupils.clear()
            started = []
            observations.clear()
            with patch.object(OpticalPlanner, "command", command):
                row, _ = play_card(feeds, entry, retained, retained, Path(tmp) / "point.json.gz",
                                   practice=False, cap=4)
            row["pupil_ticks"] = started.copy()
            row["first_teacher_motion"] = observations[0] if observations else None
            result["cards"].append(entry)
            result["rows"].append(row)
            _write(output, result)
            print(entry["id"], entry["seed"], row["score"]["learner"],
                  row["pupil_ticks"], row["first_teacher_motion"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("feeds", "reference", "checkpoint", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--steps", type=int, default=5)
    a = p.parse_args()
    roll_in(a.feeds, a.reference, a.checkpoint, a.output, steps=a.steps)
