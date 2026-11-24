"""Preserve a verified impact and use the remainder of its motor tape to return to readiness."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory as motor
from .tt4_transfer_trial import _hash, _read, _write
from .tt31_curriculum import ERRORS, MOVEMENTS, play_card
from .tt31_ready import centered


class Recovery(motor.RallyObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tried = self.verified = 0

    def trajectories(self, goals, target, cutoff):
        rates, score = super().trajectories(goals, target, cutoff)
        valid = np.flatnonzero(np.isfinite(score[:, 0]))
        if not len(valid):
            return rates, score
        i = valid[np.argmin(score[valid, 0])]
        arena = self.solver.arena
        # Readiness is the same at every location. Start after the predicted
        # collision and retain the exact initial targets through separation.
        start = int(np.ceil((score[i, 1] + 0.065) / arena.control_dt))
        if start < 1 or start >= goals.shape[1] - 2:
            return rates, score
        pose = np.asarray(arena._arm_home(1)[0])
        new = goals[i : i + 1].copy()
        q = new[0, start - 1]
        n = len(new[0]) - start
        duration = max(0.26, float(np.max(abs(pose - q))) / 3.0)
        u = np.clip(np.arange(1, n + 1) * arena.control_dt / duration, 0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        new[0, start:] = q + smooth[:, None] * (pose - q)
        self.tried += 1
        changed, verified = super().trajectories(new, target, cutoff)
        if np.isfinite(verified[0, 0]) and np.allclose(
            score[i, 1:6], verified[0, 1:6], atol=0.002, rtol=0.0
        ):
            self.verified += 1
            rates[i] = changed[0]
            score[i] = verified[0]
        return rates, score


def run(folder, feeds, reference, output, *, unpracticed=False):
    matched = _read(reference)
    if len(matched["rows"]) != len(matched["cards"]):
        raise ValueError("Reference games incomplete")
    checkpoint = folder / "ready-staged-checkpoint.json.gz"
    source_hash = matched.get(
        "initial_movement_sha256", matched.get("checkpoint_sha256")
    )
    if source_hash == _hash(MOVEMENTS):
        checkpoint, optical = MOVEMENTS, ERRORS
    else:
        optical = checkpoint
    if _hash(checkpoint) != source_hash or _hash(optical) != matched.get(
        "initial_optical_sha256", source_hash
    ):
        raise ValueError("Unrecognized movement or optical checkpoint")
    if unpracticed:
        checkpoint, optical = MOVEMENTS, ERRORS
    result = {
        "ready": matched["ready"],
        "unpracticed": unpracticed,
        "opening_side": 1,
        "phase": matched["phase"],
        "cards": matched["cards"],
        "rows": [],
        "checkpoint_sha256": _hash(checkpoint),
        "optical_sha256": _hash(optical),
        "reference_sha256": _hash(reference),
    }
    for card in matched["cards"]:
        instances = []

        def wrap(*args, log=instances, **kwargs):
            objective = Recovery(*args, **kwargs)
            log.append(objective)
            return objective

        with (
            centered(matched["ready"]),
            patch.object(motor, "RallyObjective", wrap),
            TemporaryDirectory() as temp,
        ):
            row, _ = play_card(
                feeds,
                card,
                checkpoint,
                optical,
                Path(temp) / "point.json.gz",
                practice=False,
                cap=24 if matched["phase"] == "periodic" else 16,
            )
        row["recovery_candidates"] = sum(i.tried for i in instances)
        row["verified_recovery_candidates"] = sum(i.verified for i in instances)
        result["rows"].append(row)
        _write(output, result)
        print(
            card["key"],
            row["score"]["learner"],
            row["verified_recovery_candidates"],
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "feeds", "reference", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--unpracticed", action="store_true")
    a = p.parse_args()
    run(a.folder, a.feeds, a.reference, a.output, unpracticed=a.unpracticed)
