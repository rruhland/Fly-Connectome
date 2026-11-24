"""Evaluate retained and continuing learners against placements spanning the crossover."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import (
    ERRORS,
    MOVEMENTS,
    card,
    neutral_delivery,
    opening_candidates,
    play_card,
)
from .tt31_curriculum_opponent import replay_arena


def plan(feeds, folder):
    old = _read(folder / "course-plan.json.gz")
    exclude = {c["id"] for c in old["staged"] + old["held"]}
    shots = measured_launches(feeds)
    arena = replay_arena([])
    quota = {"left": 3, "middle": 2, "right": 3}
    counts = Counter()
    cards, calibration = [], []
    try:
        for i, shot in enumerate(opening_candidates(shots)):
            if shot["id"] in exclude:
                continue
            entry = card(shot["id"], "full", i, 314271 + 7 * i)
            observed = neutral_delivery(arena, shots, entry)
            if observed is None:
                continue
            side = observed["observed_lane"]
            if counts[side] == quota[side]:
                continue
            counts[side] += 1
            cards.append(entry)
            calibration.append(observed)
            if counts == quota:
                break
    finally:
        arena.close()
    if counts != quota:
        raise ValueError("Could not build a balanced unseen cohort")
    _write(
        folder / "full-plan.json.gz",
        {"feed_sha256": _hash(feeds), "cards": cards, "delivery": calibration},
    )


def run(feeds, folder, order, continuing):
    deck = _read(folder / "full-plan.json.gz")
    if _hash(feeds) != deck["feed_sha256"]:
        raise ValueError("Measured release source changed")
    movement = (
        MOVEMENTS
        if order == "original"
        else folder / f"course-{order}-checkpoint.json.gz"
    )
    optical = ERRORS if order == "original" else movement
    if order != "original":
        training = _read(folder / f"course-practice-{order}.json.gz")
        if len(training["rows"]) != len(training["cards"]) or training[
            "checkpoint_sha256"
        ] != _hash(movement):
            raise ValueError("Practice checkpoint incomplete or mismatched")
    prefix = f"full-{order}" + ("-online" if continuing else "")
    result = {
        "order": order,
        "continuing": continuing,
        "initial_checkpoint_sha256": _hash(movement),
        "initial_optical_sha256": _hash(optical),
        "cards": deck["cards"],
        "opening_side": 1,
        "cap": 16,
        "rows": [],
    }
    with TemporaryDirectory() as d:
        for entry in deck["cards"]:
            row, learned = play_card(
                feeds,
                entry,
                movement,
                optical,
                Path(d) / "point.json.gz",
                practice=continuing,
                cap=16,
            )
            if continuing:
                movement = optical = folder / (prefix + "-checkpoint.json.gz")
                _write(movement, learned)
                result["checkpoint_sha256"] = _hash(movement)
            result["rows"].append(row)
            _write(folder / (prefix + ".json.gz"), result)
            print(
                entry["id"],
                row["score"]["learner"],
                row["score"]["returns"],
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("mode", choices=("plan", "original", "staged", "mixed"))
    p.add_argument("--online", action="store_true")
    args = p.parse_args()
    if args.mode == "plan":
        plan(args.feeds, args.folder)
    else:
        run(args.feeds, args.folder, args.mode, args.online)
