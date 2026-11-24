"""Confirm paired return-to-the-other-side transfer on unused measured opening releases."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

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
from .tt31_ready import centered


def select(feeds, folder):
    excluded = set()
    for name in ("course-plan", "full-plan", "transition-practice"):
        archive = _read(folder / (name + ".json.gz"))
        for group in ("cards", "held", "staged"):
            excluded.update(c["id"] for c in archive.get(group, []))
    shots = measured_launches(feeds)
    candidates = [s for s in opening_candidates(shots) if s["id"] not in excluded][:160]
    ready = _read(folder / "left-ready.json")["ready"]
    selected = []
    with centered(ready):
        arena = replay_arena([])
        try:
            for ordinal, i in enumerate(
                np.random.default_rng(319712).permutation(len(candidates))
            ):
                c = card(
                    candidates[i]["id"], "periodic", len(selected), 319712 + ordinal
                )
                if neutral_delivery(arena, shots, c) is not None:
                    selected.append(c)
                    if len(selected) == 4:
                        break
        finally:
            arena.close()
    if len(selected) != 4:
        raise ValueError("Insufficient physical offers")
    _write(
        folder / "confirm-plan.json.gz",
        {"cards": selected, "selection_seed": 319712, "feed_sha256": _hash(feeds)},
    )


def evaluate(feeds, folder, mode):
    plan = _read(folder / "confirm-plan.json.gz")
    if _hash(feeds) != plan["feed_sha256"]:
        raise ValueError("Launch data changed")
    ready = _read(folder / "left-ready.json")["ready"]
    checkpoint = folder / (
        "ready-staged-checkpoint.json.gz"
        if mode == "before"
        else "transition-checkpoint.json.gz"
    )
    optical = checkpoint
    if mode == "original":
        checkpoint, optical = MOVEMENTS, ERRORS
    result = {
        "phase": "periodic",
        "cards": plan["cards"],
        "checkpoint_sha256": _hash(checkpoint),
        "initial_optical_sha256": _hash(optical),
        "opening_side": 1,
        "ready": ready,
        "rows": [],
    }
    with centered(ready), TemporaryDirectory() as tmp:
        for c in plan["cards"]:
            row, _ = play_card(
                feeds,
                c,
                checkpoint,
                optical,
                Path(tmp) / "point.json.gz",
                practice=False,
                cap=24,
            )
            result["rows"].append(row)
            _write(folder / f"confirm-{mode}.json.gz", result)
            print(c["id"], row["score"]["learner"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("feeds", type=Path)
    p.add_argument("mode", choices=("plan", "before", "after", "original"))
    a = p.parse_args()
    if a.mode == "plan":
        select(a.feeds, a.folder)
    else:
        evaluate(a.feeds, a.folder, a.mode)
