"""Reuse one ready posture across all points and transfer the resulting practice."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt31_curriculum import ERRORS, MOVEMENTS, PRACTICE_OPENING, card, play_card
from .tt31_ready import centered


def practice_cards(deck):
    by_mode = {
        m: [c for c in deck["staged"] if c["mode"] == m] for m in ("fixed", "periodic")
    }
    counts = Counter()
    varied = []
    for d in deck["neutral_practice"]:
        c = d["card"]
        side = d["first_reply"]["observed_lane"]
        if c["mode"] == "continuous" and counts[side] < 3:
            varied.append(c)
            counts[side] += 1
    return by_mode["fixed"][:4] + by_mode["periodic"][:6] + varied


def run(feeds, folder, phase, order, recall_only=False):
    deck = _read(folder / "course-plan.json.gz")
    ready_info = folder / "left-ready.json"
    ready = _read(ready_info)["ready"]
    training = phase == "practice"
    if recall_only and training:
        raise ValueError("Recall-only is a held assessment")
    suffix = "-recall" if recall_only else ""
    if training:
        cards = practice_cards(deck)
        if order == "mixed":
            order_idx = np.random.default_rng(317045).permutation(len(cards))
            cards = [cards[i] for i in order_idx]
    elif phase == "periodic":
        releases = [
            PRACTICE_OPENING,
            PRACTICE_OPENING,
            deck["held"][0]["id"],
            deck["held"][1]["id"],
        ]
        cards = [
            card(shot, "periodic", i, 315509 + i) for i, shot in enumerate(releases)
        ]
    else:
        cards = _read(folder / "full-plan.json.gz")["cards"]
    movement, optical = MOVEMENTS, ERRORS
    checkpoint = folder / f"ready-{order}-checkpoint.json.gz"
    if not training and order != "original":
        history = _read(folder / f"ready-practice-{order}.json.gz")
        if (
            len(history["rows"]) != len(history["cards"])
            or _hash(checkpoint) != history["checkpoint_sha256"]
        ):
            raise ValueError("Incomplete training")
        movement = optical = checkpoint
    result = {
        "ready": ready,
        "ready_sha256": _hash(ready_info),
        "recall_only": recall_only,
        "phase": phase,
        "order": order,
        "cards": cards,
        "opening_side": 1,
        "initial_movement_sha256": _hash(movement),
        "initial_optical_sha256": _hash(optical),
        "rows": [],
    }
    with centered(ready), TemporaryDirectory() as tmp:
        for c in cards:
            row, learned = play_card(
                feeds,
                c,
                movement,
                optical,
                Path(tmp) / "point.json.gz",
                practice=training,
                cap=2 if training else 24 if phase == "periodic" else 16,
                recall_only=recall_only,
            )
            result["rows"].append(row)
            if training:
                _write(checkpoint, learned)
                movement = optical = checkpoint
                result["checkpoint_sha256"] = _hash(checkpoint)
            _write(folder / f"ready-{phase}-{order}{suffix}.json.gz", result)
            print(c["key"], row["score"]["learner"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("phase", choices=("practice", "full", "periodic"))
    p.add_argument("order", choices=("staged", "mixed", "original"))
    p.add_argument("--recall-only", action="store_true")
    args = p.parse_args()
    if args.phase == "practice" and args.order == "original":
        p.error("The original checkpoint has no additional practice")
    run(args.feeds, args.folder, args.phase, args.order, args.recall_only)
