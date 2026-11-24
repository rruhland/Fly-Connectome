"""Create a release-disjoint placement course and evaluate the fixed recovery reference."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory as motor
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import card, neutral_delivery, opening_candidates, play_card
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt31_recovery import Recovery


def generate(feeds, folder, output):
    omitted = {12330}
    for p in ("course-plan", "full-plan", "confirm-plan", "transition-practice"):
        dataset = _read(folder / (p + ".json.gz"))
        for label in ("cards", "staged", "held"):
            omitted.update(k["id"] for k in dataset.get(label, []))
    shots = measured_launches(feeds)
    candidates = [s for s in opening_candidates(shots) if s["id"] not in omitted][:300]
    counts = Counter()
    cards = []
    target = {"left": 3, "middle": 2, "right": 3}
    seed = 324129
    with centered(_read(folder / "left-ready.json")["ready"]):
        arena = replay_arena([])
        try:
            for i, j in enumerate(
                np.random.default_rng(seed).permutation(len(candidates))
            ):
                trial = card(candidates[j]["id"], "full", i, seed + i)
                delivery = neutral_delivery(arena, shots, trial)
                if (
                    delivery
                    and counts[delivery["observed_lane"]]
                    < target[delivery["observed_lane"]]
                ):
                    counts[delivery["observed_lane"]] += 1
                    cards.append(trial)
                    if counts == target:
                        break
        finally:
            arena.close()
    if counts != target:
        raise ValueError("Unable to balance the physically delivered openings")
    _write(
        output, {"cards": cards, "feed_sha256": _hash(feeds), "selection_seed": seed}
    )


def fixed(feeds, folder, plan, output):
    deck = _read(plan)
    if _hash(feeds) != deck["feed_sha256"]:
        raise ValueError("Feeds changed")
    checkpoint = folder / "ready-staged-checkpoint.json.gz"
    ready = _read(folder / "left-ready.json")["ready"]
    result = {
        "cards": deck["cards"],
        "rows": [],
        "ready": ready,
        "phase": "full",
        "opening_side": 1,
        "plan_sha256": _hash(plan),
        "checkpoint_sha256": _hash(checkpoint),
    }
    with (
        centered(ready),
        patch.object(motor, "RallyObjective", Recovery),
        TemporaryDirectory() as tmp,
    ):
        for c in deck["cards"]:
            row, _ = play_card(
                feeds,
                c,
                checkpoint,
                checkpoint,
                Path(tmp) / "point.json.gz",
                practice=False,
                cap=16,
            )
            result["rows"].append(row)
            _write(output, result)
            print(c["id"], row["score"]["learner"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "fixed"))
    for name in ("feeds", "folder", "plan", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    if a.mode == "plan":
        generate(a.feeds, a.folder, a.plan)
    else:
        fixed(a.feeds, a.folder, a.plan, a.output)
