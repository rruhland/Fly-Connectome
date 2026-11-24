"""Change incoming shots continuously around six practiced openings before a frozen test."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt31_curriculum_opponent as opponent
from . import tt36_local as local
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import neutral_delivery
from .tt31_ready import centered
from .tt33_joint import resources
from .tt36_course import OUT
from .tt36_staged import body_model, choose_from, movement


def displaced(card, delta):
    original = opponent.placement

    def request(mode, index, rng):
        normal = original(mode, index, rng)
        return tuple(np.array(normal) + delta) if index == card["index"] else normal

    return patch.object(opponent, "placement", request)


def plan(feeds):
    shots = measured_launches(feeds)
    orig = _read(OUT / "local-plan.json.gz")
    rows = []
    with centered(orig["ready"]):
        arena = opponent.replay_arena([])
        try:
            for i, case in enumerate(orig["cases"]):
                card = case["original"]
                # An independent neutral delivery establishes the physical comparison.
                base = neutral_delivery(arena, shots, card)
                for n, delta in enumerate(
                    (
                        (0.025, 0.0, 0.0, 0.04),
                        (-0.025, 0.0, 0.0, -0.04),
                        (0.05, 0.0, 0.01, -0.10),
                        (-0.05, 0.0, -0.01, 0.10),
                    )
                ):
                    with displaced(card, delta):
                        offer = neutral_delivery(arena, shots, card)
                    if offer is not None and base is not None:
                        rows.append(
                            {
                                "card": card,
                                "delta": delta,
                                "reply_seed": 367030 + i * 4 + n,
                                "original": base["near_bounce"],
                                "offered": offer["near_bounce"],
                                "bounce_shift_m": np.subtract(
                                    offer["near_bounce"]["ball"],
                                    base["near_bounce"]["ball"],
                                ).tolist(),
                            }
                        )
        finally:
            arena.close()
    _write(
        OUT / "nearby-plan.json",
        {"points": rows, "neutral_attempts": 24, "feed_sha256": _hash(feeds)},
    )
    print(len(rows))


def play(feeds, mode):
    plan = _read(OUT / "nearby-plan.json")
    shots = measured_launches(feeds)
    model = _read(local.BASE)
    ck = _read(OUT / "staged-checkpoint.json")

    def zero(_):
        return np.zeros(5)

    early = (
        zero
        if mode == "fixed"
        else local.choose_from(_read(OUT / "local-checkpoint.json"), "dense")
    )
    late = choose_from(ck["states"], "dense")
    context = (
        movement(body_model(), late, include_reach=mode != "late-only")
        if mode in ("both", "late-only")
        else nullcontext()
    )
    result = {
        "ready": model["ready"],
        "opening_side": 1,
        "cards": [r["card"] for r in plan["points"]],
        "mode": mode,
        "course_sha256": _hash(OUT / "nearby-plan.json"),
        "rows": [],
    }
    with centered(model["ready"]), context:
        arena = opponent.replay_arena([])
        stereo, rig, stride, lo, hi = resources(arena)
        try:
            for sample in plan["points"]:
                c = sample["card"]
                with displaced(c, sample["delta"]):
                    row = local.trial(
                        arena,
                        stereo,
                        rig,
                        stride,
                        lo,
                        hi,
                        model,
                        shots,
                        c,
                        early,
                        sample["reply_seed"],
                        progress=False,
                    )
                result["rows"].append(row)
                _write(OUT / f"nearby-{mode}.json.gz", result)
        finally:
            stereo.close()
            arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "fixed", "early", "both", "late-only"))
    p.add_argument("feeds", type=Path)
    args = p.parse_args()
    if args.mode == "plan":
        plan(args.feeds)
    else:
        play(args.feeds, args.mode)
