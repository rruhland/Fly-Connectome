"""Change only the lateral request after the opening and re-evaluate feedback and blind replay."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt31_curriculum_opponent as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_ready import centered
from .tt33_joint import point, resources


def run(source, checkpoint, output):
    d = _read(source)
    if d["checkpoint_sha256"] != _hash(checkpoint):
        raise ValueError("Student differs")
    model = _read(checkpoint)
    original = course.placement
    result = {
        "ready": d["ready"],
        "cards": d["cards"],
        "opening_side": 1,
        "rows": [],
        "source_sha256": _hash(source),
        "checkpoint_sha256": _hash(checkpoint),
    }
    with centered(d["ready"]):
        arena = course.replay_arena([])
        stereo, rig, stride, lo, hi = resources(arena)
        try:
            for c, ref in zip(d["cards"], d["rows"], strict=True):

                def mirror(mode, index, rng, opening=c["index"]):
                    target = original(mode, index, rng)
                    if index > opening:
                        return (-target[0], *target[1:])
                    return target

                with patch.object(course, "placement", mirror):
                    arena.cards = iter([c])
                    live = point(
                        arena,
                        stereo,
                        rig,
                        model,
                        stride,
                        lo,
                        hi,
                        ref["shot"],
                        c,
                        cap=16,
                    )
                    arena.launch(ref["shot"], card=c)
                    arena.partner.reset(seed=c["seed"])
                    for i in range(3600):
                        action = (
                            ref["actions"][i]
                            if i < len(ref["actions"])
                            else np.zeros(6)
                        )
                        _, _, end, cap, _ = arena.exchange(action, np.zeros(5))
                        if (
                            end
                            or cap
                            or arena.legal_returns + arena.rival_returns - 1 >= 16
                        ):
                            break
                    blind = audit(arena.events, opening_side=1)
                    current = live["placements"]["delivered"]
                    for r in [current, arena.point_placements()["delivered"]]:
                        for k in (
                            "far_contact",
                            "near_bounce",
                            "near_contact",
                            "near_landed",
                        ):
                            if digest_json(r[0][k]) != digest_json(
                                ref["placements"]["delivered"][0][k]
                            ):
                                raise ValueError("Opening stroke changed")
                    if len(current) > 1:
                        old = ref["placements"]["requested"][1]["target"]
                        new = live["placements"]["requested"][1]["target"]
                        if new != [-old[0], *old[1:]]:
                            raise ValueError(
                                "Second request was not the intended lateral mirror"
                            )
                    live["original_arm"] = ref["score"]["learner"]
                    live["blind_arm"] = blind["learner"]
                    live["blind_total"] = blind["returns"]
                    result["rows"].append(live)
                    _write(output, result)
                    print(
                        c["id"],
                        live["original_arm"],
                        blind["learner"],
                        live["score"]["learner"],
                        flush=True,
                    )
        finally:
            stereo.close()
            arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "checkpoint", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    run(a.source, a.checkpoint, a.output)
