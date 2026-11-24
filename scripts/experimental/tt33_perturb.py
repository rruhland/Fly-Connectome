"""Change only the opponent seed and compare fresh joint feedback with reissuing the original commands."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered


def make(source, output):
    d = _read(source)
    cards = []
    for i, c in enumerate(d["cards"]):
        cards.append(c | {"seed": 335071 + 13 * i, "key": f"feedback-change-{i}"})
    _write(output, {"cards": cards, "reference_sha256": _hash(source)})


def blind(source, plan, output):
    d = _read(source)
    pairs = _read(plan)
    if pairs["reference_sha256"] != _hash(source):
        raise ValueError("Replay source changed")
    rows = []
    with centered(d["ready"]):
        arena = replay_arena([])
        try:
            for prior, old, c in zip(
                d["rows"], d["cards"], pairs["cards"], strict=True
            ):
                for card, check in [(old, True), (c, False)]:
                    arena.launch(prior["shot"], card=card)
                    arena.partner.reset(seed=card["seed"])
                    actions = prior["actions"]
                    for i in range(3600):
                        action = actions[i] if i < len(actions) else np.zeros(6)
                        _, _, end, cap, _ = arena.exchange(action, np.zeros(5))
                        if (
                            end
                            or cap
                            or arena.legal_returns + arena.rival_returns - 1 >= 16
                        ):
                            break
                    if check:
                        if digest_json(arena.events) != digest_json(prior["events"]):
                            raise ValueError("Open-loop positive control diverged")
                    else:
                        score = audit(arena.events, opening_side=1)
                        rows.append(
                            {
                                "id": c["id"],
                                "seed": c["seed"],
                                "legal": score["learner"],
                                "combined": score["returns"],
                                "placements": arena.point_placements(),
                                "events_sha256": digest_json(arena.events),
                                "own_original": prior["score"]["learner"],
                            }
                        )
        finally:
            arena.close()
    _write(output, {"plan_sha256": _hash(plan), "rows": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "blind"))
    p.add_argument("source", type=Path)
    p.add_argument("plan", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.mode == "plan":
        make(a.source, a.plan)
    else:
        blind(a.source, a.plan, a.output)
