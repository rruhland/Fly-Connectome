"""Verify the entire random target stream, native ball/paddle exchanges and CAD clearance."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_game_replay import film
from .tt37_joint_report import replay_case
from .tt37_opponent import draw


def witness(source, ident, output, movie):
    d = _read(source)
    ix = next(i for i, c in enumerate(d["cards"]) if c["id"] == ident)
    c, r = d["cards"][ix], d["rows"][ix]
    if c.get("opening", "random") != "random" or c.get("profile", "reach") != "reach":
        raise ValueError(
            "Requires the broad, continuously random first and later goals"
        )
    requests = r["placements"]["requested"]
    for j, request in enumerate(requests):
        expected, kind = draw(c, j, "random", None)
        if (
            request["leg"] != j
            or request["draw"] != kind
            or list(expected) != request["target"]
        ):
            raise ValueError("Random target stream or ordinal mismatch")
    targets = np.array([p["target"] for p in requests])
    if len(np.unique(targets, axis=0)) != len(requests):
        raise ValueError("Duplicate random placement")
    sc = audit(r["events"], opening_side=1)
    if sc["learner"] != r["score"]["learner"] or sc["returns"] != r["score"]["returns"]:
        raise ValueError("Game score and tape disagree")
    doc = {"ready": d["ready"], "cards": [c], "rows": [r], "opening_side": 1}
    tape = output.with_name(output.stem + "-tape.json.gz")
    _write(tape, doc)
    if movie:
        video = output.with_suffix(".mp4")
        film(tape, video, f"SO-101: {sc['learner']} legal arm returns")
        receipt = _read(video.with_suffix(".json"))
    else:
        receipt = replay_case(d["ready"], "random", c, r)
    receipt.update(
        course=source.name,
        card=c,
        source_index=ix,
        tape=tape.name,
        tape_sha256=_hash(tape),
        event_sha256=digest_json(r["events"]),
        own=sc["learner"],
        opponent=sc["opponent"],
        alternating=sc["returns"],
        exact_random_draws=len(requests),
        requested_extent=[targets.min(axis=0).tolist(), targets.max(axis=0).tolist()],
        physical_speed=sc["outgoing_mps"],
        interval=sc["interval_s"],
        issued_controls=len(r["actions"]),
        terminal_reason=r["events"][-1].get("reason"),
    )
    _write(output, receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("id", type=int)
    p.add_argument("output", type=Path)
    p.add_argument("--movie", action="store_true")
    a = p.parse_args()
    witness(a.source, a.id, a.output, a.movie)
