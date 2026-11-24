"""Collect a wider framewise optical curriculum from verified physical returns."""

import argparse
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt35_replies import varied_opponent


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
OUTPUT = ARTIFACTS / "tt37"
PUPIL = ARTIFACTS / "tt33" / "joint-raw.json.gz"
SOURCES = (
    "tt36/reach-new-aligned", "tt36/reach-confirm-aligned",
    "tt35/search-held-confirmation", "tt33/joint-fresh-corrected",
    "tt36/paired-new-hit", "tt36/paired-confirm-hit",
    "tt35/search-held-novel", "tt36/online-replies-carry",
    "tt35/replies-nominal", "tt31/confirm-recovery", "tt31/recovery-periodic",
    "tt31/transition-6810-reference",
)


def curriculum(ready):
    """Keep varied successful recordings and limit the dose of any one release."""
    choices = {}
    for source in SOURCES:
        path = ARTIFACTS / f"{source}.json.gz"
        doc = _read(path)
        if doc["ready"] != ready or doc["opening_side"] != 1:
            raise ValueError(f"Incompatible motor replay source {path.name}")
        varied = doc.get("reply_seeds", [None] * len(doc["cards"]))
        for card, row, reply in zip(doc["cards"], doc["rows"], varied, strict=True):
            if not row["score"]["learner"]:
                continue
            if row["opponent_seed"] != card["seed"]:
                raise ValueError("Recorded opponent seed differs from course card")
            key = card["id"], card["seed"], reply
            previous = choices.get(key)
            if previous is None or row["score"]["learner"] > previous[2]["score"]["learner"]:
                choices[key] = (path, card, row, reply)
    by_release = {}
    for entry in choices.values():
        by_release.setdefault(entry[1]["id"], []).append(entry)
    roster = []
    for group in by_release.values():
        group.sort(key=lambda r: -r[2]["score"]["learner"])
        roster.extend(group[:2])
    # Capture ordinary single-return cases early and long repeated rallies after.
    return sorted(roster, key=lambda r: (r[2]["score"]["learner"], r[1]["id"]))


def observed(vision, tick, frames, q):
    vision.update(tick, frames, q)
    estimate = vision.eye.latest
    time = tick * vision.dt
    if (estimate is None or time - estimate.time >= .105 or estimate.rms >= 2.4
            or estimate.depth_sigma >= .16 or abs(estimate.velocity[1]) < .90):
        return None
    return np.r_[*estimate.advance(time)]


def samples(arena, renderer, rig, entry, row, reply_seed):
    arena.launch(row["shot"], card=entry)
    arena.partner.reset(seed=entry["seed"])
    contacts = [s["hit"] for s in audit(row["events"], opening_side=1)["shots"] if s["side"] == 1]
    eye = Vision(rig, arena.control_dt, raw=True)
    axes = arena._arm_ids[1][2]
    states = []
    actual_labels = []
    marks = []
    previous = arena.joints
    with varied_opponent(entry["index"], reply_seed) if reply_seed is not None else nullcontext():
        for tick, action in enumerate(row["actions"]):
            time = tick * arena.control_dt
            q = arena.joints
            # These windows select training samples only. The deployed field sees every exposure.
            optical = None
            if any(-.84 < time - t < .49 for t in contacts):
                optical = observed(eye, tick, renderer.capture() if tick % 2 == 0 else None, q)
            arena.exchange(action, FAR_COMMAND)
            actual_labels.append(arena.data.ctrl[axes].copy())
            if optical is not None:
                delta = min((time - t for t in contacts), key=abs)
                if (-.58 < delta < 0. and optical[4] < -.9
                        or .055 < delta < .46 and optical[4] > .9 and optical[1] < .65):
                    states.append(np.r_[optical, q[:5], ((q - previous) / arena.control_dt)[:5]])
                    marks.append((tick, delta))
            previous = q
    if digest_json(arena.events) != digest_json(row["events"]):
        raise ValueError(f"Training action tape diverged on {entry['id']} ({reply_seed=})")
    if not marks:
        return np.empty((0, 16)), np.empty((0, 6)), np.empty((0, 6)), np.empty((0, 2))
    near = [actual_labels[i] for i, _ in marks]
    lead = [actual_labels[min(i + 2, len(actual_labels) - 1)] for i, _ in marks]
    return np.asarray(states), np.asarray(near), np.asarray(lead), np.asarray(marks)


def capture(path):
    pupil = _read(PUPIL)
    roster = curriculum(pupil["ready"])
    receipts = []
    rows = []
    with centered(pupil["ready"]):
        arena = replay_arena([])
        renderer, rig, *_ = resources(arena)
        try:
            for i, (source, entry, row, reply) in enumerate(roster, 1):
                packet = samples(arena, renderer, rig, entry, row, reply)
                rows.append(packet)
                receipts.append(dict(source=str(source.relative_to(ARTIFACTS)), id=entry["id"], card=entry,
                                     reply=reply, own_legal=row["score"]["learner"], samples=len(packet[0])))
                print(i, len(roster), entry["id"], row["score"]["learner"], len(packet[0]), flush=True)
        finally:
            renderer.close()
            arena.close()
    output = dict(zip(("states", "targets_now", "targets_lead", "marks"),
                      (np.concatenate(chunk) for chunk in zip(*rows, strict=True)), strict=True))
    output["releases"] = np.repeat([row["id"] for row in receipts], [row["samples"] for row in receipts])
    np.savez_compressed(path, **output)
    _write(path.with_suffix(".json"), dict(
        sources={str(path.relative_to(ARTIFACTS)): _hash(path) for path, _, _, _ in roster},
        pupil_sha256=_hash(PUPIL), records=receipts, releases=sorted({r["id"] for r in receipts}),
        legal_returns=sum(r["own_legal"] for r in receipts), samples=len(output["states"]),
        sha256=_hash(path),
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT / "joint-observations.npz")
    args = p.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    capture(args.output)
