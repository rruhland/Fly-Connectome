"""Carry local memory through uninterrupted alternating-side SO-101 rallies."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from .tt4_arena import BALL_RADIUS
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import ERRORS, MOVEMENTS, OUTPUT, card, neutral_delivery, opening_candidates, play_card
from .tt31_curriculum_opponent import lane, replay_arena
from .tt31_curriculum_summary import compare_delivery, first_deliveries, held_difference
from .tt31_ready import centered


CAPS = {"practice": 12, "periodic": 24, "full": 16}
START = "ready-staged-checkpoint.json.gz"
RETAINED = "transition-checkpoint.json.gz"


def plan(feeds, output):
    held = {phase: _read(output / file)["cards"] for phase, file in (
        ("periodic", "ready-periodic-original.json.gz"), ("full", "full-plan.json.gz"),
    )}
    previous = _read(output / "course-plan.json.gz")
    exclusion = {c["id"] for packet in held.values() for c in packet}
    exclusion.update(c["id"] for c in previous["staged"] + previous["held"])
    shots = measured_launches(feeds)
    arena = replay_arena([])
    selected = []
    try:
        for i, shot in enumerate(opening_candidates(shots)):
            if shot["id"] in exclusion:
                continue
            entry = card(shot["id"], "periodic", len(selected), 315893 + 17 * i)
            reply = neutral_delivery(arena, shots, entry)
            if reply is not None:
                selected.append(dict(card=entry, first_reply=reply))
                print("neutral", entry["id"], reply["observed_lane"], flush=True)
            if len(selected) == 4:
                break
    finally:
        arena.close()
    if len(selected) != 4:
        raise ValueError("Four new alternating-side releases did not clear the neutral screen")
    deck = dict(feed_sha256=_hash(feeds), ready_sha256=_hash(output / "left-ready.json"),
                initial_checkpoint_sha256=_hash(output / START), caps=CAPS,
                cards={"practice": [item["card"] for item in selected], **held},
                neutral_practice=selected)
    _write(output / "transition-plan.json.gz", deck)
    return deck


def run(feeds, output, phase):
    ready_file = output / "left-ready.json"
    ready = _read(ready_file)["ready"]
    training = phase == "practice"
    deck_path = output / "transition-plan.json.gz"
    initial = output / START
    with centered(ready), TemporaryDirectory(prefix="tt31-transition-") as work:
        if training:
            parent = _read(output / "ready-practice-staged.json.gz")
            if len(parent["rows"]) != 16 or parent["ready"] != ready or parent["checkpoint_sha256"] != _hash(initial):
                raise ValueError("The starting checkpoint does not match the 16 drills and calibrated posture")
            deck = plan(feeds, output)
        else:
            deck = _read(deck_path)
            initial = output / RETAINED
            practice = _read(output / "transition-practice.json.gz")
            if len(practice["rows"]) != 4 or practice["checkpoint_sha256"] != _hash(initial):
                raise ValueError("The four-rally checkpoint is incomplete")
        if _hash(feeds) != deck["feed_sha256"] or _hash(ready_file) != deck["ready_sha256"]:
            raise ValueError("Measured feed or calibrated posture changed")
        current = initial
        initial_hash = _hash(initial)
        state = _read(initial)
        result = dict(phase=phase, ready=ready, ready_sha256=_hash(ready_file),
                      plan_sha256=_hash(deck_path), opening_side=1, cap=deck["caps"][phase],
                      initial_movement_sha256=initial_hash, initial_optical_sha256=initial_hash,
                      initial_examples=len(state["checkpoint"]), rows=[], cards=deck["cards"][phase])
        temporary = Path(work) / "point.json.gz"
        for entry in result["cards"]:
            row, retained = play_card(feeds, entry, current, current, temporary,
                                      practice=training, cap=result["cap"])
            if training:
                current = output / RETAINED
                _write(current, retained)
                row["final_checkpoint_sha256"] = _hash(current)
                result["checkpoint_sha256"] = row["final_checkpoint_sha256"]
            elif retained != state:
                raise ValueError("Frozen held game changed a motor or optical example")
            row["final_optical_samples"] = len(retained["optical_outcome_checkpoint"]["samples"])
            row["final_censored_samples"] = len(retained["optical_outcome_checkpoint"].get("censored_samples", []))
            result["rows"].append(row)
            _write(output / f"transition-{phase}.json.gz", result)
            arrivals = [r["observed_lane"] for r in row["placements"]["delivered"]]
            print(phase, entry["id"], arrivals, row["score"]["learner"], row["score"]["returns"], flush=True)


def reference(feeds, output):
    """Reproduce the first successful switch card from the original checkpoint."""
    practice_path = output / "transition-practice.json.gz"
    practice = _read(practice_path)
    entry = next(card for card in practice["cards"] if card["id"] == 6810)
    with centered(practice["ready"]), TemporaryDirectory(prefix="tt31-transition-control-") as work:
        row, retained = play_card(feeds, entry, MOVEMENTS, ERRORS, Path(work) / "point.json.gz",
                                  practice=True, cap=practice["cap"])
    result = dict(phase="starting-reference", ready=practice["ready"], cap=practice["cap"],
                  cards=[entry], rows=[row], opening_side=1, learn_within_point=True,
                  source_sha256=_hash(practice_path), initial_movement_sha256=_hash(MOVEMENTS),
                  initial_optical_sha256=_hash(ERRORS),
                  final_optical_samples=len(retained["optical_outcome_checkpoint"]["samples"]))
    _write(output / "transition-6810-reference.json.gz", result)
    print("starting-reference", row["score"]["learner"], row["score"]["returns"], flush=True)


def outcome(reply):
    position = (reply["receive_line"] or {}).get("xyz")
    if position is None and reply["near_contact"] is not None:
        position = reply["near_contact"]["ball"]
    return dict(lane=reply["observed_lane"], intended=lane(reply["target"][0]), contact=reply["near_contact"] is not None,
                legal=reply["near_landed"], x_m=position[0] if position is not None else None)


def transitions(rows):
    counts = dict(first=Counter(), later=Counter(), switches=Counter(), second_switch=Counter())
    points = []
    for row in rows:
        score = audit(row["events"], opening_side=1)
        if any(score[field] != row["score"][field] for field in ("returns", "learner", "opponent")):
            raise ValueError("Referee counts changed")
        replies = [outcome(r) for r in row["placements"]["delivered"]]
        if len(replies) != score["opponent"] or sum(r["legal"] for r in replies) != score["learner"]:
            raise ValueError("Incoming audit omitted a legal shot")
        point_switches = []
        for i, current in enumerate(replies):
            side = current["lane"] or "unknown"
            bins = [counts["first" if i == 0 else "later"]]
            previous = replies[i - 1] if i else None
            if previous and previous["legal"] and {previous["lane"], current["lane"]} == {"left", "right"}:
                bins.append(counts["switches"])
                if i == 1:
                    bins.append(counts["second_switch"])
                point_switches.append(dict(incoming=i + 1, prior_lane=previous["lane"], **current))
            for group in bins:
                group[(side, "offered")] += 1
                group[(side, "contacts")] += current["contact"]
                group[(side, "legal")] += current["legal"]
        terminal = row["events"][-1]
        points.append(dict(id=row["shot"]["id"], seed=row["opponent_seed"], arm=score["learner"],
                           both=score["returns"], offered=replies, opposite_side_replies=point_switches,
                           ending=terminal.get("reason", terminal["kind"]), winner=terminal.get("winner")))
    totals = {name: {lane: {key: counter[(lane, key)] for key in ("offered", "contacts", "legal")}
                    for lane in ("left", "middle", "right", "unknown")}
              for name, counter in counts.items()}
    return dict(arm=sum(p["arm"] for p in points), both=sum(p["both"] for p in points),
                cpu_s=sum(r["process_cpu_s"] for r in rows), counts=totals, points=points)


def second_miss(row):
    replies = row["placements"]["delivered"]
    if len(replies) > 1 and replies[0]["near_landed"] and not replies[1]["near_landed"]:
        if {r["observed_lane"] for r in replies[:2]} == {"left", "right"}:
            return replies[1]
    return None


def replay_point(row, miss=None):
    arena = replay_arena([row["placements"]["card"]])
    try:
        arena.launch(row["shot"])
        arena.partner.reset(seed=row["opponent_seed"])
        times = []
        nearest = None
        original = arena._contacts
        blade = arena.model.geom("learner_racket").id
        ball = arena.model.geom("ball").id
        radius, thickness = arena.model.geom_size[blade, :2]

        def observe(incoming):
            nonlocal nearest
            if miss is not None and arena.data.time >= miss["near_bounce"]["time"]:
                center = arena.data.geom_xpos[blade]
                axes = arena.data.geom_xmat[blade].reshape(3, 3)
                position = arena.data.geom_xpos[ball]
                local = axes.T @ (position - center)
                outside = [np.linalg.norm(local[:2]) - radius, abs(local[2]) - thickness]
                gap = float(np.linalg.norm(np.maximum(0., outside)) - BALL_RADIUS)
                if nearest is None or gap < nearest["sphere_to_blade_gap_m"]:
                    nearest = dict(sphere_to_blade_gap_m=gap, time_s=float(arena.data.time),
                                   ball_m=position.tolist(), blade_m=center.tolist())
            original(incoming)

        with patch.object(arena, "_contacts", observe):
            for action in row["actions"]:
                arena.exchange(action, np.zeros(5))
                times.append(float(arena.data.time))
        if not np.all(np.diff([0., *times]) > 0):
            raise ValueError("The live clock reset inside the point")
        events = digest_json(arena.events)
        placements = digest_json(arena.point_placements())
        if events != digest_json(row["events"]) or placements != digest_json(row["placements"]):
            raise ValueError("Action-only replay changed a referee event or placement")
        result = dict(id=row["shot"]["id"], seed=row["opponent_seed"], events_sha256=events,
                      placement_sha256=placements, control_ticks=len(times), final_time_s=times[-1],
                      arena_launches=1, physics_states=round(times[-1] / arena.model.opt.timestep))
        if miss is not None:
            commands = np.asarray(row["actions"])[miss["near_bounce"]["step"]:]
            result["second_miss"] = dict(lane=miss["observed_lane"], contact=miss["near_contact"] is not None,
                                         terminal=row["events"][-1].get("reason"), closest=nearest,
                                         active_command_ticks_after_bounce=int(np.count_nonzero(np.any(commands != 0, axis=1))))
        return result
    finally:
        arena.close()


def summarize(output):
    plan_path = output / "transition-plan.json.gz"
    deck = _read(plan_path)
    trial = {}
    for phase in CAPS:
        path = output / f"transition-{phase}.json.gz"
        document = _read(path)
        if (document["cards"] != deck["cards"][phase] or len(document["rows"]) != len(document["cards"])
                or document["cap"] != deck["caps"][phase]):
            raise ValueError(f"Incomplete or changed {phase} schedule")
        trial[phase] = dict(totals=transitions(document["rows"]), sha256=_hash(path))
        if phase == "practice":
            with centered(document["ready"]):
                replay = [replay_point(row, second_miss(row)) for row in document["rows"]]
            _write(output / "transition-practice-replay.json", dict(source_sha256=_hash(path), points=replay))
            continue
        base_path = output / f"ready-{phase}-staged.json.gz"
        base = _read(base_path)
        if (base["cards"] != document["cards"] or len(base["rows"]) != len(document["rows"])
                or base["ready"] != document["ready"] or base.get("recall_only", False)
                or base["initial_movement_sha256"] != deck["initial_checkpoint_sha256"]
                or base["initial_optical_sha256"] != deck["initial_checkpoint_sha256"]):
            raise ValueError(f"Baseline for {phase} differs in schedule, posture or initial memory")
        rows = document["rows"]
        trial[phase].update(baseline_sha256=_hash(base_path), baseline=transitions(base["rows"]),
                            paired=held_difference(rows, base["rows"]), first_delivery=compare_delivery(
                                first_deliveries(rows), first_deliveries(base["rows"])))
        if phase == "periodic":
            with centered(document["ready"]):
                trial[phase]["miss_replays"] = {
                    checkpoint: [replay_point(row, miss) for row in cases if (miss := second_miss(row)) is not None]
                    for checkpoint, cases in (("baseline", base["rows"]), ("carried", rows))
                }
    practice_path = output / "transition-practice.json.gz"
    practice = _read(practice_path)
    ref_path = output / "transition-6810-reference.json.gz"
    ref = _read(ref_path)
    previous = next(r for r in practice["rows"] if r["shot"]["id"] == 6810)
    if (ref["source_sha256"] != _hash(practice_path) or ref["ready"] != practice["ready"]
            or ref["cap"] != practice["cap"] or ref["cards"] != [previous["placements"]["card"]]):
        raise ValueError("The original 6810 control used a different game")
    _write(output / "transition-summary.json", dict(
        plan_sha256=_hash(plan_path), phases=trial, checkpoint_sha256=_hash(output / RETAINED),
        reference_6810=dict(sha256=_hash(ref_path), original=transitions(ref["rows"]),
                            after_drills=transitions([previous]),
                            checkpoint_changed_before=previous["initial_checkpoint_sha256"] != practice["initial_movement_sha256"],
                            first_delivery=compare_delivery(first_deliveries(ref["rows"]), first_deliveries([previous]))),
    ))
    for phase, info in trial.items():
        print(phase, info["totals"]["arm"], info["totals"]["both"], info.get("paired"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("phase", choices=(*CAPS, "summary", "starting-reference"))
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    if args.phase == "summary":
        summarize(args.output)
    elif args.phase == "starting-reference":
        reference(args.feeds, args.output)
    else:
        run(args.feeds, args.output, args.phase)
