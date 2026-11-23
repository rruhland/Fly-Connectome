"""Expand the static receiving bank with disjoint recorded training launches."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt25_nearby_lessons import features
from .tt25_pose_demonstrations import episode, poses, suggest
from .tt25_recorded_exchange import table_model
from .tt26_grip_course import Grip, OUT, PRIOR, ReferenceReceiver, load_match
from .tt26_match_rebound import incoming
from .tt26_optical_transfer import MOUNTS, decoder, transport


FRESH = OUT / "fresh-recordings.json.gz"
OPTICS = OUT / "optical-face.json.gz"
RECEIPT = OUT / "match-broader.json.gz"
CHECKPOINT = OUT / "match-broader.npz"
MOUNT = "yaw_plus"
SEED = 1900121
FEEDS = 64


def ids_in(report):
    """Read shot rosters, stored rows and synthetic feeds named by their donor."""
    ids = set()
    if isinstance(report, dict):
        for key, value in report.items():
            if key in ("id", "donor", "source_id") and type(value) is int:
                ids.add(value)
            if key in ("train", "held", "conditional_roster", "held_roster", "donors", "ids") and isinstance(value, list):
                ids.update(v for v in value if type(v) is int)
            ids.update(ids_in(value))
    elif isinstance(report, list):
        for value in report:
            ids.update(ids_in(value))
    return ids


def fixed_rosters(shots):
    fresh = _read(FRESH)["records"]
    held = [(r["group"], r["id"]) for r in fresh[0]["rows"]]
    if len(set(sid for _, sid in held)) != 136 or len(held) != 136:
        raise ValueError("Unexpected frozen recorded test roster")
    if any(held != [(r["group"], r["id"]) for r in m["rows"]] for m in fresh):
        raise ValueError("The frozen recordings differ by mount")
    names = sorted(set(PRIOR.glob("recorded-*.json")) | set(PRIOR.glob("recorded-*.json.gz")))
    excluded = {sid for _, sid in held}
    for path in names:
        excluded.update(ids_in(_read(path)) & shots.keys())
    return held, excluded, names


def schedule(shots, excluded, arena):
    prohibited = {tuple(np.round(features(shots[sid]), 6)) for sid in excluded}
    rng = np.random.default_rng(SEED)
    roster = [sid for sid, shot in shots.items()
              if sid not in excluded and shot["pos_y"] > 0 and shot["vel_y"] < 0]
    bank = poses(arena)
    model = table_model()
    flight = json.loads((PRIOR / "recorded-flight.json").read_text())
    coeff = flight["held_scores"]["match_fit"]["coefficients"]
    course = []
    screened = 0
    for sid in rng.permutation(roster):
        shot = shots[int(sid)]
        identity = tuple(np.round(features(shot), 6))
        if identity in prohibited:
            continue
        screened += 1
        options = suggest(bank, [r[:3] for r in incoming(model, shot, coeff)])
        if options:
            prohibited.add(identity)
            course.append((shot, options))
            if len(course) == FEEDS:
                break
    if len(course) != FEEDS:
        raise ValueError("Insufficient eligible practice feeds")
    print("Course", FEEDS, "from", screened, "ball-only geometric screens", flush=True)
    return course, screened


def teach(data):
    shot, options = data
    arena = ReferenceReceiver()
    trials = []
    example = None
    try:
        for goal, loss in options:
            result, _ = episode(arena, shot, goal)
            trials.append(dict(goal=goal.tolist(), rank_loss=loss, **result))
            if result["legal"]:
                replay, key = episode(arena, shot, goal, collect=True)
                if replay != result or key is None:
                    raise ValueError("Teaching example did not reproduce with camera observations")
                example = dict(key=key.tolist(), goal=goal.tolist())
                break
    finally:
        arena.close()
    return dict(id=shot["id"], trials=trials, example=example)


def evaluate(arena, shots, held, keys, goals, scale, old_count, source):
    axes = arena._arm_ids[1][2][:5]
    bounds = arena._joint_low[axes], arena._joint_high[axes]
    rig = decoder()
    grip = Grip(arena.model)
    rot = np.asarray(source[MOUNT]["rotation_from_nominal"])
    cases = []
    for mount in ("nominal", MOUNT):
        grip.set(*MOUNTS[mount])
        bank = goals if mount == "nominal" else np.stack([transport(rig, g, rot, bounds) for g in goals])
        rows = []
        for group, sid in held:
            shot = shots[sid]
            blank, key = episode(arena, shot, collect=True)
            outcomes, scores = {}, {}
            for mode, count in (("original", old_count), ("expanded", len(bank))):
                if key is None:
                    scores[mode] = dict(choice=None, **blank)
                    continue
                choice = int(np.argmin(np.linalg.norm((keys[:count] - key) / scale, axis=1)))
                if choice not in outcomes:
                    result, _ = episode(arena, shot, bank[choice])
                    if result["legal"]:
                        replay, _ = episode(arena, shot, memory=(keys[:count], bank[:count], scale))
                        if replay != result:
                            raise ValueError("Held legal return changed on camera replay")
                    outcomes[choice] = result
                scores[mode] = dict(choice=choice, **outcomes[choice])
            rows.append(dict(group=group, id=sid, results=scores))
        cases.append(dict(mount=mount, rows=rows, goals=bank.tolist()))
        print(mount, {g: {mode: sum(r["results"][mode]["legal"] for r in rows if r["group"] == g)
                         for mode in ("original", "expanded")} for g in ("nearby", "unrestricted")}, flush=True)
    return cases


def run(path, workers):
    shots, (keys, goals, scale) = load_match(path)
    held, exclusion, reports = fixed_rosters(shots)
    old_count = len(goals)
    report = dict(
        seed=SEED, feed_source_sha256=_hash(path), held_source_sha256=_hash(FRESH),
        optical_source_sha256=_hash(OPTICS), mount=MOUNT, excluded_ids=sorted(exclusion),
        exclusion_receipts=[p.name for p in reports], held_roster=held, original_count=old_count,
        training=[], assessment=[],
    )
    source = _read(OPTICS)["mounts"]
    arena = ReferenceReceiver()
    try:
        course, screened = schedule(shots, exclusion, arena)
        report.update(ball_only_screened=screened, training_roster=[s["id"] for s, _ in course])
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            for result in pool.map(teach, course):
                report["training"].append(result)
                print("Teach", result["id"], len(result["trials"]), result["example"] is not None, flush=True)
                _write(RECEIPT, report)
        examples = [r for r in report["training"] if r["example"] is not None]
        if not examples:
            print("No additional legal demonstrations", flush=True)
            return
        keys = np.vstack([keys, *[r["example"]["key"] for r in examples]])
        goals = np.vstack([goals, *[r["example"]["goal"] for r in examples]])
        np.savez_compressed(CHECKPOINT, keys=keys, goals=goals, scale=scale,
                            original_count=old_count, added_ids=np.array([r["id"] for r in examples]))
        report["checkpoint_sha256"] = _hash(CHECKPOINT)
        print("Examples", len(examples), "Practical trials", sum(len(r["trials"]) for r in report["training"]), flush=True)
        report["assessment"] = evaluate(arena, shots, held, keys, goals, scale, old_count, source)
        _write(RECEIPT, report)
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.feeds, args.workers)
