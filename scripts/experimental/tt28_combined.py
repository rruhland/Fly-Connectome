"""Test solo blade registration followed by three optical regulation probes."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import Grip, ReferenceReceiver, load_match
from .tt26_optical_transfer import decoder, transport
from .tt27_world_camera import hit
from .tt28_dose import MOUNTS
from .tt28_outgoing import (
    COVERAGE,
    OUT,
    POSES,
    PROBES,
    learn_outgoing,
    learn_reward,
    play,
    variant,
)

SEED = 357102482
PER_GROUP = 6
HELD = 6
PROBE_BIASES = PROBES[:3]


def select_donors(archive):
    """Enroll using historical physical contact only, with fixed stratum counts."""
    rng = np.random.default_rng(SEED)
    selected = []
    for group in ("nearby", "unrestricted"):
        eligible = [
            row for row in archive if row["group"] == group and any(a["contact"] for a in row["scores"])
        ]
        for i in rng.choice(len(eligible), PER_GROUP, replace=False):
            selected.append(eligible[i])
    return selected


def practice(arena, feeds, key, goal, scale):
    results = [
        play(arena, feed, key, [goal], scale, bias) for feed, bias in zip(feeds, PROBE_BIASES, strict=True)
    ]
    public = [{k: r[k] for k in ("bias", "legal", "forecast")} for r in results]
    policies = {"frozen": 0.0, "score": learn_reward(public), "outgoing": learn_outgoing(public)}
    return results, policies


def course(arena, tool, sources, key, goals, scale, training, held):
    practice_rows, policies = {}, {}
    for stage in goals:
        practice_rows[stage], policies[stage] = practice(arena, training, key, goals[stage], scale)
    assigned = {
        f"{stage}_{mode}": (stage, bias) for stage in goals for mode, bias in policies[stage].items()
    }
    held_rows = []
    replayed = 0
    for feed in held:
        tests, executed = {}, {}
        for name, (stage, bias) in assigned.items():
            goal = goals[stage].copy()
            goal[4] += bias
            action = tuple(goal)
            if action not in executed:
                score = hit(arena, feed, goal)
                if hit(arena, feed, goal, tape=score["actions"]) != score:
                    raise ValueError(f"Motor replay diverged on {tool}/{feed['id']}/{name}")
                executed[action] = score
                replayed += 1
            tests[name] = executed[action]
        held_rows.append(tests)
    return {
        "tool": tool,
        "source": sources,
        "goals": {stage: goal.tolist() for stage, goal in goals.items()},
        "practice": practice_rows,
        "policies": policies,
        "held": held_rows,
        "replayed": replayed,
    }


def summary(rows):
    result = {}
    for tool in MOUNTS:
        tests = [r for row in rows for r in row["mounts"][tool]["held"]]
        result[tool] = {
            mode: {metric: sum(r[mode][metric] for r in tests) for metric in ("legal", "contact")}
            for mode in tests[0]
        }
    return result


def run(path):
    shots, _ = load_match(path)
    with np.load(POSES) as archive:
        keys, library, scale = (archive[n] for n in ("keys", "goals", "scale"))
    donors = select_donors(_read(COVERAGE)["rows"])
    checkpoints = {}
    for tool in MOUNTS:
        checkpoint = _read(OUT / f"dose-solo-{tool}.json.gz")
        source = checkpoint["checkpoints"]["16"]
        checkpoints[tool] = {k: source[k] for k in ("rotation", "samples")}
        checkpoints[tool]["seed"] = checkpoint["source_seed"]
        checkpoints[tool]["checkpoint_sha256"] = digest_json(checkpoints[tool])

    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    rig = decoder()
    axes = arena._arm_ids[1][2][:5]
    bounds = arena._joint_low[axes], arena._joint_high[axes]
    receipt = {
        "seed": SEED,
        "expected_donors": len(donors),
        "probes": PROBE_BIASES,
        "held_per_donor_and_mount": HELD,
        "sources": checkpoints,
        "feeds_sha256": _hash(path),
        "coverage_sha256": _hash(COVERAGE),
        "action_library_sha256": _hash(POSES),
        "rows": [],
    }
    try:
        for row in donors:
            shot = shots[row["id"]]
            grip.set(0.0)
            _, signature = episode(arena, shot, collect=True)
            supported = np.array([i for i, r in enumerate(row["scores"]) if r["contact"]])
            distance = np.linalg.norm((keys[supported] - signature) / scale, axis=1)
            idx = int(supported[np.argmin(distance)])
            base = np.clip(library[idx], *bounds)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, shot["id"]]))
            train = [variant(shot, rng, 0.70) for _ in PROBE_BIASES]
            held = [variant(shot, rng, 1.0 + 0.35 * (tick % 2)) for tick in range(HELD)]
            results = {}
            for tool, mount in MOUNTS.items():
                grip.set(*mount)
                rotation = np.asarray(checkpoints[tool]["rotation"])
                solo = transport(rig, base, rotation, bounds)
                goals = {"zero": base, "solo": solo}
                results[tool] = course(
                    arena, tool, checkpoints[tool], keys[[idx]], goals, scale, train, held
                )
            receipt["rows"].append(
                {
                    "donor": row["id"],
                    "group": row["group"],
                    "anchor": idx,
                    "historical_contacts": len(supported),
                    "training_feeds": train,
                    "held_feeds": held,
                    "mounts": results,
                }
            )
            receipt["summary"] = summary(receipt["rows"])
            _write(OUT / "combined-held.json.gz", receipt)
            print(
                len(receipt["rows"]), "/", len(donors), row["id"], summary(receipt["rows"][-1:]), flush=True
            )
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    run(parser.parse_args().feeds)
