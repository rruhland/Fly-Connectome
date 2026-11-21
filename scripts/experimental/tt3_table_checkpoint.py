"""Save the learned player between table points and verify its next physical action tape."""

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np

from tt3_arena import TT3Arena, heldout_cases
from tt3_cycle import CyclePlayer, ConditionedNumeric
from tt3_learning import play_point
from tt3_recovery_learning import DelayMemory
from tt3_visual import RoleModel
from scripts.experimental.tt3_role_discovery import DiscoveryBinder


def make_player(seed, roles, memory, timing, clock):
    player = CyclePlayer(
        roles,
        memory,
        seed=seed,
        forecast=True,
        recall=True,
        learn_roles=True,
        require_rebound=False,
        timing_memory=timing,
        timing_seed=seed + 72401,
    )
    if clock is not None:
        player.binder.binder = DiscoveryBinder(roles, clock, learn=True)
    return player


def export_state(player, seed, clock):
    body = player.motor.body
    vision = player.binder.binder
    evidence = player.evidence
    return {
        "schema": "numeric-table-player-v1",
        "seed": seed,
        "roles": asdict(vision.model),
        "role_prior": asdict(vision._initial),
        "clock": clock,
        "memory": player.memory.carried.state(),
        "timing": player.timing_memory.state(),
        "policy_rng": player.rng.bit_generator.state,
        "timing_rng": player.timing_rng.bit_generator.state,
        "visual_examples": {
            "control": np.asarray(vision._bank.control).tolist(),
            "relevance": np.asarray(vision._bank.relevance).tolist(),
            "null_control": list(map(float, vision._bank.null_control)),
        },
        "local_colors": [
            [(int(color), int(n)) for color, n in counter.items()]
            for counter in player.binder.votes
        ],
        "motor": player.motor.state(),
        "body_cells": [
            [list(map(int, key)), q.tolist(), p.tolist(), count]
            for key, (q, p, count) in body.cells.items()
        ],
        "body_fitted": None if body.fitted is None else body.fitted.tolist(),
        "body_changed": body.changed,
        "accelerations": [a.tolist() for a in evidence.accelerations],
        "gravity": None if evidence.gravity is None else evidence.gravity.tolist(),
        "surfaces": [
            [k, sign, list(values)] for (k, sign), values in evidence.surfaces.items()
        ],
    }


def load_player(state):
    if state["schema"] != "numeric-table-player-v1":
        raise ValueError("unknown table player schema")
    timing = DelayMemory(state["timing"]["delays"])
    timing.uses[:] = state["timing"]["uses"]
    timing.continuation[:] = state["timing"]["continuations"]
    timing.public_score[:] = state["timing"]["public_scores"]
    player = make_player(
        state["seed"],
        RoleModel(**state["role_prior"]),
        ConditionedNumeric.restore(state["memory"]),
        timing,
        state["clock"],
    )
    vision = player.binder.binder
    vision.model = RoleModel(**state["roles"])
    for label in ("control", "relevance"):
        setattr(
            vision._bank,
            label,
            [np.asarray(value) for value in state["visual_examples"][label]],
        )
    vision._bank.null_control = list(state["visual_examples"]["null_control"])
    player.binder.votes = tuple(Counter(dict(items)) for items in state["local_colors"])
    player.motor.restore_local(state["motor"])
    body = player.motor.body
    body.cells = {
        tuple(key): (np.asarray(q), np.asarray(p), count)
        for key, q, p, count in state["body_cells"]
    }
    body.fitted = (
        None if state["body_fitted"] is None else np.asarray(state["body_fitted"])
    )
    body.changed = state["body_changed"]
    evidence = player.evidence
    evidence.accelerations.extend(np.asarray(a) for a in state["accelerations"])
    evidence.gravity = (
        None if state["gravity"] is None else np.asarray(state["gravity"])
    )
    for k, sign, values in state["surfaces"]:
        evidence.surfaces[k, sign].extend(tuple(entry) for entry in values)
    player.rng.bit_generator.state = state["policy_rng"]
    player.timing_rng.bit_generator.state = state["timing_rng"]
    return player


def qualify(receipt_path, output):
    with gzip.open(receipt_path, "rt") as stream:
        source = json.load(stream)
    course = source["matches"][0]["trained"]
    seed = source["seed"]
    player = make_player(
        seed,
        RoleModel(**source["roles"]),
        ConditionedNumeric(source["source"]),
        DelayMemory(),
        source.get("reward_clock"),
    )
    cases = heldout_cases(seed, len(course["points"]) + 1)
    view = cases[0].view_seed
    for case, original in zip(cases[:-1], course["points"], strict=True):
        row = play_point(TT3Arena(case.seed, view_seed=view), player)
        if row["action_sha256"] != original["action_sha256"]:
            raise ValueError("reproduced course diverged from recorded actions")
    # Serialize through JSON before checking continuation, including the local motor map.
    state = export_state(player, seed, source.get("reward_clock"))
    restored = load_player(json.loads(json.dumps(state, allow_nan=False)))
    final_case = cases[-1].seed
    live = play_point(TT3Arena(final_case, view_seed=view), player)
    resumed = play_point(TT3Arena(final_case, view_seed=view), restored)
    for key in ("action_sha256", "events", "public_reward", "motor_samples"):
        if live[key] != resumed[key]:
            raise ValueError(
                f"{key} diverged after restoring the point-boundary checkpoint"
            )
    state["next_point_check"] = {
        "seed": final_case,
        "view_seed": view,
        "action_sha256": live["action_sha256"],
        "frames": live["frames"],
        "reward": live["public_reward"],
    }
    with gzip.open(output, "wt") as stream:
        json.dump(state, stream, separators=(",", ":"), allow_nan=False)
    print(state["next_point_check"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    qualify(args.receipt, args.out)
