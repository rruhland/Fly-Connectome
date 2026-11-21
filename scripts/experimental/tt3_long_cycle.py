"""Continue the saved arm from 20 to 64 points and pair the resulting state against native Pong."""

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from tt3_arena import TT3Arena, heldout_cases
from tt3_cycle import metrics
from tt3_learning import play_point
from tt3_pong import assess_retention, cases_from_seeds, evaluate_pong
from tt3_pong_probe import ShorterDwell
from tt3_table_checkpoint import load_player


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--native-seed", type=int, default=81752634)
    args = parser.parse_args()
    with gzip.open(args.checkpoint, "rt") as stream:
        checkpoint = json.load(stream)
    with gzip.open(args.receipt, "rt") as stream:
        study = json.load(stream)
    player = load_player(checkpoint)
    seed = checkpoint["seed"]
    prior = study["matches"][0]["trained"]["points"]
    if not len(prior) < args.points or seed != study["seed"]:
        raise ValueError("an extended course from the same installation is required")
    cases = heldout_cases(seed, args.points)
    view = cases[0].view_seed
    rows = []
    for case in cases[len(prior) :]:
        row = play_point(TT3Arena(case.seed, view_seed=view), player)
        rows.append({key: row[key] for key in prior[0]})
    if rows[0]["action_sha256"] != checkpoint["next_point_check"]["action_sha256"]:
        raise ValueError("resumed course diverged")
    torch.set_num_threads(1)
    native_seed = study.get("native", {}).get("seed", args.native_seed)
    rng = np.random.default_rng(native_seed)
    starts = cases_from_seeds(rng.choice(100_000_000, 48, replace=False) + 1740_000_000)
    roles = player.binder.model
    memory = player.memory.carried
    native_after = evaluate_pong(
        starts,
        roles,
        memory,
        max_ticks=1440,
        points_per_match=5,
        motor_factory=ShorterDwell,
    )
    expected = len(prior) + len(rows)
    if int(sum(memory.shared.counts) - sum(study["source"]["counts"])) != expected:
        raise ValueError("missing table outcome credit")
    with gzip.open(args.receipt, "rt") as stream:
        native = json.load(stream)["native"]
    if native_seed != native["seed"]:
        raise ValueError("native reference is on a different schedule")
    comparison = assess_retention(native["before_points"], native_after)
    result = dict(
        seed=seed,
        previous_points=len(prior),
        new_points=rows,
        table=metrics(prior + rows),
        last_block=metrics(rows),
        native_seed=native["seed"],
        comparison=comparison,
        native_after=native_after,
        roles=asdict(roles),
        memory=memory.state(),
        timing=player.timing_memory.state(),
    )
    with gzip.open(args.out, "wt") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
    print(
        result["table"],
        comparison["before"],
        comparison["after"],
        comparison["established"],
    )


if __name__ == "__main__":
    main()
