"""Compare frozen rebound aiming with matched controls on complete native boards."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import random
from statistics import median

import numpy as np
import torch

from arcade_transfer_schedule import CHECKPOINT_SHA256, canonical_json
from breakout_arena import BreakoutEpisode, BreakoutLayout, breakout_schedule
from breakout_visual_rebound import VisualReboundPolicy
from run_arcade_alternation import COURSE_SEEDS, schedule as alternation_schedule
from run_arcade_transfer_qualification import _candidate_receipt, _git, _paired_difference
from run_breakout_brick_placement import COHORT_SEEDS
from run_breakout_visual_rebound import CHECKPOINT, TEST_ARMS, play, save, summarize


ROOT = Path(__file__).resolve().parents[2]
BOARDS = 12
SECONDS = 75
ENVIRONMENT_SEED = 734273193
POLICY_SEED = 1239496674


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _layout_seed_cells(seed, split="heldout"):
    return BreakoutLayout.generate(seed, split=split).cells


def registration(pilot):
    pilot_spec = (pilot / "spec.json").read_bytes()
    pilot_practice = (pilot / "practice-summary.json").read_bytes()
    pilot_cases = json.loads(pilot_spec)["phase_cases"]
    endpoint = json.loads(pilot_practice)["final_state"]
    empty = VisualReboundPolicy(updates_enabled=False).to_state()
    if len(pilot_cases["practice"]) != 12 or min(endpoint["visual_updates"], endpoint["score_updates"]) < 1:
        raise ValueError("the learned endpoint must come from the completed 12-life visual practice")

    blind_path = ROOT / "artifacts/m2/2026-11-20-breakout-sealed-schedule.json"
    blind = json.loads(blind_path.read_bytes())
    seen = {BreakoutLayout(case["layout"]["cells"]).cells for group in pilot_cases.values() for case in group}
    seen.update(_layout_seed_cells(life["layout_seed"]) for course in blind["courses"] for life in course["lives"])
    for seeds in COHORT_SEEDS.values():
        for seed, split in zip(seeds, ("practice", "heldout", "heldout"), strict=True):
            seen.update(_layout_seed_cells(case.layout_seed, split) for case in breakout_schedule(seed, 256, split=split))
    for course in COURSE_SEEDS:
        seen.update(_layout_seed_cells(row["parameters"]["layout_seed"])
                    for row in alternation_schedule(course, 64) if row["game"] == "breakout")

    policy_rng = random.Random(POLICY_SEED)
    cases = []
    pool = breakout_schedule(ENVIRONMENT_SEED, 256)
    for pair in zip(pool[::2], pool[1::2], strict=True):
        layouts = [_layout_seed_cells(case.layout_seed) for case in pair]
        if layouts[0] in seen or layouts[1] in seen or layouts[0] == layouts[1]:
            continue
        seen.update(layouts)
        for case, cells in zip(pair, layouts, strict=True):
            cases.append({"id": len(cases) + 1, "episode": asdict(case), "layout": {"cells": cells},
                          "board": "full", "agent_seed": policy_rng.randrange(1, 2**31)})
        if len(cases) == BOARDS:
            break
    if len(cases) != BOARDS or sum(case["episode"]["serve_side"] for case in cases):
        raise ValueError("expected six left and six right full-board starts")

    return dict(schema="breakout-visual-rebound-clearance-v1", source_commit=_git("rev-parse", "HEAD"),
                checkpoint_sha256=CHECKPOINT_SHA256, duration_seconds=SECONDS, arms=TEST_ARMS,
                primary="paired native full boards cleared before natural loss or time limit",
                environment_seed=ENVIRONMENT_SEED, policy_seed=POLICY_SEED,
                pilot_spec_sha256=_digest(pilot_spec), pilot_practice_sha256=_digest(pilot_practice),
                blind_schedule_sha256=_digest(blind_path.read_bytes()),
                trained_state=endpoint, trained_state_sha256=_digest(canonical_json(endpoint)),
                empty_state_sha256=_digest(canonical_json(empty)), cases=cases)


def _play_pair(case, checkpoint, spec, output):
    torch.set_num_threads(1)
    public_case = dict(case, episode=BreakoutEpisode(**case["episode"]), layout=BreakoutLayout(**case["layout"]))
    rows = {}
    for arm in TEST_ARMS:
        if arm == "visual_trained":
            policy = VisualReboundPolicy.from_state(spec["trained_state"], updates_enabled=False)
        elif arm == "visual_empty":
            policy = VisualReboundPolicy(updates_enabled=False)
        else:
            policy = None
        initial = None if policy is None else _digest(canonical_json(policy.to_state()))
        row = play(public_case, checkpoint, policy, spec["duration_seconds"])
        final = None if policy is None else _digest(canonical_json(policy.to_state()))
        expected = {"visual_trained": spec["trained_state_sha256"], "visual_empty": spec["empty_state_sha256"],
                    "center": None}[arm]
        if initial != expected or final != initial or row["score_updates"]:
            raise ValueError("frozen shot weights changed or failed their digest check")
        rows[arm] = dict(row, arm=arm, id=case["id"], initial_shot_sha256=initial, final_shot_sha256=final)
    _audit_pair(rows, case, spec["duration_seconds"])
    save(output / f"board-{case['id']:02}.json", rows)
    return rows


def _audit_pair(rows, case, duration):
    first_trajectory = None
    for row in rows.values():
        if (row["case"] != case["episode"] or canonical_json(row["layout"]) != canonical_json(case["layout"])
                or row["agent_seed"] != case["agent_seed"] or row["initial_bricks"] != len(case["layout"]["cells"])):
            raise ValueError("mismatched paired environment, layout or policy seed")
        opening = row["contacts"][0]
        trajectory = (opening["tick"], opening["time_seconds"], opening["ball"], opening["velocity"])
        if first_trajectory is not None and trajectory != first_trajectory:
            raise ValueError("initial ball trajectory changed across paired arms")
        first_trajectory = trajectory
        if (sum(bool(row[field]) for field in ("cleared", "life_lost", "truncated")) != 1
                or row["cleared"] != (row["remaining_bricks"] == 0)
                or row["score"] != len(row["brick_events"])
                or row["score"] != row["initial_bricks"] - row["remaining_bricks"]
                or row["post_own_bricks"] != sum(event["after_return"] for event in row["brick_events"])
                or row["own_returns"] != sum(event["success"] for event in row["contacts"])
                or row["own_opportunities"] != len(row["contacts"])
                or row["truncated"] and row["ticks"] != 120 * duration):
            raise ValueError("native end-state or event counts disagree")


def _summarize(rows, duration):
    all_arms = {arm: [pair[arm] for pair in rows] for arm in TEST_ARMS}
    cutoff = [15, 30, 45, 60, duration]
    groups = {}
    for arm, lives in all_arms.items():
        clears = [life["elapsed_seconds"] for life in lives if life["cleared"]]
        capped = [life["elapsed_seconds"] if life["cleared"] else duration for life in lives]
        groups[arm] = dict(summarize(lives),
                           clear_at_seconds=[{"seconds": t, "cleared": sum(time <= t for time in clears)} for t in cutoff],
                           time_to_clear_seconds=[life["elapsed_seconds"] if life["cleared"] else None for life in lives],
                           median_seconds_if_cleared=median(clears) if clears else None,
                           mean_capped_clear_seconds=float(np.mean(capped)))
    contrasts = {}
    trained = all_arms["visual_trained"]
    draws = np.random.default_rng(36916763).integers(len(trained), size=(8192, len(trained)))
    for arm in TEST_ARMS[1:]:
        other = all_arms[arm]
        pairs = list(zip(trained, other, strict=True))
        cleared = [int(a["cleared"]) - int(b["cleared"]) for a, b in pairs]
        post_return = [a["post_own_bricks"] - b["post_own_bricks"] for a, b in pairs]
        capped_seconds = [(b["elapsed_seconds"] if b["cleared"] else duration)
                          - (a["elapsed_seconds"] if a["cleared"] else duration) for a, b in pairs]
        simulated_rates = []
        for group in (trained, other):
            bricks = np.array([life["post_own_bricks"] for life in group])[draws].sum(axis=1)
            times = np.array([life["elapsed_seconds"] for life in group])[draws].sum(axis=1)
            simulated_rates.append(60 * bricks / times)
        contrasts[arm] = dict(boards_cleared_difference=sum(cleared),
                              board_clear_fraction_gain=_paired_difference(cleared),
                              post_return_bricks_per_life_gain=_paired_difference(post_return),
                              capped_seconds_saved=_paired_difference(capped_seconds),
                              post_return_bricks_per_minute_gain=(groups["visual_trained"]["bricks_after_own_return_per_minute"]
                                                                  - groups[arm]["bricks_after_own_return_per_minute"]),
                              per_minute_gain_bootstrap_95=np.quantile(simulated_rates[0] - simulated_rates[1],
                                                                      [.025, .975]).tolist())
    return {"by_arm": groups, "paired_trained_vs": contrasts}


def run(registration_path, output, workers):
    register_bytes = registration_path.read_bytes()
    spec = json.loads(register_bytes)
    checkpoint_bytes = CHECKPOINT.read_bytes()
    if _digest(checkpoint_bytes) != spec["checkpoint_sha256"] or _digest(canonical_json(spec["trained_state"])) != spec["trained_state_sha256"]:
        raise ValueError("the frozen Pong checkpoint or visual endpoint does not match registration")
    if spec["arms"] != list(TEST_ARMS) or spec["duration_seconds"] != SECONDS or len(spec["cases"]) != BOARDS:
        raise ValueError("the registered comparison must include all arms, boards and full horizon")
    receipt = dict(_candidate_receipt(spec["source_commit"]), registration_sha256=_digest(register_bytes),
                   checkpoint_sha256=spec["checkpoint_sha256"], trained_state_sha256=spec["trained_state_sha256"],
                   empty_state_sha256=spec["empty_state_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "registration.json").write_bytes(register_bytes)
    save(output / "receipt.json", receipt)
    checkpoint = json.loads(checkpoint_bytes)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(_play_pair, case, checkpoint, spec, output) for case in spec["cases"]]
        rows = [job.result() for job in jobs]
    save(output / "summary.json", _summarize(rows, spec["duration_seconds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    register = modes.add_parser("register")
    register.add_argument("pilot", type=Path)
    register.add_argument("registration", type=Path)
    evaluate = modes.add_parser("evaluate")
    evaluate.add_argument("registration", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.mode == "register":
        with args.registration.open("xb") as file:
            file.write(canonical_json(registration(args.pilot)))
    else:
        run(args.registration, args.output, args.workers)


if __name__ == "__main__":
    main()
