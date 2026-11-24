"""Retain a frozen incumbent when a newer bounce checkpoint fails a fresh comparison."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt28_dose import mkplan, solo_summary
from .tt29_bounce import COURSE, HELD_CAP, OUTPUT, held_score

PROMOTION = OUTPUT / "tt29_bounce_promotion.json.gz"
RETAINED = OUTPUT / "tt29_bounce_retained.json"
VALIDATION = (("4", 29102467), ("12", 29344611), ("24", 29737003))
ASSESSMENT_SEED = 29530441
VALIDATION_COUNT = 6
VALIDATION_CAP = 180
ASSESSMENT_COUNT = 8


def accept(rows, incumbent, challenger):
    old = [row["results"][incumbent] for row in rows]
    new = [row["results"][challenger] for row in rows]
    old_streak = np.array([result["streak"] for result in old])
    new_streak = np.array([result["streak"] for result in new])
    gains = int((new_streak > old_streak).sum())
    losses = int((new_streak < old_streak).sum())
    rebound_change = sum(r["rebounds"] for r in new) - sum(r["rebounds"] for r in old)
    median_change = float(np.median(new_streak) - np.median(old_streak))
    promote = gains > losses and rebound_change >= 2 and median_change >= 0
    return dict(promote=bool(promote), streak_gains=gains, streak_losses=losses,
                qualified_rebound_change=rebound_change, median_streak_change=median_change)


def run(output):
    report = _read(COURSE)
    if not report.get("held"):
        raise ValueError("The chronological course is incomplete")
    snapshots = report["snapshots"]
    source_hash = _hash(COURSE)
    reserved = {r["seed"] for r in report["practice"] + report["held"]}
    planned = [seed for _, stream in VALIDATION for seed, _ in mkplan(stream, VALIDATION_COUNT)]
    planned.extend(seed for seed, _ in mkplan(ASSESSMENT_SEED, ASSESSMENT_COUNT))
    if len(set(planned)) != len(planned) or reserved.intersection(planned):
        raise ValueError("Practice, promotion and assessment seeds must be disjoint")
    output.mkdir(parents=True, exist_ok=True)
    path = output / PROMOTION.name
    if path.exists():
        trace = _read(path)
        if trace["course_sha256"] != source_hash:
            raise ValueError("The existing promotion log belongs to another course")
    else:
        trace = dict(schema="tt29-bounce-checkpoint-promotion-1", course_sha256=source_hash,
                     rule="a positive paired streak win balance, at least two more qualified rebounds, and a nondecreasing median streak",
                     validation_cap_ticks=VALIDATION_CAP, validation_drops_per_gate=VALIDATION_COUNT,
                     assessment_cap_ticks=HELD_CAP, assessment_seed=ASSESSMENT_SEED,
                     rounds=[], retained=["0"], verified_validation=0)
    rounds = trace["rounds"]
    incumbent = trace["retained"][-1]
    for challenger, seed in VALIDATION[len(rounds):]:
        rows = held_score({incumbent: snapshots[incumbent], challenger: snapshots[challenger]},
                          seed=seed, held_count=VALIDATION_COUNT, cap=VALIDATION_CAP)
        decision = accept(rows, incumbent, challenger)
        rounds.append(dict(incumbent=incumbent, challenger=challenger, seed=seed, rows=rows, **decision))
        if decision["promote"]:
            incumbent = challenger
        trace["retained"].append(incumbent)
        trace["verified_validation"] += 2 * len(rows)
        _write(path, trace)
        print("promotion", challenger, "retained", incumbent, decision, flush=True)
    deployed = output / RETAINED.name
    _write(deployed, snapshots[incumbent])
    if "held" not in trace:
        stages = {d: snapshots[d] for d in dict.fromkeys(trace["retained"])}
        trace["held"] = held_score(stages, seed=ASSESSMENT_SEED, held_count=ASSESSMENT_COUNT)
    trace.update(held_summary=solo_summary(trace["held"]), verified_held=len(trace["held"]) * len(trace["held"][0]["results"]),
                 held_seed=ASSESSMENT_SEED, published_checkpoint_sha256=_hash(deployed))
    _write(path, trace)
    print("untouched promotion assessment", trace["held_summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
