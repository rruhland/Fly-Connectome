"""Test the first six registered source learning streams on independent physical arms."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_world_arm import DIR, trial
from .tt24_world_prior import WorldPrior

TREATMENTS = ("pong", "shuffled", "cold", "matched_cold", "hold")
SEED = 693174522
SENSOR = DIR / "public-body-fused.npz"
PLAN = DIR / "native-bank-arm-plan.json"


def register():
    rng = np.random.default_rng(SEED)
    groups = [rng.integers(1, 2**31-1, 12).tolist() for _ in range(6)]
    seeds = sum(groups, [])
    if len(seeds) != len(set(seeds)):
        raise ValueError("overlapping held drops")
    plan = dict(seed=SEED, groups=groups, primary_streams=[1,2,3,4,5], replication_of_first=0,
                modes=TREATMENTS, horizon=450, sensor=SENSOR.name,
                sensor_sha256=sha256(SENSOR.read_bytes()).hexdigest(),
                outcome="mean longest consecutive physical rebound streak per drop; compare paired source training streams",
                secondary="total separate physical rebounds and fraction reaching ten",
                held_bounce_training=0, observation="wrist RGB and five encoders")
    contents = json.dumps(plan, indent=2)+"\n"
    if PLAN.exists() and PLAN.read_text() != contents:
        raise ValueError("previous registration differs")
    PLAN.write_text(contents)
    return plan


def test_group(group, plan):
    seeds = plan["groups"][group]
    arena = BounceArena(view="wrist", width=192, height=144, max_steps=plan["horizon"])
    modes, hashes = {}, {}
    try:
        for mode in TREATMENTS:
            source = None
            if mode != "hold":
                path = DIR/f"pong-world-strong-bank-{group:02d}-{mode}.npz"
                source = WorldPrior.load(path)
                hashes[mode] = sha256(path.read_bytes()).hexdigest()
            modes[mode] = [trial(arena, seed, "hold" if mode=="hold" else "prior", source,
                                 False, True, SENSOR) for seed in seeds]
            print(group, mode, sum(row["run"] for row in modes[mode]), flush=True)
    finally:
        arena.close()
    with gzip.open(DIR/f"native-bank-arm-{group:02d}.json.gz", "wt") as out:
        json.dump(dict(group=group, seeds=seeds, hashes=hashes, verified=True, results=modes),
                  out, separators=(",",":"))


def summarize(plan):
    summary = {}
    for group in range(len(plan["groups"])):
        with gzip.open(DIR/f"native-bank-arm-{group:02d}.json.gz","rt") as data:
            doc = json.load(data)
        if doc["seeds"] != plan["groups"][group] or not doc["verified"]:
            raise ValueError("unverified or wrong holdout")
        summary[group] = {mode:dict(consecutive=sum(r["run"] for r in rows),
                                     total=sum(r["score"] for r in rows),
                                     ten=sum(r["run"]>=10 for r in rows), maximum=max(r["run"] for r in rows))
                          for mode,rows in doc["results"].items()}
    primary = plan["primary_streams"]
    totals = {mode:{key:sum(summary[group][mode][key] for group in primary)
                    for key in ("consecutive","total","ten")} for mode in TREATMENTS}
    (DIR/"native-bank-arm-summary.json").write_text(json.dumps(dict(groups=summary,primary_totals=totals),
                                                               indent=2)+"\n")
    print(totals, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    plan = register()
    if not args.register:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(test_group, group, plan) for group in range(6)]
            for future in futures:
                future.result()
        summarize(plan)
