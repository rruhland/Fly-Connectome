"""Measure online motor-map learning during single-environment wrist play, including unseen rewiring."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_effects import transmission
from .tt24_motor_recall import MotorRecall, STOCK, calibrate


def changed(drop, tick):
    return (drop == 1 and tick >= 72) or 2 <= drop <= 5 or (drop == 6 and tick < 72)


class RelativeIssued:
    """The encoder prediction estimates a target relative to the first joint sample."""
    def __init__(self, policy):
        self.policy = policy
        self.origin = policy.issued.copy()

    @property
    def issued(self):
        return self.policy.issued - self.origin

    @issued.setter
    def issued(self, value):
        self.policy.issued = value + self.origin


def simulate(arena, policy, adapter, wiring, seed, drop, kind, tape=None):
    obs = arena.reset(seed=seed, split="held", render=tape is None)
    if tape is None:
        policy.begin_release(obs)
        servo = RelativeIssued(policy)
        adapter.reset(obs)
    records = []
    received = []
    while True:
        t = len(records)
        switched = changed(drop, t)
        if tape is None:
            nominal = policy.act(obs)
            if kind == "adaptive":
                drive = adapter.actuate(nominal)
            elif kind == "known" and switched:
                drive = np.clip(np.linalg.solve(wiring, nominal), -1., 1.)
            else:
                drive = nominal
            applied = wiring @ drive if switched else drive
        else:
            applied = tape[t]
        obs, reward, done = arena.step(applied, render=tape is None)
        records.append(np.asarray(applied).tolist())
        if reward:
            received.append(t)
        if tape is None and kind == "adaptive":
            adapter.observe(drive, obs, servo)
        if done:
            break
    metrics = arena.judge_metrics()
    return dict(tape=records, events=received, count=metrics["rebounds"],
                longest=metrics["longest_consecutive_rebounds"], metrics=metrics,
                recalls=adapter.events if kind=="adaptive" and tape is None else [])


def stream(index):
    observer, scale, _ = calibrate(STOCK)
    rng = np.random.default_rng(77864326 + index)
    seeds = rng.integers(1, 2**31-1, 9).tolist()
    wiring = transmission(rng)
    arena = BounceArena(view="wrist", width=192, height=144, max_steps=480)
    policy = CarriedWristPolicy()
    out = {}
    try:
        for kind in ("adaptive", "unadapted", "known"):
            adapter = MotorRecall(observer, scale)
            rows = []
            for drop, seed in enumerate(seeds):
                row = simulate(arena, policy, adapter, wiring, seed, drop, kind)
                replay = simulate(arena, policy, adapter, wiring, seed, drop, kind, row["tape"])
                if (row["metrics"] != replay["metrics"] or row["events"] != replay["events"]
                    or row["tape"] != replay["tape"]):
                    raise ValueError("motor-map physics replay mismatch")
                rows.append(row)
            out[kind] = rows
        with gzip.open(ROOT/f"carried-drive-{index}.json.gz", "wt") as f:
            json.dump(dict(seeds=seeds, wiring=wiring.tolist(), modes=out), f, separators=(",",":"))
    finally:
        arena.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("streams", type=int, nargs="*", default=list(range(4)))
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=min(3,len(args.streams))) as pool:
        list(pool.map(stream,args.streams))
