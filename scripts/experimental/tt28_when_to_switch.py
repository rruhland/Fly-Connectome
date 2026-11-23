"""Switch from solo practice to receiving when the optical tool estimate stabilizes."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_bounce_optics import unit, separation
from .tt26_grip_course import Grip, ReferenceReceiver, load_match
from .tt27_flight_feedback import load_bank
from .tt28_dose import NOMINAL, OUTPUT, MOUNTS, SoloLearner, roster, target_banks


def stable(directions):
    n = len(directions)
    if n < 6:
        return False, None, None
    x = np.stack(directions)
    center = unit(np.median(x, axis=0))
    rng = np.random.default_rng(593814 + n)
    resampled = np.median(x[rng.integers(0, n, (400, n))], axis=1)
    resampled /= np.linalg.norm(resampled, axis=1)[:, None]
    spread = float(np.percentile(np.degrees(np.arccos(np.clip(resampled @ center, -1, 1))), 90))
    drift = separation(unit(np.median(x[-3:], axis=0)), center)
    return spread <= 1.5 and drift <= 2.0, spread, drift


def gate(rows):
    base = _read(NOMINAL)["mounts"]["nominal"]["normal_link"]
    learner = SoloLearner(np.asarray(base))
    consecutive = 0
    log = []
    for i, r in enumerate(rows):
        impact = r["first_impact"]
        learner.update(r["action"], r["reward"], impact["raw_normal_link"] if impact else None)
        ready, spread, drift = stable(learner.directions)
        consecutive = consecutive + 1 if ready and impact else 0
        log.append(dict(n=i + 1, spread_deg=spread, drift_deg=drift, consecutive=consecutive))
        if consecutive == 2:
            return i + 1, learner.snapshot(), log
    raise ValueError("Stability threshold not reached in available source stream")


def run(path, tool):
    shots, (_, _, scale) = load_match(path)
    learner = load_bank()
    keys, goals = (np.stack([r[n] for r in learner["rows"]]) for n in ("key", "goal"))
    source = _read(OUTPUT / f"dose-solo-{tool}.json.gz")
    stop, state, log = gate(source["rows"])
    print(tool, "switch after", stop, log[-1], flush=True)
    arena = ReferenceReceiver()
    Grip(arena.model).set(*MOUNTS[tool])
    results = []
    try:
        chosen = target_banks(arena, goals, {"gate": state})["gate"]

        def select(k):
            return chosen[np.argmin(np.linalg.norm((keys - k) / scale, axis=1))]

        for group, s in roster(shots, learner):
            score, _ = episode(arena, s, choose_goal=select)
            if score["legal"] and score != episode(arena, s, choose_goal=select)[0]:
                raise ValueError("Gated checkpoint did not reproduce")
            results.append(dict(group=group, shot=s, result=score))
    finally:
        arena.close()
    _write(
        OUTPUT / f"switch-{tool}.json.gz", dict(drops=stop, snapshot=state, stability=log, results=results)
    )
    for group in sorted({r["group"] for r in results}):
        print(tool, group, sum(r["result"]["legal"] for r in results if r["group"] == group), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("tool", choices=MOUNTS)
    args = p.parse_args()
    run(args.feeds, args.tool)
