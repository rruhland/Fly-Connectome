"""Repeat the optical second-return test on independent original recorded launches."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .tt4_transfer_trial import _hash, _read, _write
from .tt25_pose_demonstrations import poses
from .tt26_grip_course import PRIOR, Grip, ReferenceReceiver, load_match
from .tt26_match_continuation import SOURCE as VARIANTS
from .tt26_match_continuation import practice
from .tt27_reply import DEV_DONORS, HELD_DONORS, OUT, evaluate, schedule

SOURCE = PRIOR / "recorded-live-opponent.json.gz"


def recorded(path):
    shots, _ = load_match(path)
    return [dict(angle=0., idx=i, shot=shots[row["id"]], results=dict(bounce=row))
            for i, row in enumerate(_read(SOURCE)["cases"]) if row["replies"] > 0]


def transported(memory, grip, model, angle):
    """Keep the demonstration's paddle center and normal after a known mount change."""
    homes = []
    for e in memory:
        grip.set(e["angle"])
        q = e["q"]
        center, normal = model.face(q)
        grip.set(angle)

        def residual(candidate):
            p, n = model.face(candidate)
            return np.r_[(p - center) / .01, (n - normal) / .035, (candidate - q) / .85]

        start = np.clip(q, model.low + 1e-6, model.high - 1e-6)
        fit = least_squares(residual, start, bounds=(model.low, model.high), max_nfev=45)
        if np.linalg.norm(residual(fit.x)[:6]) > 3.:
            continue
        goal, impact, rate, lead = e["action"]
        shift = fit.x - q
        moved = np.asarray(goal) + shift
        homes.append(dict(e, angle=angle, action=(moved.tolist(), impact, rate, lead), q=fit.x))
    return homes


def feasibility(cases, destination, upstream):
    flight = json.loads((PRIOR / "recorded-flight.json").read_text())
    coeff = flight["held_scores"]["match_fit"]["coefficients"]
    arena = ReferenceReceiver()
    arena.max_steps = 160
    grip = Grip(arena.model)
    banks, results = {}, []
    try:
        for case in cases:
            angle = case["angle"]
            grip.set(angle)
            if angle not in banks:
                banks[angle] = poses(arena)
            result = practice(arena, case, banks[angle], coeff)
            results.append(dict(angle=angle, idx=case["idx"], id=case["shot"]["id"], **result))
            print("feasible", angle, case["idx"], case["shot"]["id"], result["trigger"], len(result["trials"]),
                  sum(r["contact"] for r in result["trials"]), bool(result["selected"]), flush=True)
            _write(destination, dict(privileged=True, held_opening_source_sha256=_hash(upstream), cases=results))
    finally:
        arena.close()


def run(path, stage, reuse_reply=False):
    if stage == "held-feasibility":
        held = [c for c in schedule() if c["shot"]["id"] in HELD_DONORS]
        feasibility(held, OUT / "reply-held-feasibility.json.gz", VARIANTS)
        return
    cases = recorded(path)
    traces = OUT / "reply-traces.json.gz"
    if stage == "recorded":
        if set(DEV_DONORS) & {c["shot"]["id"] for c in cases}:
            raise ValueError("The original-recording assessment includes a demonstration donor")
        evaluate(traces, OUT / "reply-recorded.json.gz", ("recall", "track", "shadow"), cases=cases,
                 transport=transported, upstream=SOURCE, reuse_reply=reuse_reply)
    else:
        feasibility(cases, OUT / "reply-recorded-feasibility.json.gz", SOURCE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path, nargs="?")
    parser.add_argument("stage", choices=("recorded", "recorded-feasibility", "held-feasibility"))
    parser.add_argument("--reuse-reply", action="store_true")
    options = parser.parse_args()
    if options.stage != "held-feasibility" and options.feeds is None:
        parser.error("provide the recorded feed archive")
    run(options.feeds, options.stage, options.reuse_reply)
