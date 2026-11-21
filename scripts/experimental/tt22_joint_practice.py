"""Jointly learn a visual lateral preference and a physical SO-101 stroke from point scores."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import ROOT, _read, _write
from .tt19_flight import optical_start
from .tt20_natural import NaturalViewTransport
from .tt22_motor_transfer import MotorPrimitive, lateral_goal
from .tt22_motor_transfer_trial import MotorArena, _observed
from .tt22_openloop import stroke

READS = 8
CHUNK = 8
PARAMETER_SCALE = np.array([0.68, 0.75, 0.8, 0.24, 0.75, 0.70])
LIMITS = np.array([2.0, 2.0, 3.0, 0.9, 3.0, 3.0])


def observed_prefix(arena, seed, lane):
    arena.lane = lane
    arena.reset(seed=seed, server=-1, render=False)
    tracker = NaturalViewTransport(
        "fixed", lens_fov=arena.lens_intrinsics(), flight=False, associate=True
    ).tracker()
    history = []
    ready = None
    for t in range(READS):
        tracker.update(arena.render_cameras(), tick=t)
        p = _observed(tracker.ball)
        q = _observed(tracker.actor)
        history.append((t, p))
        if ready is None and q is not None:
            ready = q
        arena.step(np.zeros(6), render=False)
    mv = optical_start(history, 0.0109)
    return dict(
        seed=seed,
        lane=lane,
        ready=None if ready is None else ready.tolist(),
        mover=None if mv is None else [np.asarray(a).tolist() for a in mv],
        rgb=tracker.digest.hexdigest(),
    )


def point(arena, template, theta, obs):
    # The entire optical prefix uses identical zero controls on every model.
    arena.lane = obs["lane"]
    arena.reset(seed=obs["seed"], server=-1, render=False)
    for _ in range(READS):
        arena.step(np.zeros(6), render=False)
    macro = template.copy()
    macro[10] += theta[2]
    macro[5:10] *= 1 + theta[3]
    macro[8] += theta[4]
    macro[6] += theta[5]
    controller = MotorPrimitive(macro, None)
    goal = None
    if obs["mover"] is not None:
        mover, velocity = np.array(obs["mover"])
        ready = controller.ready_world if obs["ready"] is None else np.array(obs["ready"])
        goal, _ = lateral_goal(ready, mover, velocity, np.r_[0.0, theta[:2]])
        controller._translate(goal, ready)
    target = np.zeros(5)
    for t in range(READS, arena.max_steps):
        change = np.clip(stroke(controller.parameters, t) - target, -1.0, 1.0)
        target += change
        _, _, done, cap, info = arena.step(np.r_[change, 0.0], render=False)
        if done or cap:
            break
    return dict(
        seed=obs["seed"],
        lane=obs["lane"],
        goal=goal,
        legal=info["legal_returns"],
        contact=info["contacts"],
        rival=info["rival_returns"],
        ticks=arena.steps,
        reason=info["reason"],
    )


def draw(rng, count, span):
    sides = np.resize(np.array([-1.0, 1.0]), count)
    rng.shuffle(sides)
    return [(int(rng.integers(2**31)), float(s * rng.uniform(*span))) for s in sides]


def run(args):
    observer = observed_prefix
    if args.generic:
        from .tt22_motion_prefix import prefix

        observer = prefix
    roots = ROOT / "artifacts"
    template = np.array(json.loads((roots / "tt22/openloop-practice.json").read_text())["parameters"])
    source = json.loads((roots / "tt3/discovered-numeric-source.json").read_text())
    best = np.argmax(np.array(source["totals"]) / np.array(source["counts"]))
    gains = np.array(source["coefficients"][best][1:])
    arena = MotorArena()
    rng = np.random.default_rng(args.seed)
    testing = draw(rng, args.held, (0.17, 0.62))
    data = dict(
        generic=args.generic,
        seed=args.seed,
        held=testing,
        read_frames=READS,
        template=template.tolist(),
        source_index=int(best),
        replicates=[],
        source_gains=gains.tolist(),
    )
    now = perf_counter()
    try:
        held = [observer(arena, s, lane) for s, lane in testing]
        population = _read(roots / "tt3/discovered-source-search.json.gz")[0]["state"]["coefficients"]
        population_order = np.random.default_rng(4730826).permutation(len(population))
        initial_cache = {}
        data["held_optics"] = held
        for rep in range(args.replicates):
            training = draw(rng, args.points, (0.12, 0.55))
            policy_rng = np.random.default_rng(int(rng.integers(2**31)))
            # All models receive the same initial body primitive, exploration draws, and new physical starts.
            mean = {"pong": np.r_[gains, np.zeros(4)], "cold": np.zeros(6)}
            if args.broad:
                broad = population[population_order[rep % len(population)]]
                mean["unselected"] = np.r_[broad[1:], np.zeros(4)]
            rec = dict(train=[], eval={k: [] for k in mean})
            data["replicates"].append(rec)
            for budget in range(0, args.points + 1, CHUNK):
                for name in mean:
                    key = tuple(mean[name])
                    if budget == 0 and key in initial_cache:
                        results = initial_cache[key]
                    else:
                        results = [point(arena, template, mean[name], prefix) for prefix in held]
                        if budget == 0:
                            initial_cache[key] = results
                    rec["eval"][name].append(dict(budget=budget, params=mean[name].tolist(), rows=results))
                    print(
                        rep,
                        budget,
                        name,
                        sum(bool(r["legal"]) for r in results),
                        "/",
                        len(results),
                        flush=True,
                    )
                data["seconds"] = perf_counter() - now
                _write(args.output, data)
                if budget == args.points:
                    break
                batch = {k: [] for k in mean}
                for seed, lane in training[budget : budget + CHUNK]:
                    prefix = observer(arena, seed, lane)
                    noise = policy_rng.normal(size=6) * PARAMETER_SCALE
                    for name in mean:
                        theta = np.clip(mean[name] + noise, -LIMITS, LIMITS)
                        result = point(arena, template, theta, prefix)
                        batch[name].append((theta, min(3, result["legal"])))
                        rec["train"].append(dict(arm=name, params=theta.tolist(), **result))
                for name, probes in batch.items():
                    candidates, score = map(np.asarray, zip(*probes))
                    mean[name] = (2.0 * mean[name] + score @ candidates) / (2.0 + sum(score))
            _write(args.output, data)
    finally:
        arena.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--generic", action="store_true")
    ap.add_argument("--broad", action="store_true")
    ap.add_argument("--held", default=20, type=int)
    ap.add_argument("--points", default=24, type=int)
    ap.add_argument("--replicates", default=3, type=int)
    ap.add_argument("--seed", default=288627221, type=int)
    args = ap.parse_args()
    if args.points % CHUNK:
        raise ValueError("practice length must include complete update batches")
    run(args)
