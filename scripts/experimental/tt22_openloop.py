"""Outcome-only stroke search and a camera-blind control for cooperative rally tests."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt22_rally_trial import RallyArena, opponent


def stroke(parameters, tick):
    # Targets are offsets from the starting setpoints, in 30 Hz action units.
    ready, forward = parameters[:5], parameters[5:10]
    start, duration, period = parameters[10:]
    t = (tick - start) % period
    if tick < start:
        return ready
    rise = min(1.0, t / duration)
    fall = max(0.0, 1 - (t - duration) / 5.0)
    return ready + forward * min(rise, fall)


def rally_count(events):
    sides = [int(e["side"]) for e in events if e["kind"] == "legal_return"]
    if any(a == b for a, b in zip(sides, sides[1:])):
        raise ValueError("consecutive legal landings by the same player")
    return len(sides)


def play(arena, parameters, seed):
    arena.reset(seed=seed, server=-1, render=False)
    target = np.zeros(5)
    for tick in range(arena.max_steps):
        action = np.clip(stroke(parameters, tick) - target, -1, 1)
        target += action
        _, _, done, capped, info = arena.step(np.r_[action, 0.0], render=False)
        if done or capped:
            break
    return dict(
        seed=int(seed),
        legal=rally_count(arena.events),
        player=info["legal_returns"],
        rival=info["rival_returns"],
        reason=info["reason"],
        ticks=arena.steps,
    )


def evaluate(arena, candidate, seeds):
    return [play(arena, candidate, int(seed)) for seed in seeds]


def train(arena, population, generations, seed):
    rng = np.random.default_rng(seed)
    center = np.r_[np.zeros(10), 19.0, 4.0, 47.0]
    width = np.r_[np.full(5, 0.8), np.full(5, 2.2), 2.0, 1.2, 4.0]
    low = np.r_[np.full(5, -4), np.full(5, -8), 9.0, 1.0, 28.0]
    high = np.r_[np.full(5, 4), np.full(5, 8), 25.0, 8.0, 70.0]
    history, previous = [], []
    for generation in range(generations):
        seeds = rng.integers(2**31, size=3)
        candidates = list(np.clip(rng.normal(center, width, size=(population, 13)), low, high))
        candidates[: len(previous)] = previous
        # Equal random starts for every candidate in a generation. Only scores enter the fit.
        scores = [sum(row["legal"] for row in evaluate(arena, p, seeds)) for p in candidates]
        order = np.argsort(-np.asarray(scores), kind="stable")[: max(4, population // 5)]
        elite = np.stack([candidates[i] for i in order])
        if max(scores):
            previous = [elite[0].copy(), elite[1].copy()]
            center = 0.25 * center + 0.75 * elite.mean(axis=0)
            width = np.maximum(0.70 * width + 0.30 * elite.std(axis=0), 0.12)
        history.append(dict(seeds=seeds.tolist(), scores=scores, best=candidates[int(order[0])].tolist()))
        print(
            "generation", generation + 1, "best", max(scores), "mean", round(np.mean(scores), 2), flush=True
        )
    return np.array(history[-1]["best"]), history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--spread", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=220749)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--generations", type=int, default=7)
    ap.add_argument("--opponent", choices=("base", "anchor", "varied"), default="base")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--held", type=int, default=16)
    args = ap.parse_args()
    arena = RallyArena(args.spread, 1.0)
    if args.opponent != "base":
        opponent(arena, args.opponent)
    start = perf_counter()
    try:
        if args.checkpoint:
            params = np.array(json.loads(args.checkpoint.read_text())["parameters"])
            history = []
        else:
            params, history = train(arena, args.population, args.generations, args.seed)
        seeds = np.random.default_rng(88581306).integers(2**31, size=args.held)
        result = dict(
            parameters=params.tolist(),
            training=history,
            held=evaluate(arena, params, seeds),
            stationary=evaluate(arena, np.r_[np.zeros(10), 19.0, 4.0, 47.0], seeds),
            training_seed=args.seed,
            spread=args.spread,
            opponent=args.opponent,
            seconds=perf_counter() - start,
        )
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("held", [x["legal"] for x in result["held"]], flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    main()
