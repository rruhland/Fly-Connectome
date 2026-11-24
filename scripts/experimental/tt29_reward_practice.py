"""Reward-only online search for a receiving posture on the new edge-mounted body."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _write
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def practice_feed(rng, index, spread=1.0):
    """A cooperative partner's centered opening with independent lateral, pace, lift and spin variation."""
    j = rng.uniform(-1, 1, 9) * spread
    values = [
        0.045 * j[0],
        1.17 + 0.06 * j[1],
        0.42 + 0.025 * j[2],
        0.10 * j[3],
        -5.4 + 0.38 * j[4],
        0.38 + 0.17 * j[5],
        26 + 12 * j[6],
        4 * j[7],
        9 * j[8],
    ]
    keys = (
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "w_vel_x",
        "w_vel_y",
        "w_vel_z",
    )
    return dict(id=int(index), **dict(zip(keys, values, strict=True)))


def play(arena, shot, goal, follow=None):
    arena.launch(shot)
    axes = arena._arm_ids[1][2]
    increment = (
        arena._joint_increments[axes] * arena.control_dt / arena.model.opt.timestep
    )
    target = arena.joints.copy()
    end = None
    crossing = None
    previous = arena.ball.copy()
    for t in range(arena.max_steps):
        u = np.zeros(6)
        if goal is not None:
            current = (
                goal
                if follow is None or t * arena.control_dt < follow[1]
                else goal + follow[0]
            )
            u[:5] = np.clip((current - target[:5]) / increment[:5], -1, 1)
        target += increment * u
        _, _, done, cap, _ = arena.step(u, render=False)
        if arena.contacts:
            here = arena.ball.copy()
            if previous[2] > 0.78 >= here[2]:
                f = (0.78 - previous[2]) / (here[2] - previous[2])
                end = (previous + f * (here - previous))[:2].tolist()
            if previous[1] < 0 <= here[1]:
                f = -previous[1] / (here[1] - previous[1])
                crossing = float(previous[2] + f * (here[2] - previous[2]))
        previous = arena.ball.copy()
        if arena.legal_returns or done or cap:
            break
    return dict(
        contact=int(arena.contacts > 0),
        legal=int(arena.legal_returns > 0),
        incoming=int(arena.rival_returns > 0),
        reason=arena.reason,
        descending_xy=end,
        net_z=crossing,
    )


def run(seed, games, out):
    start = perf_counter()
    arena = ReferenceReceiver()
    axes = arena._arm_ids[1][2][:5]
    low, high = arena._joint_low[axes], arena._joint_high[axes]
    home = arena.joints[:5].copy()
    rng = np.random.default_rng(seed)
    # The learner only receives joint limits plus binary racket contact and legal return feedback.
    # Separate RNGs keep the environment sequence independent of learner actions.
    env_rng = np.random.default_rng(seed + 1)
    rows, memory = [], []
    seen = 0
    best = home
    try:
        for i in range(games):
            if memory and rng.random() < 0.77:
                m = memory[rng.integers(min(len(memory), 12))]
                scale = 0.08 if m[0] >= 2 else 0.24
                goal = np.clip(m[1] + rng.normal(size=5) * scale, low, high)
            else:
                goal = np.clip(
                    home + rng.normal(size=5) * [0.48, 0.63, 0.77, 0.75, 0.79],
                    low,
                    high,
                )
            shot = practice_feed(env_rng, i)
            score = play(arena, shot, goal)
            credit = 2 * score["legal"] + score["contact"]
            if credit:
                # Revisit successful controls on an independently sampled subsequent delivery.
                repeat = play(arena, practice_feed(env_rng, games + i), goal)
                credit += 2 * repeat["legal"] + repeat["contact"]
                memory.append((credit, goal))
                memory.sort(key=lambda v: -v[0])
                memory = memory[:32]
                seen += 1
            rows.append(dict(**score, goal=goal.tolist(), replay_credit=credit))
            if (i + 1) % 32 == 0:
                print(i + 1, seen, sum(r["legal"] for r in rows), flush=True)
        best = memory[0][1] if memory else home
        held = np.random.default_rng(29180311)
        cases = []
        for i in range(32):
            shot = practice_feed(held, 100000 + i)
            cases.append(
                dict(
                    shot=shot,
                    stationary=play(arena, shot, None),
                    learned=play(arena, shot, best),
                )
            )
    finally:
        arena.close()
    record = dict(
        seed=seed,
        trained_goal=best.tolist(),
        training_draws=games,
        extra_training_draws=seen,
        wall_seconds=perf_counter() - start,
        training=rows,
        held=cases,
    )
    out.parent.mkdir(exist_ok=True)
    _write(out, record)
    print(
        {k: sum(row[k]["legal"] for row in cases) for k in ("stationary", "learned")},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=291023)
    p.add_argument("--games", type=int, default=256)
    p.add_argument(
        "--out", type=Path, default=OUTPUT.parent / "tt29" / "reward-practice.json.gz"
    )
    a = p.parse_args()
    run(a.seed, a.games, a.out)
