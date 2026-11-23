"""Use stereo flight after a return to adapt the next strike on a regulation table."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_arena import (
    AIR_DRAG,
    BALL_RADIUS,
    GRAVITY,
    NET_HEIGHT,
    TABLE_HALF_LENGTH,
    TABLE_HALF_WIDTH,
)
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_rally_trial import PublicBody
from .tt25_pose_demonstrations import OBSERVE, episode
from .tt26_grip_course import Grip, ReferenceReceiver, load_match
from .tt27_flight_feedback import OUT as PRIOR
from .tt27_flight_feedback import Eyes, command, load_bank

OUT = PRIOR.parent / "tt28"
COVERAGE = PRIOR / "action-coverage.json.gz"
POSES = PRIOR / "lateral-memory.npz"
# All adaptive branches receive exactly these same physical practice settings.
PROBES = (0.0, -0.15, 0.15, -0.29, 0.29)
GRID = np.linspace(-0.36, 0.36, 25)
MOUNTS = (-0.18, 0.0, 0.18)
SEEDS = (953751, 526013)
HELD = 8
MODES = ("frozen", "reward", "outgoing")


def outbound(points):
    """Take the first short, increasing down-table RGB track after a reversal."""
    if len(points) < 4:
        return None
    near = min(p[1] for _, p in points)
    if near > -0.90:
        return None
    (_, first), (_, middle), (_, last) = points[-3:]
    if first[1] < -1.78 or first[1] > near + 0.42:
        return None
    if middle[1] - first[1] < 0.028 or last[1] - middle[1] < 0.028 or last[1] > -0.13:
        return None
    ticks = np.array([tick for tick, _ in points[-3:]])
    if ticks[-1] - ticks[0] > 4:
        return None
    return points[-3:]


def forecast(segment, dt):
    """Project an airborne segment to the table and net with approximate drag."""
    ticks, positions = zip(*segment, strict=True)
    t = (np.asarray(ticks) - ticks[-1]) * dt
    p = np.asarray(positions)
    straight = p.copy()
    straight[:, 2] += 0.5 * GRAVITY * t**2
    v, x = np.linalg.lstsq(np.column_stack((t, np.ones_like(t))), straight, rcond=None)[0]
    if np.linalg.norm(straight - (x + t[:, None] * v), axis=1).max() > 0.045 or v[1] < 0.85:
        return None
    net_z = None
    step = 1 / 120
    for _ in range(160):
        v += step * (-AIR_DRAG * min(15, np.linalg.norm(v)) * v - np.array([0, 0, GRAVITY]))
        y = x + step * v
        if x[1] < 0 < y[1]:
            net_z = float(x[2] + (y[2] - x[2]) * -x[1] / (y[1] - x[1]))
        if y[2] <= BALL_RADIUS:
            f = np.clip((x[2] - BALL_RADIUS) / (x[2] - y[2]), 0, 1)
            land = x + f * (y - x)
            # A ball already falling before it reaches the net has zero clearance.
            return [float(land[0]), float(land[1]), 0.0 if net_z is None else net_z]
        x = y
    return None


def play(arena, shot, keys, goals, scale, bias, *, observe_flight=True, tape=None):
    """The controller receives images, encoders and its own past commands."""
    arena.launch(shot)
    eye, body = Eyes(arena), PublicBody(arena)
    trace, actions = [], []
    idx, goal, prediction, segment = None, None, None, None
    for tick in range(arena.max_steps):
        if tape is None:
            if tick <= OBSERVE or (observe_flight and prediction is None and tick < 52):
                key, position = eye.observe(arena, body)
                if position is not None:
                    trace.append((tick, position))
                    recent = outbound(trace)
                    if recent is not None:
                        prediction = forecast(recent, arena.control_dt)
                        if prediction is not None:
                            segment = [[t, *p.tolist()] for t, p in recent]
            if tick == OBSERVE:
                idx = int(np.argmin(np.linalg.norm((keys - key) / scale, axis=1)))
                goal = np.asarray(goals[idx]).copy()
                goal[4] += bias
            u = command(goal, body)
        else:
            u = tape[tick]
        _, _, done, cap, _ = arena.step(u, render=False)
        body.issued(u)
        eye.view.issued(u)
        actions.append(np.asarray(u).tolist())
        if done or cap or arena.legal_returns:
            break
    hit = next((e for e in arena.events if e["kind"] == "contact" and e["side"] == 1), None)
    landing = next(
        (e for e in arena.events if e["kind"] == "bounce" and hit and e["time"] > hit["time"]), None
    )
    return {
        "legal": arena.legal_returns,
        "contact": arena.contacts,
        "hit_tick": hit["step"] if hit else None,
        "landing": landing["location"] if landing else None,
        "reason": arena.reason,
        "events": digest_json(arena.events),
        "idx": idx,
        "goal": goal.tolist() if goal is not None else None,
        "bias": bias,
        "forecast": prediction,
        "segment": segment,
        "commands": actions,
    }


def curve_cost(flight):
    x, y, z = np.asarray(flight).T
    # Aim broadly inside the far half, with margin for stereo and spin error.
    return (
        1.4 * np.maximum(0, abs(x) - (TABLE_HALF_WIDTH - 0.13))
        + np.maximum(0, 0.34 - y)
        + np.maximum(0, y - (TABLE_HALF_LENGTH - 0.19))
        + 2.4 * np.maximum(0, NET_HEIGHT + BALL_RADIUS + 0.025 - z)
        + 0.12 * abs(x)
        + 0.10 * abs(y - 0.76)
    )


def learn_reward(practice):
    """Choose a bias with an actual successful practice shot, starting at zero."""
    successful = [r["bias"] for r in practice if r["legal"]]
    return min(successful, key=abs) if successful else 0.0


def learn_outgoing(practice):
    """Use local interpolation to find an early outbound flight into the target."""
    reward_bias = learn_reward(practice)
    seen = [r for r in practice if r["forecast"] is not None]
    if len(seen) < 2:
        return reward_bias
    offsets = np.array([r["bias"] for r in seen])
    targets = np.array([r["forecast"] for r in seen])
    grid = np.unique(np.r_[GRID, offsets, reward_bias])
    distance = abs(grid[:, None] - offsets)
    allowed = (grid >= min(offsets) - 0.04) & (grid <= max(offsets) + 0.04)
    allowed &= distance.min(axis=1) <= 0.105
    successful = np.array([r["bias"] for r in practice if r["legal"]])
    if len(successful):
        # Preserve local support when a practice ball already landed legally.
        allowed &= np.min(abs(grid[:, None] - successful), axis=1) <= 0.125
    grid = grid[allowed]
    if not len(grid):
        return reward_bias
    forecasts = []
    for bias in grid:
        delta = offsets - bias
        design = np.column_stack((np.ones(len(seen)), delta))
        weight = np.exp(-0.5 * (delta / 0.14) ** 2)
        coefficients = np.linalg.solve(
            design.T @ (weight[:, None] * design) + np.diag((1e-8, 0.0025)),
            design.T @ (weight[:, None] * targets),
        )
        forecasts.append(coefficients[0])
    costs = curve_cost(forecasts)
    costs += 0.16 * abs(grid) + 0.18 * np.min(abs(grid[:, None] - offsets), axis=1)
    return float(grid[np.argmin(costs)])


def variant(shot, rng, width):
    perturbed = shot.copy()
    for prefix, sigma in (("pos", 0.005), ("vel", 0.024), ("w_vel", 1.0)):
        variation = rng.normal(0, width * sigma, 3)
        for axis, delta in zip("xyz", variation, strict=True):
            perturbed[f"{prefix}_{axis}"] += delta
    return perturbed


def contact_course(args):
    path, index, seed, angle, cohort = args
    recorded, _ = load_match(path)
    row = _read(COVERAGE)["rows"][index]
    shot = recorded[row["id"]]
    rng = np.random.default_rng(seed + row["id"])
    arena = ReferenceReceiver()
    with np.load(POSES) as archive:
        keys, goals, scale = (archive[n] for n in ("keys", "goals", "scale"))
    try:
        _, signature = episode(arena, shot, collect=True)
        nearby = np.linalg.norm((keys - signature) / scale, axis=1)
        # Use only the historical contact bits to support each donor stream.
        # Neither selection here nor adaptation reads the archived landing rewards.
        eligible = np.array([i for i, r in enumerate(row["scores"]) if r["contact"]])
        anchor = int(eligible[np.argmin(nearby[eligible])])
        keys, goals = keys[[anchor]], goals[[anchor]]
        Grip(arena.model).set(angle)
        practice = []
        for bias in PROBES:
            feed = variant(shot, rng, 0.70)
            result = play(arena, feed, keys, goals, scale, bias)
            practice.append(dict(feed=feed, **result))
        observations = [
            {name: attempt[name] for name in ("bias", "legal", "forecast")} for attempt in practice
        ]
        policies = {
            str(n): {
                "frozen": 0.0,
                "reward": learn_reward(observations[:n]),
                "outgoing": learn_outgoing(observations[:n]),
            }
            for n in (3, len(PROBES))
        }
        held = []
        for tick in range(HELD):
            feed = variant(shot, rng, 1.0 + 0.35 * (tick % 2))
            unique = sorted({bias for p in policies.values() for bias in p.values()})
            scores = {}
            for bias in unique:
                result = play(arena, feed, keys, goals, scale, bias, observe_flight=False)
                replay = play(arena, feed, keys, goals, scale, bias, tape=result["commands"])
                if replay["events"] != result["events"]:
                    raise ValueError("Held command tape changed its referee events")
                scores[str(bias)] = result
            held.append({"feed": feed, "scores": scores})
        return {
            "donor": row["id"],
            "group": row["group"],
            "seed": seed,
            "mount": angle,
            "cohort": cohort,
            "anchor": anchor,
            "historical_contact_goals": len(eligible),
            "practice": practice,
            "policies": policies,
            "held": held,
        }
    finally:
        arena.close()


def summarize(rows):
    return {
        str(n): {
            mode: {
                metric: sum(
                    x["scores"][str(r["policies"][str(n)][mode])][metric] for r in rows for x in r["held"]
                )
                for metric in ("legal", "contact")
            }
            for mode in MODES
        }
        for n in (3, len(PROBES))
    }


def run(path, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    source = _read(COVERAGE)["rows"]
    indices = [i for i, row in enumerate(source) if any(r["contact"] for r in row["scores"])]
    tasks = [
        (path, i, seed, MOUNTS[(j + k) % len(MOUNTS)], k)
        for k, seed in enumerate(SEEDS)
        for j, i in enumerate(indices)
    ]
    inputs = {"feeds": _hash(path), "coverage": _hash(COVERAGE), "poses": _hash(POSES)}
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(contact_course, task) for task in tasks]
        for job in as_completed(jobs):
            row = job.result()
            records.append(row)
            complete = sorted(records, key=lambda r: (r["cohort"], r["donor"]))
            print(
                len(records),
                "/",
                len(tasks),
                row["donor"],
                row["mount"],
                sum(r["forecast"] is not None for r in row["practice"]),
                str(summarize([row])),
                flush=True,
            )
            _write(
                OUT / "outgoing-held.json.gz",
                {
                    "inputs": inputs,
                    "probes": PROBES,
                    "seeds": SEEDS,
                    "expected": len(tasks),
                    "roster": len(source),
                    "held_per_stream": HELD,
                    "rows": complete,
                    "totals": summarize(complete),
                },
            )


def demo(args):
    path, index = args
    lookup, _ = load_match(path)
    row = load_bank()["rows"][index]
    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    try:
        records = []
        for angle in MOUNTS:
            grip.set(angle)
            for probe in PROBES:
                result = play(
                    arena,
                    lookup[row["id"]],
                    np.asarray([row["key"]]),
                    [row["goal"]],
                    np.ones(14),
                    probe,
                )
                records.append(dict(angle=angle, **result))
        print(
            row["id"],
            [
                [
                    (
                        t["contact"],
                        t["legal"],
                        None if t["forecast"] is None else np.round(t["forecast"], 2).tolist(),
                    )
                    for t in records
                    if t["angle"] == angle
                ]
                for angle in MOUNTS
            ],
            flush=True,
        )
        return {"id": row["id"], "records": records}
    finally:
        arena.close()


def run_demo(path, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(demo, [(path, i) for i in (0, 1, 2, 3, 6, 12, 16)]))
    _write(OUT / "outgoing-development.json.gz", {"rows": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if args.development:
        run_demo(args.feeds, args.workers)
    else:
        run(args.feeds, args.workers)
