"""Practice new lateral feeds using measured motor effects near successful contacts."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _write
from .tt24_rally_trial import PublicBody
from .tt25_pose_demonstrations import OBSERVE, episode
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_optical_transfer import decoder
from .tt27_flight_feedback import OUT, command, load_bank
from .tt27_growing_receiver import bank, strategy
from .tt27_outcome_receiver import independent

OFFSETS = (-0.16, -0.10, -0.055, 0.055, 0.10, 0.16)


def body_motion(arena, shot, rig, goal, ticks):
    arena.launch(shot)
    body = PublicBody(arena)
    for tick in range(ticks):
        # The practice fixture holds the ball clear of the arm for dry motor identification.
        arena.ball[:] = [0, 1.15, 2.1]
        arena.velocity[:] = 0.0
        u = command(goal if tick >= OBSERVE else None, body)
        _, _, done, cap, _ = arena.step(u, render=False)
        body.issued(u)
        if done or cap:
            raise ValueError("Incomplete dry motion")
    position, orientation = rig.links.read(arena.joints[None, :5])
    return np.r_[position[0] + orientation[0] @ rig.offset[3:], orientation[0] @ rig.normal]


def identify(args):
    path, index = args
    lookup, _ = load_match(path)
    row = load_bank()["rows"][index]
    shot = lookup[row["id"]]
    arena = ReferenceReceiver()
    rig = decoder()
    rng = np.random.default_rng(130151 + index)
    goal = np.array(row["goal"])
    try:
        episode(arena, shot, goal=goal)
        contact = next(e for e in arena.events if e["kind"] == "contact" and e["side"] == 1)
        ticks = contact["step"] - 1
        center = body_motion(arena, shot, rig, goal, ticks)
        eps = 0.045
        axes = np.eye(5) * eps
        jac = np.stack(
            [
                (
                    body_motion(arena, shot, rig, goal + e, ticks)
                    - body_motion(arena, shot, rig, goal - e, ticks)
                )
                / (2 * eps)
                for e in axes
            ],
            axis=1,
        )
        weights = np.array([1.0, 1.0, 1.0, 0.16, 0.16, 0.16])
        d = jac * weights[:, None]
        inverse = np.linalg.solve(d.T @ d + 0.0008 * np.eye(5), d.T * weights)
        records = []
        for dx in OFFSETS:
            new = shot.copy()
            new["pos_x"] += dx
            q = goal.copy()
            target = center + np.array([dx, 0.0, 0.0, 0.0, 0.0, 0.0])
            for _ in range(3):
                seen = body_motion(arena, shot, rig, q, ticks)
                q += np.clip(inverse @ (target - seen), -0.4, 0.4)
                q = np.clip(q, goal - 0.8, goal + 0.8)
            attempts = []
            for trial in range(5):
                test = q if trial == 0 else q + rng.normal(0, [0.075, 0.06, 0.060, 0.085, 0.12])
                r, _ = episode(arena, new, goal=test)
                attempts.append(dict(goal=test.tolist(), score=r))
                if r["legal"]:
                    verify, key = episode(arena, new, goal=test, collect=True)
                    if r != verify or key is None:
                        raise ValueError("Dry-motor curriculum example failed physical replay")
                    break
            else:
                key = None
            records.append(dict(dx=dx, key=key.tolist() if key is not None else None, attempts=attempts))
        print("taught", row["id"], sum(r["key"] is not None for r in records), flush=True)
        return dict(
            id=row["id"],
            ticks=ticks,
            contact_reference=center.tolist(),
            jacobian=jac.tolist(),
            rows=records,
        )
    finally:
        arena.close()


def run(path):
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        source = list(pool.map(identify, [(path, n) for n in range(19)]))
    _write(OUT / "lateral-practice.json.gz", dict(source=source, offsets=OFFSETS))
    keys, goals, scales = bank()
    original = len(keys)
    for b in source:
        for r in b["rows"]:
            if r["key"] is not None:
                keys = np.vstack((keys, r["key"]))
                goals = np.vstack((goals, r["attempts"][-1]["goal"]))
    np.savez_compressed(OUT / "lateral-memory.npz", keys=keys, goals=goals, scale=scales)
    shots, _ = load_match(path)
    rng = np.random.default_rng(179031)
    tests = []
    for r in load_bank()["rows"]:
        for sign in (-1, 1):
            for v in (-0.22, 0.22):
                s = shots[r["id"]].copy()
                s["pos_x"] += sign * rng.uniform(0.04, 0.15)
                s["vel_x"] += v
                s["vel_z"] += rng.uniform(-0.09, 0.09)
                tests.append(("new_lateral", s))
    tests += independent(shots)
    arena = ReferenceReceiver()
    rows = []
    try:
        for group, s in tests:
            base, k = episode(arena, s, collect=True)
            scores = {}
            for name in ("original", "expanded", "blend"):
                choose = lambda key: strategy(key, keys, goals, scales, original, name)
                scored = base if k is None else episode(arena, s, goal=choose(k))[0]
                if scored["legal"] and scored != episode(arena, s, choose_goal=choose)[0]:
                    raise ValueError("Held lateral return failed live replay")
                scores[name] = scored
            rows.append(dict(group=group, id=s["id"], results=scores))
    finally:
        arena.close()
    _write(OUT / "lateral-held.json.gz", dict(rows=rows))
    for c in ("nearby", "unrestricted", "new_lateral"):
        print(
            c,
            {k: sum(r["results"][k]["legal"] for r in rows if r["group"] == c) for k in scores},
            flush=True,
        )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
