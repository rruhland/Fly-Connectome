"""Fit a legal, repeatable, pre-point receiving stance against independent varying deliveries."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .tt4_transfer_trial import _write
from .tt25_recorded_flight import OUTPUT
from .tt29_rally_physics import BallFlight, BladeIK, RallyArena
from .tt29_reward_practice import practice_feed


class ReadyStance:
    ready = None

    def _arm_home(self, side):
        if side == 1 and self.ready is not None:
            return np.r_[self.ready, -0.1], ()
        return super()._arm_home(side)


class BlockArena(ReadyStance, RallyArena):
    pass


def trajectory(env, flight, shot):
    p = [shot["pos_x"], shot["pos_y"], 0.76 + shot["pos_z"]]
    v = [shot["vel_" + k] for k in "xyz"]
    w = [shot["w_vel_" + k] for k in "xyz"]
    return flight.path(p, v, w, 1, horizon=0.98)


def block(env, shot, goal):
    env.ready = goal
    env.launch(shot)
    near = np.zeros(env.action_size)
    far = np.zeros(len(env._arm_ids[-1][2]))
    for _ in range(env.max_steps):
        _, _, done, cap, _ = env.exchange(near, far)
        if done or cap or env.legal_returns:
            break
    hit = next(
        (
            e
            for e in env.events
            if e["kind"] == "separation" and e.get("collider") == "learner_racket"
        ),
        None,
    )
    return dict(
        legal=int(env.legal_returns > 0),
        contact=int(env.contacts > 0),
        reason=env.reason,
        outgoing=None if hit is None else hit["velocity"],
    )


def run(output, size, seed):
    env = BlockArena(frameskip=8)
    ik = BladeIK(env)
    try:
        qs = qmc.scale(
            qmc.Sobol(5, scramble=True, seed=29025).random_base2(16), ik.low, ik.high
        )
        poses = [ik.pose(q) for q in qs]
        pos = np.stack([p for p, n in poses])
        norm = np.stack([n for p, n in poses])
        tree = cKDTree(np.c_[pos, 0.22 * norm])
        shot = practice_feed(np.random.default_rng(seed), 1, spread=0)
        track = trajectory(env, BallFlight(env), shot)
        rng = np.random.default_rng(seed)
        records = []
        for i in range(size):
            t = track[rng.integers(len(track))]
            if not (-1.64 < t[2] < -0.77 and t[3] > 0.81):
                continue
            tilt, yaw = rng.uniform(-0.05, 0.59), rng.uniform(-0.17, 0.17)
            n = np.array(
                [np.sin(yaw) * np.cos(tilt), np.cos(yaw) * np.cos(tilt), np.sin(tilt)]
            )
            p = t[1:4] - 0.026 * n + rng.uniform(-0.02, 0.02, 3)
            raw = qs[tree.query(np.r_[p, 0.22 * n], k=4)[1][rng.integers(4)]]
            q, (at, normal, *_) = ik.fit(p, n, raw)
            if np.linalg.norm(at - p) > 0.075 or normal @ n < 0.78:
                continue
            score = block(env, shot, q)
            records.append(
                dict(
                    q=q.tolist(), point=t[1:4].tolist(), normal=normal.tolist(), **score
                )
            )
        candidates = [r for r in records if r["legal"]]
        print("nominal", len(records), len(candidates), flush=True)
        validation = []
        dev = np.random.default_rng(29021)
        deck = [practice_feed(dev, 80000 + i) for i in range(16)]
        # Uniform sampling once, independent of the validation and held decks.
        idx = rng.choice(len(candidates), min(20, len(candidates)), replace=False)
        for j in idx:
            c = candidates[j]
            counts = [block(env, s, np.array(c["q"]))["legal"] for s in deck]
            validation.append((sum(counts), c))
        held = []
        selected = None
        if validation:
            selected = max(validation, key=lambda r: r[0])[1]
            test = np.random.default_rng(29390156)
            for i in range(64):
                s = practice_feed(test, 90000 + i)
                held.append(
                    dict(
                        shot=s,
                        learned=block(env, s, np.array(selected["q"])),
                        stock=block(env, s, None),
                    )
                )
        print(
            "validation",
            max((v[0] for v in validation), default=0),
            "held",
            sum(r["learned"]["legal"] for r in held),
            len(held),
            flush=True,
        )
        _write(
            output,
            dict(
                seed=seed,
                nominal=shot,
                trials=records,
                selected=selected,
                validation=[s for s, c in validation],
                held=held,
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=896)
    p.add_argument("--seed", type=int, default=293614)
    p.add_argument(
        "--output", type=Path, default=OUTPUT.parent / "tt29" / "ready-block.json.gz"
    )
    args = p.parse_args()
    run(args.output, args.size, args.seed)
