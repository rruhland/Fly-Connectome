"""Calibrate a continuous joint posture surface and test physical reaching across both sides."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from .tt4_transfer_trial import _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, measured_launches
from .tt31_ready import centered
from .tt33_joint import JointPolicy, Vision, resources
from .tt34_latency import PUPIL
from .tt37_opponent import arena as live_arena

OUT = Path(__file__).resolve().parents[2] / "artifacts/tt37"


class Surface:
    """Select a connected series of arm solutions across lateral task space."""

    def __init__(self, ik, source, depth, tilt=0.37):
        self.ik = ik
        self.depth = depth
        self.tilt = tilt
        self.xs = np.linspace(-0.34, 0.34, 25)
        raw, poses = source
        self.index = cKDTree(poses * [1, 1, 1, 0.14, 0.14, 0.14])
        paths, parents = [], []
        loss = np.empty(0)
        for x in self.xs:
            p = np.array([x, depth, 1.02])
            n = np.array([-0.18 * x, 1.0, tilt])
            n /= np.linalg.norm(n)
            _, j = self.index.query(np.r_[p, 0.14 * n], 150)
            # Solve from different joint branches and retain good matches on each.
            seeds = []
            for idx in j:
                if (
                    not seeds
                    or np.min(np.max(abs(np.asarray(seeds) - raw[idx]), axis=1)) > 0.75
                ):
                    seeds.append(raw[idx])
                if len(seeds) >= 10:
                    break
            choices = []
            costs = []
            for seed in seeds:
                q, (at, face, *_) = ik.fit(p, n, seed)
                if (
                    choices
                    and np.min(np.max(abs(np.asarray(choices) - q), axis=1)) < 0.07
                ):
                    continue
                choices.append(q)
                costs.append(np.linalg.norm(at - p) + 0.23 * np.linalg.norm(face - n))
            choices = np.asarray(choices)
            cost = np.asarray(costs)
            if paths:
                travel = cdist(paths[-1], choices, metric="chebyshev")
                v = (
                    loss[:, None]
                    + 0.11 * travel
                    + 0.12 * np.square(np.maximum(0.0, travel - 0.40))
                )
                parents.append(v.argmin(axis=0))
                loss = cost + v.min(axis=0)
            else:
                loss = cost
            paths.append(choices)
        j = loss.argmin()
        path = [paths[-1][j]]
        for i in range(len(paths) - 2, -1, -1):
            j = parents[i][j]
            path.append(paths[i][j])
        self.q = np.array(path[::-1])
        self.ready = np.r_[self.q[len(self.xs) // 2], -0.10]

    def solution(self, x, z, normal):
        p = np.array([x, self.depth, z]) - 0.025 * normal
        seed = np.array([np.interp(x, self.xs, q) for q in self.q.T])
        q, (_at, _face, j, jrot) = self.ik.fit(p, normal, seed)
        direction = np.linalg.lstsq(
            np.r_[j, 0.07 * jrot, 0.016 * np.eye(5)],
            np.r_[normal, np.zeros(8)],
            rcond=None,
        )[0]
        return q, direction


def sample(ik):
    q = qmc.scale(
        qmc.Sobol(5, scramble=True, seed=3701).random_base2(16), ik.low, ik.high
    )
    return q, np.asarray([np.r_[*ik.pose(x)] for x in q])


def play(a, eye, stereo, model, stride, lo, hi, surface, card, shot, settings, exact):
    a.launch(shot, card=card)
    a.partner.reset(seed=card["seed"])
    actor = JointPolicy(model | {"ready": surface.ready.tolist()}, stride, lo, hi)
    ready = surface.ready.copy()
    qstar = ready[:5].copy()
    direction = np.zeros(5)
    target = None
    eta = None
    commands = []
    for i in range(a.max_steps):
        q = a.joints.copy()
        time = i * a.control_dt
        optical = (
            eye.update(i, stereo.capture() if i % 2 == 0 else None, q)
            if not exact
            else None
        )
        # Oracle used only for debugging the ideal-sensing ceiling.
        if exact and a._last_hitter == -1 and a.velocity[1] < -0.9 and i % 4 == 0:
            track = a.partner.flight.path(
                a.ball,
                a.velocity,
                a.spin,
                1,
                bounced=a._received_bounce,
                horizon=0.95,
            )
            good = np.where(track[:, 2] < surface.depth)[0] if len(track) else []
            if len(good) and good[0] > 0:
                b = good[0]
                t = (surface.depth - track[b - 1, 2]) / (track[b, 2] - track[b - 1, 2])
                cut = track[b - 1] + t * (track[b] - track[b - 1])
                optical = np.r_[cut[0], cut[1], cut[3], cut[4:7]]
        desired = ready.copy()
        if (
            optical is not None
            and -0.08 < optical[0] < 0.85
            and (i % 3 == 0 or eta is None or abs(time + optical[0] - eta) > 0.065)
        ):
            x, z = optical[1:3]
            vx, _vy, vz = optical[3:6]
            if target is None or optical[0] > 0.10:
                normal = np.array(
                    [-0.22 * x - 0.09 * vx, 1.0, settings["tilt"] - 0.105 * vz]
                )
                normal /= np.linalg.norm(normal)
                qstar, direction = surface.solution(x, z, normal)
                target = [x, z]
                eta = time + optical[0]
        if eta is not None and -0.10 < eta - time:
            dt = time - eta + settings["lead"]
            delta = settings["speed"] * np.clip(dt, -0.115, 0.075)
            desired[:5] = qstar + np.clip(delta * direction, -0.85, 0.85)
        elif eta is not None:
            target = None
            eta = None
        command = actor.drive(desired, q, 0.8)
        _, _, done, cap, _ = a.exchange(command, np.zeros(5))
        commands.append(command.tolist())
        if done or cap or a.legal_returns + a.rival_returns - 1 >= settings["cap"]:
            break
    return {
        "actions": commands,
        "events": a.events,
        "placements": a.point_placements(),
        "score": audit(a.events, opening_side=1),
        "shot": shot,
        "opponent_seed": card["seed"],
    }


def run(args):
    from . import tt33_joint as raw

    raw.RECEIVE_Y = args.depth
    cards = _read(args.plan)
    cards = cards[args.group] if args.group else cards["cards"]
    a = live_arena()
    ik = BladeIK(a)
    surface = Surface(ik, sample(ik), args.depth, args.tilt)
    a.close()
    model = _read(PUPIL)
    shots = measured_launches(args.feeds)
    result = {
        "ready": surface.ready.tolist(),
        "cards": cards[args.start : args.start + args.n],
        "oracle": args.exact,
        "depth": args.depth,
        "speed": args.speed,
        "lead": args.lead,
        "tilt": args.tilt,
        "rows": [],
    }
    with centered(surface.ready):
        a = live_arena()
        stereo, rig, stride, lo, hi = resources(a)
        try:
            for c in result["cards"]:
                r = play(
                    a,
                    Vision(rig, a.control_dt),
                    stereo,
                    model,
                    stride,
                    lo,
                    hi,
                    surface,
                    c,
                    shots[c["id"]],
                    vars(args),
                    args.exact,
                )
                result["rows"].append(r)
                print(
                    c["id"],
                    r["score"]["learner"],
                    len(r["placements"]["delivered"]),
                    flush=True,
                )
                _write(args.output, result)
        finally:
            stereo.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--group")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--depth", type=float, default=-1.13)
    p.add_argument("--tilt", type=float, default=0.35)
    p.add_argument("--speed", type=float, default=0.70)
    p.add_argument("--lead", type=float, default=0.035)
    p.add_argument("--cap", type=int, default=24)
    p.add_argument("--exact", action="store_true")
    run(p.parse_args())
