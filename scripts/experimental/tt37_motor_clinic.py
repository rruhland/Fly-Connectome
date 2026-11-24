"""Practice joint corrections using isolated physical counterfactuals, then replay with a frozen memory."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from . import tt33_joint as joint
from .tt4_transfer_trial import _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, Rollout, measured_launches
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt37_belief import ArrivalHistory
from .tt37_opponent import arena
from .tt37_posture import Surface, sample

GRIP = -0.1
FAR = np.zeros(5)
AIM = np.array([0.0, 1.29, 1.05, 3.6], dtype=float)
SCALE = np.array([0.15, 0.1, 0.8, 1.2, 1.7, 0.9, 0.9, 0.9, 0.9, 0.9])


def geometry(surf, opt):
    x, z = opt[1:3]
    vx, _, vz = opt[3:6]
    normal = np.array([-0.18 * x - 0.085 * vx, 1.0, 0.29 - 0.095 * vz])
    normal /= np.linalg.norm(normal)
    qs, v = surf.solution(x, z, normal)
    return qs, v


def generate(n):
    # Position and joint velocity at the action center, discovered by bounded search.
    raw = (
        2
        * qmc.Sobol(10, scramble=True, seed=37091).random_base2(
            int(np.ceil(np.log2(n)))
        )[:n]
        - 1
    )
    return np.r_[
        np.zeros((1, 10)),
        raw * np.r_[[0.23, 0.33, 0.38, 0.36, 0.50], [2.2, 3.0, 2.8, 3.0, 3.6]],
    ]


def schedules(center, motion, eta, params, dt, home, frames):
    t = (1 + np.arange(frames)) * dt
    span = np.clip(t - eta + 0.030, -0.13, 0.065)
    poses = center + params[:, :5]
    velocities = 0.86 * motion + params[:, 5:]
    result = poses[:, None, :] + span[None, :, None] * velocities[:, None, :]
    return np.concatenate((result, np.full((*result.shape[:2], 1), GRIP)), axis=-1)


def recall(bank, x):
    if not bank:
        return np.zeros(10)
    d = np.linalg.norm((np.array([e["x"] for e in bank]) - x) / SCALE, axis=1)
    return np.array(bank[int(np.argmin(d))]["change"])


def consequence_search(history, elite, rng, sizes, count):
    """Use local observed sensitivities of outgoing velocity to propose motor changes."""
    actions = np.asarray([h[0] for h in history]) / sizes
    observations = np.asarray([h[1] for h in history])
    hit = observations[:, 1] > 0
    if hit.sum() < 8:
        return np.empty((0, 10))
    actions = actions[hit]
    observed = observations[hit]
    proposals = []
    for seed in elite[:5]:
        center = seed / sizes
        d = np.linalg.norm(actions - center, axis=1)
        idx = np.argsort(d)[: min(48, len(d))]
        w = np.exp(-0.5 * (d[idx] / max(0.5, np.median(d[idx]))) ** 2)
        w /= w.sum()
        a = actions[idx]
        v = observed[idx, 2:5]
        da = a - w @ a
        dv = v - w @ v
        jac = np.linalg.solve(
            da.T @ (w[:, None] * da) + 0.03 * np.eye(10), da.T @ (w[:, None] * dv)
        ).T
        base = w @ v + jac @ (center - w @ a)
        position = w @ observed[idx, 5:8]
        for vy in (2.9, 3.5, 4.2, 4.9):
            flight = max(0.30, (0.65 - position[1]) / (vy * 0.9))
            target = np.array(
                [
                    -position[0] / flight,
                    vy,
                    (0.78 - position[2]) / flight + 5.0 * flight,
                ]
            )
            change = jac.T @ np.linalg.solve(
                jac @ jac.T + 0.15 * np.eye(3), target - base
            )
            for gain in (0.65, 1.15):
                value = (
                    center + gain * np.clip(change, -1.5, 1.5) + rng.normal(0, 0.04, 10)
                )
                proposals.append(np.clip(value, -2.7, 2.7) * sizes)
    return np.asarray(proposals[:count])


def landing_error(feedback):
    p, v = feedback[:, 5:8], feedback[:, 2:5]
    gy = 9.81
    airtime = np.maximum(
        0.02,
        (v[:, 2] + np.sqrt(np.maximum(0.0, v[:, 2] ** 2 + 2 * gy * (p[:, 2] - 0.78))))
        / gy,
    )
    travel = 0.94 * airtime
    xy = p[:, :2] + v[:, :2] * travel[:, None]
    net_t = -p[:, 1] / np.maximum(0.6, v[:, 1] * 0.94)
    net_h = p[:, 2] + v[:, 2] * net_t - gy * 0.5 * net_t**2
    return (
        np.maximum(0.0, np.abs(xy[:, 0]) - 0.48) * 2
        + np.abs(xy[:, 1] - 0.68) * 0.55
        + np.maximum(0.0, 0.98 - net_h) * 2
        + np.maximum(0.0, 2.0 - v[:, 1]) * 0.6
        + 0.7 * feedback[:, 8]
    )


def search(
    engine, fit, eta, choices, dt, home, rounds, rng, guided=False, experience=None
):
    evaluations = []
    history = []
    sizes = np.array([0.23, 0.33, 0.38, 0.36, 0.50, 2.2, 3.0, 2.8, 3.0, 3.6])
    count = len(choices)
    distance = []
    for round_index in range(max(1, rounds)):
        goals = schedules(*fit, eta, choices, dt, home, round((eta + 1.05) / dt))
        feedback = np.zeros((len(choices), 9))
        _, outcomes = engine.trajectories(goals, AIM, eta + 0.12, feedback=feedback)
        ok = np.flatnonzero(np.isfinite(outcomes[:, 0]))
        if experience is not None:
            value = np.where(np.isfinite(outcomes[:, 0]), outcomes[:, 0], -1.0)
            experience.append(np.c_[choices, feedback, value].astype(np.float32))
        # The labels below stay inside simulator practice. Only the winning
        # joint corrections and their pre-action sensory contexts are exported.
        b, v = feedback[:, 5:8], feedback[:, 2:5]
        # Nominal outgoing velocity to an interior landing, used on faults only.
        flight = np.maximum(0.32, (0.85 - b[:, 1]) / 4.0)
        wanted = np.c_[
            -b[:, 0] / flight,
            np.full(len(v), 4.4),
            (0.78 - b[:, 2]) / flight + 4.95 * flight,
        ]
        error = (
            np.linalg.norm((v - wanted) * [0.8, 0.8, 1.3], axis=1)
            + 2.0 * feedback[:, 8]
        )
        loss = np.where(
            feedback[:, 1] > 0,
            landing_error(feedback) if guided else error,
            15.0 + 12.0 * feedback[:, 0],
        )
        if guided:
            history.extend(zip(choices, feedback, strict=True))
        distance.append(
            {
                "trials": len(choices),
                "contacts": int(feedback[:, 1].sum()),
                "legal": len(ok),
                "closest": float(feedback[:, 0].min()),
                "best": float(loss.min()),
            }
        )
        if len(ok):
            idx = ok[np.argmin(outcomes[ok, 0])]
            return choices[idx], outcomes[idx].tolist(), distance
        if rounds == 0:
            return choices[0], None, distance
        evaluations.extend(zip(loss, choices, strict=True))
        elite = np.asarray([a for _, a in sorted(evaluations, key=lambda x: x[0])[:10]])
        parent = elite[rng.integers(len(elite), size=count)]
        noise = (
            rng.normal(0.0, 1.0, (count, 10))
            * sizes
            * (0.58 if round_index < 2 else 0.35)
        )
        choices = np.clip(parent + noise, -2.2 * sizes, 2.2 * sizes)
        if guided:
            inverse = consequence_search(history, elite, rng, sizes, count // 2)
            choices[: len(inverse)] = inverse

    return min(evaluations, key=lambda x: x[0])[1], None, distance


def point(
    a,
    stereo,
    rig,
    model,
    surf,
    stride,
    lo,
    hi,
    card,
    shot,
    bank,
    params,
    mode,
    cap,
    cut,
    rounds,
    guided,
    archive,
    response,
    refine,
):
    actor = joint.JointPolicy(model | {"ready": surf.ready.tolist()}, stride, lo, hi)
    eye = joint.Vision(rig, a.control_dt)
    a.launch(shot, card=card)
    a.partner.reset(seed=card["seed"])
    engine = Rollout(a, 1) if mode == "coach" else None
    records = []
    actions = []
    tape = None
    deadline = -100
    tape_start = retry_tick = -1
    lastprep = -100
    prep = surf.ready
    history = ArrivalHistory()
    if response is not None:
        response.last = None
    for tick in range(a.max_steps):
        q = a.joints.copy()
        obs = eye.update(tick, stereo.capture() if tick % 2 == 0 else None, q)
        now = tick * a.control_dt
        if refine:
            obs = history.update(now, obs)
        if (
            refine
            and engine is not None
            and tape is not None
            and retry_tick == tick
            and obs is not None
            and 0.035 < obs[0] < 0.22
            and 0.76 < obs[2] < 1.38
        ):
            tape = None
            deadline = -100
        if tape is not None and tick - tape_start < len(tape):
            desired = tape[tick - tape_start]
        else:
            tape = None
            desired = surf.ready
            if obs is not None and 0.03 < obs[0] < 0.80 and now > deadline:
                if obs[0] <= cut and (not guided or 0.76 < obs[2] < 1.38):
                    eta = obs[0]
                    fit = geometry(surf, obs)
                    x = np.r_[obs[1:6], q[:5]]
                    if response is not None and response.last is not None:
                        response.observe(False)
                    remembered = (
                        recall(bank, x) if response is None else response(bank, x)
                    )
                    choices = (
                        np.array([remembered])
                        if engine is None
                        else np.vstack((remembered, params))
                    )
                    selected = remembered
                    trace = []
                    score = None
                    if engine is not None:
                        experience = []
                        selected, score, trace = search(
                            engine,
                            fit,
                            eta,
                            choices,
                            a.control_dt,
                            surf.ready,
                            rounds,
                            np.random.default_rng([37173, len(bank), tick]),
                            guided,
                            experience,
                        )
                        if archive is not None:
                            table = np.concatenate(experience)
                            archive.append(
                                np.c_[
                                    np.repeat(x[None], len(table), axis=0), table
                                ].astype(np.float32)
                            )
                        if score is not None:
                            bank.append({"x": x.tolist(), "change": selected.tolist()})
                    tape = schedules(
                        *fit,
                        eta,
                        np.array([selected]),
                        a.control_dt,
                        surf.ready,
                        max(1, round((eta + 0.075) / a.control_dt)),
                    )[0]
                    desired = tape[0]
                    tape_start = tick
                    deadline = now + eta + 0.26
                    retry_tick = (
                        tick + max(3, round((eta - 0.092) / a.control_dt))
                        if score is None and eta > 0.13
                        else -1
                    )
                    records.append(
                        {
                            "tick": tick,
                            "context": x.tolist(),
                            "choices": sum(t["trials"] for t in trace),
                            "good": sum(t["legal"] for t in trace),
                            "score": score,
                            "search": trace,
                            "action": selected.tolist(),
                        }
                    )
                elif not guided or 0.76 < obs[2] < 1.38:
                    if tick - lastprep >= 4:
                        p, v = geometry(surf, obs)
                        prep = np.r_[p - 0.075 * v, GRIP]
                        lastprep = tick
                    desired = prep
        action = actor.drive(desired, q, 0.8)
        _, _, done, capped, info = a.exchange(action, FAR)
        if response is not None and response.last is not None:
            success = any(
                e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]
            )
            if success or done or capped:
                response.observe(success)
        actions.append(action.tolist())
        if done or capped or a.legal_returns + a.rival_returns - 1 >= cap:
            break
    return {
        "actions": actions,
        "records": records,
        "events": a.events,
        "placements": a.point_placements(),
        "score": audit(a.events, opening_side=1),
        "shot": shot,
        "opponent_seed": card["seed"],
    }


def run(args):
    plan = _read(args.plan)
    cards = plan[args.group][args.start : args.start + args.n]
    shots = measured_launches(args.feeds)
    model = _read(PUPIL)
    a = arena()
    ik = BladeIK(a)
    surface = Surface(ik, sample(ik), args.depth, args.surface)
    a.close()
    joint.RECEIVE_Y = args.depth
    bank = _read(args.memory)["bank"] if args.memory else []
    out = {
        "cards": cards,
        "ready": surface.ready.tolist(),
        "depth": args.depth,
        "cut": args.cut,
        "mode": args.mode,
        "rows": [],
        "bank": bank,
    }
    out.update(
        refine=args.refine,
        rounds=args.rounds,
        samples=args.samples,
        guided=args.guided,
        group=args.group,
    )
    response = None
    if args.response:
        from .tt37_response import Response

        response = Response(args.response, adaptive=args.online)
        out["pretraining"] = [str(p) for p in args.response]
    if args.forest:
        from .tt37_value import Forest
        response = Forest(args.forest, surface, adaptive=args.online)
        out["forest"] = str(args.forest)
    archive = [] if args.archive else None
    changes = generate(args.samples)
    with centered(surface.ready):
        a = arena()
        stereo, rig, stride, lo, hi = joint.resources(a)
        try:
            for c in cards:
                r = point(
                    a,
                    stereo,
                    rig,
                    model,
                    surface,
                    stride,
                    lo,
                    hi,
                    c,
                    shots[c["id"]],
                    bank,
                    changes,
                    args.mode,
                    args.cap,
                    args.cut,
                    args.rounds,
                    args.guided,
                    archive,
                    response,
                    args.refine,
                )
                if response is not None:
                    out["online_outcomes"] = [
                        {"context": x.tolist(), "action": u.tolist(), "reward": v}
                        for x, u, v in response.experience
                    ]
                out["rows"].append(r)
                _write(args.out, out)
                if archive:
                    tmp = args.archive.with_suffix(".partial.npz")
                    np.savez_compressed(tmp, rows=np.concatenate(archive))
                    tmp.replace(args.archive)
                print(
                    c["id"],
                    r["score"]["learner"],
                    [(x["good"], x["choices"]) for x in r["records"]],
                    flush=True,
                )
        finally:
            stereo.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("coach", "recall"))
    p.add_argument("out", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument(
        "--plan", type=Path, default=Path("artifacts/tt37/opponent-plan.json")
    )
    p.add_argument("--group", default="practice")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--samples", type=int, default=80)
    p.add_argument("--memory", type=Path)
    p.add_argument("--cap", type=int, default=24)
    p.add_argument("--depth", type=float, default=-1.13)
    p.add_argument("--surface", type=float, default=0.35)
    p.add_argument("--cut", type=float, default=0.185)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--guided", action="store_true")
    p.add_argument("--archive", type=Path)
    p.add_argument("--response", type=Path, nargs="+")
    p.add_argument("--online", action="store_true")
    p.add_argument("--refine", action="store_true")
    p.add_argument("--forest", type=Path)
    run(p.parse_args())
