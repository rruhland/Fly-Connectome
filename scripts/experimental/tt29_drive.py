"""Use legal outcomes to select physically driven strokes near calibrated ready postures."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_recorded_flight import OUTPUT
from .tt29_rally_physics import BallFlight, BladeIK
from .tt29_ready_block import BlockArena, trajectory
from .tt29_reward_practice import practice_feed

OUT = OUTPUT.parent / "tt29"


def play(a, s, motor, *, events=False):
    # A real point begins with the arm in its selected ready stance.
    a.ready = np.asarray(motor["ready"])
    a.launch(s)
    axes = a._arm_ids[1][2]
    inc = a._joint_increments[axes][:5] * a.control_dt / a.model.opt.timestep
    qtarget = a.joints[:5].copy()
    u = np.zeros(6)
    for k in range(a.max_steps):
        t = (k + 1) * a.control_dt
        target = a.ready + np.clip(t - motor["go"], 0.0, motor["span"]) * np.asarray(
            motor["rate"]
        )
        u[:5] = np.clip((target - qtarget) / inc, -1.0, 1.0)
        qtarget += inc * u[:5]
        _, _, done, cap, _ = a.exchange(u, np.zeros(5))
        if done or cap or a.legal_returns:
            break
    hit = next(
        (
            e
            for e in a.events
            if e["kind"] == "separation" and e.get("collider") == "learner_racket"
        ),
        None,
    )
    out = dict(
        legal=int(a.legal_returns > 0),
        contact=int(a.contacts > 0),
        reason=a.reason,
        outgoing=None if hit is None else hit["velocity"],
    )
    if events:
        out["events"] = a.events
    return out


def designs(a, examples, shot, seed, count):
    rng = np.random.default_rng(seed)
    ik = BladeIK(a)
    track = trajectory(a, BallFlight(a), shot)
    for _ in range(count):
        ex = examples[rng.integers(len(examples))]
        q = np.array(ex["q"])
        p, n, linear, angular = ik.pose(q, jacobian=True)
        ti = track[np.argmin(np.linalg.norm(track[:, 1:4] - p, axis=1)), 0]
        v = np.r_[
            rng.uniform(-0.09, 0.09), rng.uniform(0.7, 1.9), rng.uniform(-0.20, 0.65)
        ]
        jac = np.vstack((linear, 0.075 * angular, 0.035 * np.eye(5)))
        qdot = np.linalg.lstsq(jac, np.r_[v, np.zeros(8)], rcond=None)[0]
        qdot *= min(1.0, 5.2 / max(abs(qdot)))
        lead = rng.uniform(0.045, 0.135)
        offset = rng.uniform(-0.04, 0.04)
        q0 = np.clip(q - qdot * lead, ik.low, ik.high)
        yield dict(
            ready=q0.tolist(),
            rate=qdot.tolist(),
            go=ti - lead + offset,
            span=lead + 0.10,
        )


def run(size, output):
    a = BlockArena(frameskip=4)
    try:
        base = _read(OUT / "ready-block.json.gz")
        shot = base["nominal"]
        examples = [
            r
            for r in base["trials"]
            if r["outgoing"] is not None and r["outgoing"][1] > 1.35
        ]
        scored = []
        for j, m in enumerate(designs(a, examples, shot, 922114, size)):
            r = play(a, shot, m)
            scored.append(dict(motor=m, **r))
        wins = [r for r in scored if r["legal"]]
        print("nominal", len(scored), len(wins), flush=True)
        validation = []
        vrng = np.random.default_rng(290855)
        deck = [practice_feed(vrng, 90000 + i) for i in range(20)]
        for i in np.random.default_rng(290856).permutation(len(wins))[:24]:
            c = wins[i]
            results = [play(a, s, c["motor"])["legal"] for s in deck]
            validation.append((sum(results), c))
        held = []
        selected = None
        if validation:
            selected = max(validation, key=lambda r: r[0])[1]["motor"]
            rng = np.random.default_rng(290857)
            for i in range(64):
                s = practice_feed(rng, 91000 + i)
                held.append(dict(shot=s, trained=play(a, s, selected, events=True)))
        print(
            "validation",
            max((v[0] for v in validation), default=0),
            "held",
            len(held),
            sum(r["trained"]["legal"] for r in held),
            flush=True,
        )
        _write(
            output,
            dict(
                nominal=shot,
                trials=scored,
                selected=selected,
                validation=[r[0] for r in validation],
                held=held,
            ),
        )
    finally:
        a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--output", type=Path, default=OUT / "driven-receiving.json.gz")
    args = p.parse_args()
    run(args.size, args.output)
