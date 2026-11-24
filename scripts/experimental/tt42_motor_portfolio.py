"""Measure complementarity of earlier and recent motor repertoires in paired physical practice."""

import argparse
import itertools
from pathlib import Path

import numpy as np

from . import tt37_reuse as retrieval
from . import tt40_low_lift as core
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt37_motor_clinic import AIM, geometry, schedules
from .tt38_motor_value import MotorValue

ROOT = Path("artifacts")
EXPERTS = [
    ROOT / "tt37/independent-cold-value.json.gz",
    ROOT / "tt39/endurance-04.json.gz",
    ROOT / "tt39/endurance-08.json.gz",
]
SOURCE = ROOT / "tt37/mix-rollin.json.gz"


def choices(engine, banks, x, fit, dt, eta, ready):
    arrays = [retrieval.options(b, x) for b in banks]
    moves = np.vstack(arrays)
    goals = schedules(*fit, eta, moves, dt, ready, round((eta + 1.05) / dt))
    f = np.zeros((len(moves), 9))
    _, outcome = engine.trajectories(goals, AIM, eta + 0.12, feedback=f)
    valid = np.isfinite(outcome[:, 0])
    _vx, vy, vz = f[:, 2:5].T
    flight = np.clip(
        (vz + np.sqrt(np.maximum(0, vz**2 + 19.62 * (f[:, 7] - 0.78)))) / 9.81,
        0.01,
        1.6,
    )
    forward = f[:, 6] + 0.94 * vy * flight
    cost = np.where(f[:, 1] > 0, 8 + abs(forward - 0.7), 16 + 20 * f[:, 0])
    if valid.any():
        cost[valid] = engine.motor_value.cost(outcome[valid])
    cut = np.r_[0, np.cumsum([len(a) for a in arrays])]
    picks = np.array(
        [i + int(np.argmin(cost[i:j])) for i, j in itertools.pairwise(cut)]
    )
    return moves, goals, picks, valid, f, outcome


def gather(out, source):
    data = _read(source)
    banks = [_read(p)["bank"] for p in EXPERTS]
    judge = ROOT / "tt37/carried-judgement.json.gz"
    core.joint.RECEIVE_Y = data["depth"]
    surf = core.setup(data["depth"])
    states = []
    outcomes = []
    ranks = []
    groups = []
    options = []
    feedback = []
    predictions = []
    ticks = []
    with core.centered(data["ready"]):
        a = core.arena()
        renderer, rig, stride, lo, hi = core.joint.resources(a)
        engine = core.OpticalClinic(body=True)
        engine.motor_value = MotorValue(_read(judge))
        actual = Rollout(a, 1)
        try:
            for c, r in zip(data["cards"], data["rows"], strict=True):
                a.launch(r["shot"], card=c)
                a.partner.reset(seed=c["seed"])
                engine.previous = None
                engine.bounced = False
                eye = core.joint.Vision(rig, a.control_dt)
                eye.eye = core.LearnedOptics(eye.eye, core.OPTICS)
                history = core.ArrivalHistory()
                ref = np.asarray(data["ready"]).copy()
                known = {
                    m["tick"]: m for m in r["records"] if not m.get("surface_shift")
                }
                for tick, u in enumerate(r["actions"]):
                    q = a.joints.copy()
                    obs = eye.update(
                        tick, renderer.capture() if tick % 2 == 0 else None, q
                    )
                    engine.observe(tick, eye.eye, q, ref)
                    obs = history.update(tick * a.control_dt, obs)
                    if (
                        tick in known
                        and obs is not None
                        and 0.03 < obs[0] < 0.19
                        and engine.sync()
                    ):
                        x = np.r_[obs[1:6], q[:5]]
                        if not np.allclose(known[tick]["context"], x, atol=1e-8):
                            raise ValueError("Different pre-action sensor context")
                        moves, goals, picks, _ok, f, y = choices(
                            engine,
                            banks,
                            x,
                            geometry(surf, obs),
                            a.control_dt,
                            obs[0],
                            surf.ready,
                        )
                        _, reward = actual.trajectories(goals, AIM, obs[0] + 0.12)
                        states.append(np.r_[obs[0], x])
                        options.append(moves)
                        ranks.append(picks)
                        outcomes.append(np.isfinite(reward[:, 0]))
                        feedback.append(f)
                        predictions.append(y)
                        groups.append(c["id"])
                        ticks.append(tick)
                    ref = np.clip(ref + np.asarray(u) * stride, lo, hi)
                    a.exchange(u, core.clinic.FAR)
                if digest_json(a.events) != digest_json(r["events"]):
                    raise ValueError("Source replay diverged")
                np.savez_compressed(
                    out,
                    states=states,
                    moves=options,
                    picks=ranks,
                    reward=outcomes,
                    feedback=feedback,
                    nominal=predictions,
                    release=groups,
                    tick=ticks,
                )
                print(c["id"], len(states), flush=True)
        finally:
            renderer.close()
            engine.twin.close()
            a.close()
    _write(
        out.with_suffix(".json"),
        {
            "source": str(source),
            "source_sha256": _hash(source),
            "checkpoint_sha256": _hash(out),
            "sources": {str(s): _hash(s) for s in [*EXPERTS, judge]},
            "boundary": "Each private action and prediction uses the live camera and encoder history. A separate physical simulator fork supplies each possible binary landing reward in practice only.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--source", type=Path, default=SOURCE)
    a = p.parse_args()
    gather(a.out, a.source)
