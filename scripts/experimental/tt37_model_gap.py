"""Localize planning error with controlled substitutions in isolated calibration trials."""

import argparse
from pathlib import Path

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _read, _write
from .tt29_rally_physics import BladeIK, Rollout, mujoco
from .tt31_ready import centered
from .tt37_belief import ArrivalHistory
from .tt37_motor_clinic import AIM, geometry, schedules
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt37_posture import Surface, sample


def trials(engine, calls, eta):
    f = np.zeros((len(calls), 9))
    _, scores = engine.trajectories(calls, AIM, eta + 0.12, feedback=f)
    return [
        np.isfinite(scores[:, 0]).tolist(),
        f[:, 1].tolist(),
        np.round(f[:, 2:5], 4).tolist(),
    ]


def audit(path, out, budget):
    d = _read(path)
    joint.RECEIVE_Y = d["depth"]
    rows = []
    rng = np.random.default_rng(373131)
    with centered(d["ready"]):
        a = arena()
        camera, rig, stride, lo, hi = joint.resources(a)
        twin = OpticalClinic()
        t = twin.twin
        oracle = Rollout(a, 1)
        ik = BladeIK(a)
        surface = Surface(ik, sample(ik), d["depth"])
        try:
            for card, row in zip(d["cards"], d["rows"], strict=True):
                indexed = {r["tick"]: r for r in row["records"]}
                ref = surface.ready.copy()
                a.launch(row["shot"], card=card)
                a.partner.reset(seed=card["seed"])
                eye = joint.Vision(rig, a.control_dt)
                twin.previous = None
                twin.bounced = False
                hist = ArrivalHistory()
                for k, cmd in enumerate(row["actions"]):
                    now = k * a.control_dt
                    obs = eye.update(
                        k, camera.capture() if k % 2 == 0 else None, a.joints
                    )
                    obs = hist.update(now, obs)
                    twin.observe(k, eye.eye, a.joints, ref.copy())
                    if k in indexed and obs is not None and twin.sync():
                        r = indexed[k]
                        fit = geometry(surface, obs)
                        eta = obs[0]
                        anchor = np.array(r["action"])
                        choices = (
                            anchor
                            + np.r_[np.zeros((1, 10)), rng.normal(0, 1, (38, 10))]
                            * np.r_[np.full(5, 0.062), np.full(5, 0.48)]
                        )
                        calls = schedules(
                            *fit,
                            eta,
                            choices,
                            a.control_dt,
                            ref,
                            round((eta + 1.05) / a.control_dt),
                        )

                        event = {
                            "id": card["id"],
                            "tick": k,
                            "eta": eta,
                            "exact": trials(oracle, calls, eta),
                            "received": bool(a._received_bounce),
                            "saw_bounce": bool(twin.bounced),
                            "p": (a.ball - t.ball).tolist(),
                            "v": (a.velocity - t.velocity).tolist(),
                            "actual_spin": a.spin.tolist(),
                            "optical": trials(twin, calls, eta),
                        }
                        for name in ("rotation", "p_and_v", "full_ball", "full_body"):
                            twin.sync()
                            if name == "full_body":
                                ball = t.data.qpos[
                                    t._ball_joint : t._ball_joint + 7
                                ].copy()
                                speed = t.data.qvel[
                                    t._ball_dof : t._ball_dof + 6
                                ].copy()
                                t.data.qpos[:] = a.data.qpos
                                t.data.qvel[:] = a.data.qvel
                                t.data.ctrl[:] = a.data.ctrl
                                t.data.qpos[t._ball_joint : t._ball_joint + 7] = ball
                                t.data.qvel[t._ball_dof : t._ball_dof + 6] = speed
                            if name in ("p_and_v", "full_ball"):
                                t.ball[:] = a.ball
                                t.velocity[:] = a.velocity
                                t._received_bounce = a._received_bounce
                            if name in ("rotation", "full_ball"):
                                t.data.qvel[t._ball_dof + 3 : t._ball_dof + 6] = a.spin
                            mujoco.mj_forward(t.model, t.data)
                            event[name] = trials(twin, calls, eta)
                        rows.append(event)
                        _write(out, {"rows": rows, "source": str(path)})
                        if len(rows) >= budget:
                            return
                    a.exchange(cmd, np.zeros(5))
                    ref = np.clip(ref + np.array(cmd) * stride, lo, hi)
        finally:
            camera.close()
            a.close()
            t.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--budget", type=int, default=32)
    args = p.parse_args()
    audit(args.source, args.out, args.budget)
