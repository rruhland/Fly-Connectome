"""Test whether moving through the second contact can produce a legal return."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _read, _write
from .tt19_so101 import BLADE, GRIP_BODY, _GRASP_POS
from .tt25_second_ball import execute
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def center_jacobian(arena, q):
    model, data = arena.model, arena.data
    grip = model.body(GRIP_BODY).id
    local = _GRASP_POS + model.geom_pos[model.geom(BLADE).id]
    joints = arena._arm_ids[1][0][:5]

    def center(pose):
        data.qpos[joints] = pose
        mujoco.mj_kinematics(model, data)
        return data.xpos[grip] + data.xmat[grip].reshape(3, 3) @ local

    delta = np.eye(5) * 0.001
    return np.stack([(center(q + d) - center(q - d)) / 0.002 for d in delta], axis=1)


def run(path):
    shots = {r["id"]: r for r in json.loads(path.read_bytes())}
    previous = {r["id"]: r for r in _read(OUTPUT / "recorded-live-development.json.gz")["cases"]}
    cases = []
    arena = ReferenceReceiver()
    arena.max_steps = 450
    design = qmc.Sobol(4, scramble=True, seed=482).random_base2(5)
    try:
        for scene in _read(OUTPUT / "recorded-second-ball.json.gz")["practice"]:
            sid = scene["id"]
            shot, row = shots[sid], previous[sid]
            options = []
            for t in scene["trials"]:
                if t["contacts"] < 2:
                    continue
                q = np.asarray(t["goal"])
                execute(arena, shot, row, scene["trigger"], q)
                impacts = [e for e in arena.events if e.get("collider") == "learner_racket"]
                if len(impacts) < 2:
                    continue
                vx, vy, vz = impacts[-1]["velocity"]
                options.append((float(vy + 0.35 * vz - 0.4 * abs(vx)), impacts[-1]["step"] - 1, q))
            attempts = []
            for _, impact, q in sorted(options, key=lambda r: -r[0])[:5]:
                jac = center_jacobian(arena, q)
                inverse = jac.T @ np.linalg.inv(jac @ jac.T + 0.0004 * np.eye(3))
                for u in design:
                    cart = [1.6 * (u[0] - 0.5), 0.5 + u[1] * 1.4, -0.12 + u[2] * 1.1]
                    v = inverse @ cart
                    v *= min(1.0, 5.2 / np.max(np.abs(v)))
                    lead = 4 * u[3]
                    result, _ = execute(arena, shot, row, scene["trigger"], q, through=(impact, v, lead))
                    attempts.append(
                        dict(goal=q.tolist(), rate=v.tolist(), impact=impact, lead=lead, **result)
                    )
                    if result["near"] >= 2:
                        repeat, _ = execute(
                            arena, shot, row, scene["trigger"], q, through=(impact, v, lead)
                        )
                        if repeat != result:
                            raise ValueError("Second-ball swing changed on replay")
                        break
                if attempts[-1]["near"] >= 2:
                    break
            cases.append(dict(id=sid, anchor_options=len(options), attempts=attempts))
            print(sid, len(attempts), max((r["near"] for r in attempts), default=0), flush=True)
        _write(OUTPUT / "recorded-through-contact.json.gz", dict(cases=cases))
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    run(p.parse_args().deepmind)
