"""Separate the receiving arm's geometric opportunity from its policy score."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .tt4_arena import BALL_RADIUS, TABLE_HEIGHT
from .tt18_plant import PlantSettings
from .tt19_so101 import BLADE, GRIP_BODY, PADDLE_BODY, _GRASP_POS
from .tt23_sequence import CONFIG
from .tt25_arm_replay import SolidArm
from .tt25_body_contacts import MOVING
from .tt25_recorded_exchange import step_ball, table_model
from .tt25_recorded_flight import OUTPUT, quantiles


def workspace():
    arm = SolidArm(**dict(CONFIG, plant=PlantSettings()))
    try:
        arm.reset(seed=19, render=False)
        model, data = arm.model, arm.data
        table = model.geom("table").id
        meshes = [i for i, b in enumerate(model.geom_bodyid) if model.body(b).name in MOVING]
        grip, blade = model.body(GRIP_BODY).id, model.geom(BLADE).id
        paddle = model.body(PADDLE_BODY).jntadr[0]
        paddle_q = model.jnt_qposadr[paddle]
        joints = arm._arm_ids[1][0][:5]
        axes = arm._arm_ids[1][2][:5]
        sequence = qmc.Sobol(5, scramble=True, seed=1357).random_base2(13)
        poses = qmc.scale(sequence, arm._joint_low[axes], arm._joint_high[axes])
        points = []
        for pose in poses:
            data.qpos[joints] = pose
            mujoco.mj_kinematics(model, data)
            data.qpos[paddle_q : paddle_q + 3] = (
                data.xpos[grip] + data.xmat[grip].reshape(3, 3) @ _GRASP_POS
            )
            data.qpos[paddle_q + 3 : paddle_q + 7] = data.xquat[grip]
            mujoco.mj_kinematics(model, data)
            if any(
                mujoco.mj_geomDistance(model, data, i, table, 0.003, None) < 0 for i in [blade, *meshes]
            ):
                continue
            points.append(data.geom_xpos[blade].copy())
        radius = float(model.geom_size[blade, 0])
        return cKDTree(points), radius, len(poses)
    finally:
        arm.close()


def trajectories(rows, coeff, tree, radius):
    model = table_model()
    table = model.geom("table").id
    floor = model.geom("floor").id
    body = model.body("ball").id
    data = mujoco.MjData(model)
    world_spin = np.zeros(3)
    results = []
    for shot in rows:
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = [shot["pos_x"], shot["pos_y"], TABLE_HEIGHT + shot["pos_z"]]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qvel[:3] = [shot["vel_" + a] for a in "xyz"]
        data.qvel[3:6] = [shot["w_vel_" + a] for a in "xyz"]
        mujoco.mj_forward(model, data)
        bounced = False
        prev_table = False
        track = []
        for _ in range(round(2.0 / model.opt.timestep)):
            step_ball(model, data, coeff, world_spin, body)
            contact = [c for c in data.contact if c.efc_address >= 0]
            if any(floor in c.geom for c in contact):
                break
            tabletop = any(table in c.geom and abs(c.frame[2]) > 0.55 for c in contact)
            if tabletop and not prev_table:
                if bounced or data.qpos[1] >= 0:
                    break
                bounced = True
            if bounced:
                track.append(data.qpos[:3].copy())
            prev_table = tabletop
        distance = float(tree.query(track)[0].min()) if track else None
        results.append(
            dict(
                id=shot["id"],
                received_bounce=bounced,
                closest_sampled_center_m=distance,
                near_blade=bool(distance is not None and distance <= radius + BALL_RADIUS),
            )
        )
    return results


def run(path):
    raw = path.read_bytes()
    reference = json.loads((OUTPUT / "recorded-secondary-reference.json").read_text())
    if sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Unexpected DeepMind rally file")
    source = [r for r in json.loads(raw) if r["pos_y"] > 0 and r["vel_y"] < 0]
    # No centering, re-aiming, speed cuts or filtering for a success.
    selection = np.random.default_rng(26863).choice(len(source), 192, replace=False)
    rows = [source[i] for i in selection]
    tree, radius, poses = workspace()
    coeff = json.loads((OUTPUT / "recorded-flight.json").read_text())["held_scores"]["match_fit"][
        "coefficients"
    ]
    cases = trajectories(rows, coeff, tree, radius)
    summary = dict(
        source_sha256=reference["sha256"],
        incoming_population=len(source),
        sampled=len(rows),
        incoming_speed_mps=quantiles([np.linalg.norm([r["vel_" + a] for a in "xyz"]) for r in rows]),
        sampled_poses=poses,
        table_clear_poses=tree.n,
        ball_radius=BALL_RADIUS,
        blade_radius=radius,
        court_dimensions_m=[2.74, 1.525],
        valid_incoming=sum(r["received_bounce"] for r in cases),
        near_sampled_blade=sum(r["near_blade"] for r in cases),
        cases=cases,
    )
    (OUTPUT / "recorded-reach.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print({k: v for k, v in summary.items() if k != "cases"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deepmind", type=Path)
    run(parser.parse_args().deepmind)
