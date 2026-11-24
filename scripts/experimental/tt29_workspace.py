"""Can a table-clamped SO-101 reach unmodified recorded feeds, even instantaneously?"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .tt4_arena import BALL_RADIUS, TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _write
from .tt19_so101 import BLADE, GRIP_BODY, _GRASP_POS
from .tt25_recorded_exchange import step_ball, table_model
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def reach_sphere(arena):
    """Triangle inequality from the fixed shoulder pan pivot to any ball touching the blade."""
    m = arena.model
    shoulder = m.body("learner_shoulder").id
    node = m.body(GRIP_BODY).id
    lengths = []
    while node != shoulder:
        if node == 0:
            raise ValueError("Shoulder must be an ancestor of the gripper")
        lengths.append(np.linalg.norm(m.body_pos[node]))
        node = m.body_parentid[node]
    blade = m.geom(BLADE)
    # Includes the complete cylinder, ball and 1 cm allowance for the compliant weld.
    lengths.extend(
        (
            np.linalg.norm(_GRASP_POS + blade.pos),
            np.linalg.norm(blade.size[:2]),
            BALL_RADIUS,
            0.010,
        )
    )
    return arena.data.xpos[shoulder].copy(), float(sum(lengths))


def closest(path, center):
    a = path[:-1] - center
    deltas = np.diff(path, axis=0)
    fraction = np.clip(
        -np.sum(a * deltas, axis=1) / np.maximum(np.sum(deltas**2, axis=1), 1e-20),
        0.0,
        1.0,
    )
    return float(np.min(np.linalg.norm(a + fraction[:, None] * deltas, axis=1)))


def run(path, count):
    manifest = json.loads((OUTPUT / "recorded-secondary-reference.json").read_text())
    if _hash(path) != manifest["sha256"]:
        raise ValueError("Incorrect match dataset")
    all_feeds = [
        r for r in json.loads(path.read_bytes()) if r["pos_y"] > 0 and r["vel_y"] < 0
    ]
    indices = np.random.default_rng(290045).choice(len(all_feeds), count, replace=False)
    coeff = json.loads((OUTPUT / "recorded-flight.json").read_text())["held_scores"][
        "match_fit"
    ]["coefficients"]
    spheres = {}
    for fixture in ("table-edge", "archived"):
        env = ReferenceReceiver(fixture=fixture)
        try:
            spheres[fixture] = reach_sphere(env)
        finally:
            env.close()
    m = table_model()
    d = mujoco.MjData(m)
    ball = m.body("ball").id
    net, table, floor = (m.geom(name).id for name in ("net", "table", "floor"))
    w = np.empty(3)
    rows = []
    for i in indices:
        shot = all_feeds[i]
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = (shot["pos_x"], shot["pos_y"], TABLE_HEIGHT + shot["pos_z"])
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qvel[:] = [
            shot[prefix + axis] for prefix in ("vel_", "w_vel_") for axis in "xyz"
        ]
        mujoco.mj_forward(m, d)
        track = []
        bouncing = False
        for _ in range(round(2 / m.opt.timestep)):
            prev = d.qpos[:3].copy()
            step_ball(m, d, coeff, w, ball)
            contacts = [c for c in d.contact if c.efc_address >= 0]
            table_top = any(
                table in c.geom and abs(c.frame[2]) > 0.55 for c in contacts
            )
            if table_top and not bouncing:
                if track or d.qpos[1] >= 0:
                    break
                track.append(prev)
            if track:
                track.append(d.qpos[:3].copy())
            if any(
                floor in c.geom or (table in c.geom and not table_top) for c in contacts
            ):
                break
            bouncing = table_top
        distances = (
            {
                key: closest(np.asarray(track), center) - radius
                for key, (center, radius) in spheres.items()
            }
            if track
            else {}
        )
        rows.append(dict(id=shot["id"], outside_by_m=distances))
    valid = [r for r in rows if r["outside_by_m"]]
    summary = dict(
        sampled=len(rows),
        valid_near_bounce=len(valid),
        outside={k: sum(r["outside_by_m"][k] > 0 for r in valid) for k in spheres},
    )
    receipt = dict(
        dataset_sha256=manifest["sha256"],
        seed=290045,
        spheres={k: [c.tolist(), r] for k, (c, r) in spheres.items()},
        summary=summary,
        rows=rows,
    )
    out = OUTPUT.parent / "tt29"
    out.mkdir(exist_ok=True)
    _write(out / "workspace.json.gz", receipt)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--count", type=int, default=384)
    args = parser.parse_args()
    run(args.data, args.count)
