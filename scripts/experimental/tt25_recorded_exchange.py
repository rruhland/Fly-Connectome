"""Replay measured ball flights through a physical table bounce to the next racket.

This audit uses an isolated ball, net and table from the regulation MuJoCo model.
The second racket is represented only by its observed event time and position.
"""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .tt4_arena import BALL_MASS, BALL_RADIUS, MODEL_PATH, SPIN_DECAY, TABLE_HEIGHT
from .tt4_flight_validation import predict, read_shots
from .tt25_recorded_flight import HELD, OUTPUT, ROOT, SHA256, quantiles, vector


def measurements(path):
    if sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Expected the pinned match dataset")
    rallies = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            key, index = row["rally_id"].rsplit("/event", 1)
            if row["player_id"] in HELD:
                rallies[(row["player_id"], key)].append((int(index), row))
    for key, rows in rallies.items():
        rows.sort()
        first_shot = next((i for i, r in rows if r["type"].startswith("shot_")), None)
        for (i, hit), (j, bounce), (k, reply) in zip(rows, rows[1:], rows[2:]):
            if (first_shot is None or i == first_shot or j != i+1 or k != j+1
                    or not hit["type"].startswith("shot_") or not bounce["type"].startswith("bounce_")
                    or not reply["type"].startswith("shot_") or hit["type"] == reply["type"]):
                continue
            yield key, hit, bounce, reply


def rotate(vec):
    return np.array([-vec[1], vec[0], vec[2]])


def position(row):
    return rotate(vector(row, "ball_pos")) + [0., 0., TABLE_HEIGHT]


def table_model(*, thin_shell=False):
    document = ET.parse(MODEL_PATH).getroot()
    world = document.find("worldbody")
    for child in list(world):
        if child.get("name") not in {"table", "net", "ball", "floor"}:
            world.remove(child)
    if thin_shell:
        inertia = 2 / 3 * BALL_MASS * BALL_RADIUS**2
        ET.SubElement(world.find("body[@name='ball']"), "inertial", pos="0 0 0",
                      mass=str(BALL_MASS), diaginertia=" ".join([str(inertia)] * 3))
    for child in list(document):
        if child.tag not in {"option", "default", "compiler", "worldbody"}:
            document.remove(child)
    return mujoco.MjModel.from_xml_string(ET.tostring(document, encoding="unicode"))


def simulate(model, coeff, cohort):
    data = mujoco.MjData(model)
    table, net = (model.geom(name).id for name in ("table", "net"))
    body = model.body("ball").id
    w = np.zeros(3)
    errors, land_errors, time_errors = [], [], []
    touched_net = 0
    step = model.opt.timestep
    for _, hit, observed_bounce, reply in cohort:
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = position(hit)
        data.qpos[3:7] = [1., 0., 0., 0.]
        data.qvel[:3] = rotate(vector(hit, "ball_vel_out"))
        data.qvel[3:6] = rotate(vector(hit, "ball_spin_out"))
        mujoco.mj_forward(model, data)
        start = float(hit["timestamp"])
        horizon = float(reply["timestamp"]) - start
        if horizon <= 0:
            raise ValueError("Nonpositive reply interval")
        first_bounce = None
        hit_net = False
        while data.time < horizon:
            last_pos = data.qpos[:3].copy()
            vx, vy, vz = data.qvel[:3]
            mujoco.mju_rotVecQuat(w, data.qvel[3:6], data.qpos[3:7])
            wx, wy, wz = w
            drag = -BALL_MASS * coeff[0] * (vx*vx + vy*vy + vz*vz)**.5
            magnus = BALL_MASS * coeff[1]
            data.xfrc_applied[body, :3] = (drag*vx + magnus*(wy*vz - wz*vy),
                                           drag*vy + magnus*(wz*vx - wx*vz),
                                           drag*vz + magnus*(wx*vy - wy*vx))
            data.xfrc_applied[body, 3:] = -model.body_inertia[body, 0] * SPIN_DECAY * w
            mujoco.mj_step(model, data)
            for contact in data.contact:
                if contact.efc_address < 0:
                    continue
                if net in contact.geom:
                    hit_net = True
                if first_bounce is None and table in contact.geom and abs(contact.frame[2]) > .55:
                    first_bounce = data.time - step, contact.pos[:2].copy()
        frac = (data.time - horizon) / step
        at_reply = frac*last_pos + (1-frac)*data.qpos[:3]
        errors.append(100*np.linalg.norm(at_reply - position(reply)))
        touched_net += hit_net
        if first_bounce is not None:
            time_errors.append(1000*abs(first_bounce[0] - (float(observed_bounce["timestamp"])-start)))
            land_errors.append(100*np.linalg.norm(first_bounce[1] - position(observed_bounce)[:2]))
    return dict(segments=len(cohort), any_net_collision=touched_net, first_table_contacts=len(land_errors),
                next_racket_position_error_cm=quantiles(errors),
                contacted_table_xy_error_cm=quantiles(land_errors), contacted_table_time_error_ms=quantiles(time_errors))


def legacy_check(path, models):
    reference = json.loads((ROOT / "artifacts/tt4/measured-flight.json").read_text())
    if reference["archive_sha256"] != sha256(path.read_bytes()).hexdigest():
        raise ValueError("Wrong SpinDOE archive")
    data = read_shots(path)
    held = np.isin(data[0], reference["test_shots"])
    if held.sum() != len(reference["test_shots"]):
        raise ValueError("Missing legacy test shots")
    held = tuple(x[held] for x in data)
    return {name: quantiles(100*np.linalg.norm(predict(held, coeff), axis=1)) for name, coeff in models.items()}


def run(csv_path, zip_path):
    fit = json.loads((OUTPUT / "recorded-flight.json").read_text())
    models = {name: fit["held_scores"][name]["coefficients"] for name in ("current", "match_fit")}
    cohort = list(measurements(csv_path))
    model = table_model()
    results = dict(match_csv_sha256=SHA256, previous_dataset_sha256=sha256(zip_path.read_bytes()).hexdigest(),
                   model_xml_sha256=sha256(MODEL_PATH.read_bytes()).hexdigest(), unique_rallies=len({r[0] for r in cohort}),
                   table=list(model.geom("table").solref), friction=list(model.geom("table").friction),
                   dt=model.opt.timestep, legacy_spindoe_error_cm=legacy_check(zip_path, models),
                   through_bounce={name: simulate(model, coeff, cohort) for name, coeff in models.items()})
    shell = table_model(thin_shell=True)
    results["thin_shell_match_fit"] = simulate(shell, models["match_fit"], cohort)
    results["ball_inertia_kg_m2"] = dict(solid=float(model.body("ball").inertia[0]),
                                          shell=float(shell.body("ball").inertia[0]))
    (OUTPUT / "recorded-exchange.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(results["through_bounce"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("spindoe_zip", type=Path)
    args = parser.parse_args()
    run(args.csv, args.spindoe_zip)
