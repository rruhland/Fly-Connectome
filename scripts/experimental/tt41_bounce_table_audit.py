"""Blindly read a frozen solo-contact law against earlier real optical table contacts."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_public_body import stock
from .tt29_rally_physics import BladeIK
from .tt37_opponent import arena
from .tt41_bounce_effect import FRAME, OUT, error, profile

CAMERA = Path("artifacts/tt40-contact-effect/paired-camera.json.gz")
MODES = ("solo", "nominal_moving", "nominal_stationary")


def read_contact(geometry, row):
    e = row["impact"]
    if e is None:
        return None
    phase = e["time"] / FRAME - row["at"]
    i = round(phase)
    enc = np.asarray(row["q"])[:, :5]
    ix = np.arange(max(0, i - 2), min(len(enc), i + 3))
    if len(ix) < 3:
        return None
    time = (ix - phase) * FRAME
    q, dq = np.linalg.lstsq(
        np.c_[np.ones(len(ix)), time, time**2], enc[ix], rcond=None
    )[0][:2]
    center, n, j, _ = geometry.pose(q, jacobian=True)
    gap = np.asarray(e["center"]) - center
    if gap @ n < 0.0:
        n = -n
    surface = gap - (gap @ n) * n
    angular = geometry.angular[:, geometry.dofs] @ dq
    speed = j @ dq + np.cross(angular, surface)
    u = np.asarray(e["incoming"])
    v = np.asarray(e["outgoing"])
    return {
        "n": n.tolist(),
        "pre": u.tolist(),
        "post": v.tolist(),
        "blade_speed": speed.tolist(),
        "normal_blade_speed": float(n @ speed),
        "centered_radius_m": float(np.linalg.norm(surface)),
        "closing_m_s": float(-((u - speed) @ n)),
        "receive_height_m": e["center"][2],
        "score": row["score"],
        "contact_time_s": e["time"],
    }


def audit(law, camera, output):
    checkpoint = _read(law)
    doc = _read(camera)
    rows = []
    a = arena()
    b = stock(fixture="table-edge")
    try:
        k = BladeIK(a)
        # Only the static installation dimensions and issued-reference contract enter
        # this audit. Ball pose, velocity and contact labels come exclusively from RGB.
        fixtures = {
            "solo_base_m": b._model.body("learner_base").pos.tolist(),
            "table_base_m": a.model.body("learner_base").pos.tolist(),
            "same_shoulder_through_roll_ref_slew_rad_s": np.allclose(
                np.array(b._rates)[:5],
                (a._joint_increments[a._arm_ids[1][2]] / a.model.opt.timestep)[:5],
            ),
            "solo_grip_active": bool(b._rates[5]),
            "table_grip_active": bool(a._joint_increments[a._arm_ids[1][2]][5]),
            "solo_max_command": 1.0,
            "table_max_command": a.rate_scale,
            "table_speed_torque_envelope": a.motor_envelope,
            "solo_contact_material": "randomized at release",
            "table_nominal_face_solref": a.model.geom_solref[
                a.model.geom("learner_racket").id
            ].tolist(),
        }
        for course in doc["rows"]:
            for tag, tape in course["trials"].items():
                row = read_contact(k, tape)
                if row is None:
                    continue
                row.update(
                    release=course["card"]["id"],
                    contrast=tag,
                    solo=error(row, checkpoint["pooled"]),
                    nominal_moving=error(row, [1.85, 0.0]),
                    nominal_stationary=error(row, [1.85, 0.0], quiet=True),
                )
                rows.append(row)
    finally:
        b.close()
        a.close()
    pairs = []
    for new in rows:
        if new["contrast"] == "raw" or not new["score"]:
            continue
        zero = next(
            (
                r
                for r in rows
                if r["release"] == new["release"]
                and r["contrast"] == "raw"
                and not r["score"]
            ),
            None,
        )
        if zero is None:
            continue
        z0, z1 = np.asarray(zero["post"]), np.asarray(new["post"])
        signed = z1 - z0
        pairs.append(
            dict(
                release=new["release"],
                contrast=new["contrast"],
                raw=zero["score"],
                issued=new["score"],
                delta_optical_m_s=signed.tolist(),
                **{
                    m: (signed + np.asarray(new[m]) - np.asarray(zero[m])).tolist()
                    for m in MODES
                },
            )
        )
    results = {m: profile(rows, m) for m in MODES}
    out = {
        "schema": "tt41-solo-effect-optical-table-check-v1",
        "practice_source_sha256": _hash(law),
        "optical_table_source_sha256": _hash(camera),
        "table_labels": "archived actual issued RGB/encoder observations, no model contact labels",
        "description": "Not used for fitting the solo law or selecting table assessment motions.",
        "fixtures": fixtures,
        "sensor_contacts": len(rows),
        "distinct_releases": len({r["release"] for r in rows}),
        "source_closing_m_s": checkpoint["sample_range"]["closing"],
        "table_closing_m_s": [
            min(r["closing_m_s"] for r in rows),
            max(r["closing_m_s"] for r in rows),
        ],
        "results": results,
        "records": rows,
        "matched_command_deltas": pairs,
    }
    _write(output, out)
    print(results, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, nargs="?", default=OUT / "table-audit.json")
    p.add_argument("--law", type=Path, default=OUT / "effect.json")
    p.add_argument("--camera", type=Path, default=CAMERA)
    q = p.parse_args()
    audit(q.law, q.camera, q.out)
