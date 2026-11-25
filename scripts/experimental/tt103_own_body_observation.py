"""Measure if the arm misestimates the position, face or velocity of its own paddle."""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from . import tt84_installation_system as system
from . import tt84_installation_play as play
from .tt37_optical_clinic import OpticalClinic
from .tt4_transfer_trial import _read, _write, _hash
from .tt19_view_full_game import digest_json

ROOT = Path("artifacts/tt103_own_body")
mpc = play.MP


def rot(q):
    return Rotation.from_quat(np.roll(q, -1)).as_matrix()


def blade(model, data, qi, vi, geom):
    p = data.qpos[qi : qi + 7]
    v = data.qvel[vi : vi + 6]
    r = rot(p[3:])
    offset = model.geom_pos[geom]
    normal = r @ rot(model.geom_quat[geom])[:, 2]
    c = p[:3] + r @ offset
    cv = v[:3] + r @ np.cross(v[3:], offset)
    return c, normal, cv


def survey(kind):
    doc = _read(system.BUILD)
    item = doc["rows"][0 if kind == "stock" else doc["chosen"]]
    mount = item["candidate"]
    ready = item["commission"]["ready"]
    path = system.ROOT / (kind + "-online.json.gz")
    source = _read(path)
    rows = []
    with (
        system.fixture.school.enabled(),
        system.installed(mount["v"], mount["mount"]),
        mpc.centered(ready),
    ):
        a = mpc.arena(width=64, height=48)
        est = OpticalClinic(body=True)
        try:
            g = a._racket_ids[1]
            e = est.twin._racket_ids[1]
            ids = a._arm_ids[1][2]
            stride = a._joint_increments[ids] * a.control_dt / a.model.opt.timestep
            for c, r in zip(source["cards"], source["rows"], strict=True):
                a.launch(r["shot"], card=c)
                a.partner.reset(seed=c["seed"])
                ref = np.asarray(ready).copy()
                notes = {q["tick"]: q for q in r["searches"]}
                outcomes = {}
                for o in r["placements"]["delivered"]:
                    if not o["far_contact"]:
                        continue
                    end = (o["near_contact"] or r["events"][-1])["step"]
                    for t in notes:
                        if o["far_contact"]["step"] < t < end:
                            outcomes[t] = bool(o["near_landed"])
                readings = []
                for t, act in enumerate(r["actions"]):
                    est.propagate(t, a.joints, ref)
                    if t in outcomes:
                        actual = blade(a.model, a.data, est.paddle_q, est.paddle_v, g)
                        own = blade(
                            est.body_model,
                            est.body_state,
                            est.paddle_q,
                            est.paddle_v,
                            e,
                        )
                        readings.append(
                            dict(
                                tick=t,
                                landed=outcomes[t],
                                nominal=bool(notes[t].get("legal_trials")),
                                position_mm=float(
                                    1000 * np.linalg.norm(actual[0] - own[0])
                                ),
                                face_deg=float(
                                    np.degrees(
                                        np.arccos(np.clip(actual[1] @ own[1], -1, 1))
                                    )
                                ),
                                velocity_mps=float(np.linalg.norm(actual[2] - own[2])),
                            )
                        )
                    a.exchange(act, mpc.FAR_COMMAND)
                    ref = np.clip(
                        ref + stride * act, a._joint_low[ids], a._joint_high[ids]
                    )
                if digest_json(a.events) != digest_json(r["events"]):
                    raise ValueError("Replay deviated")
                rows.append(
                    dict(id=c["id"], readings=readings, events=digest_json(r["events"]))
                )
        finally:
            a.close()
            est.twin.close()
    metrics = {}
    for group in ("all", "scored", "missed", "false_accept", "false_reject"):
        samples = [
            r
            for g in rows
            for r in g["readings"]
            if group == "all"
            or group == "scored"
            and r["landed"]
            or group == "missed"
            and not r["landed"]
            or group == "false_accept"
            and r["nominal"]
            and not r["landed"]
            or group == "false_reject"
            and not r["nominal"]
            and r["landed"]
        ]
        if not samples:
            continue
        metrics[group] = dict(
            search_windows=len(samples),
            **{
                k: np.quantile([r[k] for r in samples], [0.5, 0.9, 0.99]).tolist()
                for k in ("position_mm", "face_deg", "velocity_mps")
            },
        )
    ROOT.mkdir(exist_ok=True)
    _write(
        ROOT / (kind + ".json"), dict(source=_hash(path), rows=rows, metrics=metrics)
    )
    print(kind, metrics, flush=True)


if __name__ == "__main__":
    for kind in ("stock", "installed"):
        survey(kind)
