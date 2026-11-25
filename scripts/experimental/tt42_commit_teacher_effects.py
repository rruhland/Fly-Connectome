"""Look for transferable visual action effects in actually issued same-ball alternatives."""

import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt33_joint import FAR_COMMAND
from .tt36_impact_refine import Frames
from .tt42_commit_teacher import OUT, SOURCE

CAL = Path("artifacts/tt36/impact-model.json")


def traces(source, output, archive):
    d = _read(source)
    old = _read(archive)
    if not d.get("complete") or d["source_sha256"] != _hash(archive):
        raise ValueError(
            "Use a completed physical intervention with its original matching tape"
        )
    i = next(i for i, c in enumerate(old["cards"]) if c == d["card"])
    base = old["rows"][i]
    card = d["card"]
    at = d["fork_tick"]
    steps = []
    acts = []
    view = []
    sensor_sha = None
    paired = []
    options = [dict(base, legal=False, answered=False, family="original")] + d[
        "variants"
    ]
    cal = _read(CAL)["models"]["60"]
    with mpc.centered(d["ready"]):
        a = mpc.arena()
        camera, rig, stride, lo, hi = mpc.resources(a)
        try:
            for mode, r in enumerate(options):
                a.launch(d["shot"], card=card)
                a.partner.reset(seed=card["seed"])
                rgb = Frames(rig)
                rig.begin()
                reference = np.asarray(d["ready"]).copy()
                row = []
                for tick, action in enumerate(r["actions"]):
                    current = a.joints.copy()
                    if at - 6 <= tick < at + 105:
                        rgb.observe(camera.capture(), tick * mpc.DT, current)
                        row.append((tick, current.tolist(), reference.tolist(), action))
                    a.exchange(action, FAR_COMMAND)
                    reference = np.clip(reference + stride * np.asarray(action), lo, hi)
                if digest_json(a.events) != digest_json(r["events"]):
                    raise ValueError(
                        "Raw RGB effects teacher replay changed its physical outcome"
                    )
                hits = [
                    e
                    for e in rgb.estimates(cal)
                    if at * mpc.DT < e["time"] < (at + 65) * mpc.DT
                ]
                position = [
                    [t, ball.tolist(), None if body is None else body.tolist()]
                    for t, ball, body in rgb.stereo
                ]
                common = digest_json(
                    {
                        "motion": [t for t in row if t[0] < at],
                        "rgb_positions": [t for t in position if t[0] < at * mpc.DT],
                    }
                )
                if sensor_sha is not None and common != sensor_sha:
                    raise ValueError(
                        "Matched camera, encoder and action prefix changed"
                    )
                sensor_sha = common
                pixel_label = hits[0] if hits else None
                view.append(
                    {
                        "mode": mode,
                        "source_family": r["family"],
                        "teacher_actual_legal": r["legal"],
                        "teacher_actual_answered": r["answered"],
                        "capture_hz": 60,
                        "camera_positions": position,
                        "measured_contact": pixel_label,
                        "private_codes_absent": True,
                        "referee_sha256": digest_json(r["events"]),
                    }
                )
                steps.extend((mode, t, q, ref) for t, q, ref, _ in row)
                acts.extend(u for _, _, _, u in row)
                print(
                    mode,
                    r["family"],
                    r["legal"],
                    None if pixel_label is None else pixel_label["outgoing"],
                    flush=True,
                )
        finally:
            camera.close()
            a.close()
    take = [r for r in view if r["measured_contact"] is not None]
    for j, v in enumerate(take):
        for w in take[:j]:
            p, q = v["measured_contact"], w["measured_contact"]
            paired.append(
                {
                    "a": w["mode"],
                    "b": v["mode"],
                    "a_scored": w["teacher_actual_legal"],
                    "b_scored": v["teacher_actual_legal"],
                    "delta_outgoing_m_s": (
                        np.asarray(p["outgoing"]) - q["outgoing"]
                    ).tolist(),
                    "contact_time_delta_s": p["time"] - q["time"],
                }
            )
    stamps, enc, ref = zip(
        *[(np.array([j, t]), q, s) for j, t, q, s in steps], strict=True
    )
    npz = output.with_suffix("").with_suffix(".npz")
    np.savez_compressed(
        npz,
        stamps=np.asarray(stamps),
        encoders=np.asarray(enc),
        previous_references=np.asarray(ref),
        issued_normalized_rates=np.asarray(acts),
    )
    _write(
        output,
        {
            "schema": "tt42-camera-only-same-offer-effects-v1",
            "source_sha256": _hash(source),
            "camera_calibration_sha256": _hash(CAL),
            "source_is_training": True,
            "sensory_native_truth_excluded": True,
            "common_prefix_sha256": sensor_sha,
            "motion_sha256": _hash(npz),
            "camera_response_rows": view,
            "paired_differences": paired,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=OUT / "same-offer-diversity.json.gz")
    p.add_argument("--output", type=Path, default=OUT / "same-offer-camera.json.gz")
    p.add_argument("--archive", type=Path, default=SOURCE)
    c = p.parse_args()
    traces(c.source, c.output, c.archive)
