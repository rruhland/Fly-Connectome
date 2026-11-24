"""Fit from earlier stereo play and score disjoint monocular observations."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_mono_range import MonoFlight, RangeFit, appearance, triangulated_range
from .tt29_partner import FOREHAND_LANES
from .tt29_partner_fast import FastPartnerArena
from .tt29_rally_physics import OUTPUT, measured_launches
from .tt29_vision import FlightPractice, OpticalFlight, RaySample, Rig, ball_pixels

MEMORY = OUTPUT / "camera-action-memory.json.gz"
RESULT = OUTPUT / "tt29_mono_assay.json.gz"
HELD = (15489, 15400, 7471, 2950, 12670, 8388, 9447, 7271)
TRAIN_ID = 7345
TRAIN_CAP = 336


def arena():
    return FastPartnerArena(strike_ready="lateral", width=384, height=288, clearance_guard=True,
                           targets=FOREHAND_LANES, guard_table=True, max_steps=9000, fast_search=True)


def prior_course(plant, rig):
    prior = _read(MEMORY)
    row = next(r for r in prior["rows"] if r["shot"]["id"] == TRAIN_ID)
    model = RangeFit()
    plant.launch(row["shot"])
    plant.partner.reset(seed=row.get("opponent_seed", TRAIN_ID))
    plant.streams = ("left", "right")
    agreed = []
    played = 0
    for tick, command in enumerate(row["actions"][:TRAIN_CAP]):
        images = plant.render_cameras()
        t = tick * plant.control_dt
        left, right = (ball_pixels(images[stream]) for stream in plant.streams)
        if left is not None and right is not None and not (left.clipped or right.clipped):
            lcam, rcam = (rig.fixed[stream] for stream in plant.streams)
            specimen = appearance(images["left"], lcam, left, t)
            paired = RaySample.measure(t, "right", rcam, right, 1.)
            if specimen is not None:
                label = triangulated_range(specimen.hard, paired)
                if label is not None and model.add(specimen, label):
                    agreed.append(dict(tick=tick, features=specimen.features.tolist(),
                                       prior_range_m=specimen.hard.range_m, label_m=label))
        played = tick + 1
        _, _, done, cap, _ = plant.exchange(command, np.zeros(5))
        if done or cap:
            break
    return model.freeze(), agreed, played


def one(plant, rig, model, scale, shot):
    plant.launch(shot)
    plant.streams = ("left",)
    practice = FlightPractice()
    practice.radius_samples["left"].extend([scale] * 9)
    estimates = {"existing": OpticalFlight(rig), "one_scale": OpticalFlight(rig, practice), "learned": MonoFlight(rig, model)}
    rows = []
    for tick in range(120):
        t = tick * plant.control_dt
        images = plant.render_cameras()
        point = plant.ball.copy()
        speed = plant.velocity.copy()
        record = dict(tick=tick, true_position_m=point.tolist(), true_velocity_m_s=speed.tolist(),
                      window=bool(-1.28 < point[1] < .25 and speed[1] < -.6 and point[2] > .80))
        detection = ball_pixels(images["left"])
        if detection is not None:
            camera = rig.fixed["left"]
            sample = appearance(images["left"], camera, detection, t)
            if sample is not None and sample.hard.depth_weight:
                truth = np.linalg.norm(point - camera.eye)
                radii = sample.hard.range_m
                record["range_error"] = dict(existing=abs(radii - truth), one_scale=abs(radii / scale - truth),
                                             learned=abs(model.range(sample) - truth))
        for name, filter_ in estimates.items():
            prediction = filter_.observe(t, images, plant.joints)
            if prediction is not None and t - prediction.time < .05:
                where, dt = prediction.advance(t)
                record[name] = dict(position_error_m=float(np.linalg.norm(where - point)),
                                    velocity_error_m_s=float(np.linalg.norm(dt - speed)),
                                    xyz=where.tolist(), velocity=dt.tolist())
        rows.append(record)
        _, _, done, cap, _ = plant.exchange(np.zeros(plant.action_size), np.zeros(5))
        if done or cap:
            break
    return dict(shot=shot, rows=rows)


def median(values):
    return float(np.median(values)) if values else None


def summarize(cases):
    rows = [r for case in cases for r in case["rows"]]
    observed = [r for r in rows if r["window"]]
    modes = ("existing", "one_scale", "learned")
    shared = [r for r in observed if all(x in r for x in modes)]
    return dict(frames=len(rows), receiving_window_frames=len(observed), shared_window_frames=len(shared),
                modes={name: dict(raw_range_median_error_m=median([r["range_error"][name] for r in rows if "range_error" in r]),
                                   fresh_window_estimates=sum(name in r for r in observed),
                                   window_position_median_error_m=median([r[name]["position_error_m"] for r in shared]),
                                   window_velocity_median_error_m_s=median([r[name]["velocity_error_m_s"] for r in shared]))
                       for name in modes})


def run(path, output, *, reuse=False):
    start = process_time()
    plant = arena()
    shots = measured_launches(path)
    try:
        rig = Rig.from_arena(plant)
        if reuse:
            previous = _read(output)
            if _hash(path) != previous["feeds_sha256"] or _hash(MEMORY) != previous["memory_sha256"]:
                raise ValueError("The existing stereo fit used different recorded inputs")
            model = RangeFit.restore(previous["fit"])
            sources, count = previous["calibration_stereo_labels"], previous["calibration_ticks"]
            fit = previous["fit"]
        else:
            model, sources, count = prior_course(plant, rig)
            fit = model.receipt()
        scale = float(np.median([row["prior_range_m"] / row["label_m"] for row in sources]))
        print("calibration", len(sources), scale, model.sigma, "frames", count, flush=True)
        cases = []
        for sid in HELD:
            case = one(plant, rig, model, scale, shots[sid])
            cases.append(case)
            print(sid, summarize([case]), flush=True)
    finally:
        plant.close()
    summary = summarize(cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, dict(schema="tt29-left-monocular-range-1", feeds_sha256=_hash(path), memory_sha256=_hash(MEMORY),
                        calibration_opening=TRAIN_ID, calibration_ticks=count, calibration_stereo_labels=sources,
                        fit=fit, global_radius_factor=scale, assessment_ids=HELD, rows=cases,
                        summary=summary, cpu_s=process_time() - start))
    print("held", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--reuse-fit", action="store_true")
    args = parser.parse_args()
    run(args.feeds, args.output, reuse=args.reuse_fit)
