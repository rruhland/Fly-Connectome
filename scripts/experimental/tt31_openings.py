"""Independent receiving test using measured ball releases stratified by lateral position."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_action_memory import run
from .tt29_partner_fast import NativeBallFlight
from .tt29_rally_physics import RallyArena, measured_launches

# Common openings in the preceding wide and cooperative comparisons.
EXCLUDE = {
    7271,
    7345,
    11292,
    4187,
    10542,
    4230,
    4527,
    15489,
    7331,
    10712,
    3152,
    13410,
    5736,
    11650,
    6739,
    10881,
}


def select(feeds, output):
    shots = measured_launches(feeds)
    ids = sorted(
        i
        for i, s in shots.items()
        if i not in EXCLUDE and s["pos_y"] > 0 and s["vel_y"] < 0
    )
    rng = np.random.default_rng(312809)
    lateral_centers = np.array([-0.28, -0.21, -0.14, -0.07, 0.07, 0.14, 0.21, 0.28])
    bins = [[] for _ in lateral_centers]
    arena = RallyArena(strike_ready="lateral")
    try:
        forecast = NativeBallFlight(arena)
        for i in rng.permutation(ids):
            s = shots[int(i)]
            p, v, w = (
                np.array([s[prefix + dim] for dim in "xyz"])
                for prefix in ("pos_", "vel_", "w_vel_")
            )
            p[2] += TABLE_HEIGHT
            flight = forecast.path(p, v, w, 1, False, horizon=1.7)
            plane = -1.29
            if len(flight) < 2 or not flight[-1, 2] < plane < flight[0, 2]:
                continue
            sample = np.array(
                [np.interp(plane, flight[::-1, 2], c[::-1]) for c in flight.T]
            )
            t, x, _, z, vx, vy, vz = sample[:7]
            slot = int(np.argmin(abs(x - lateral_centers)))
            if abs(x - lateral_centers[slot]) > 0.027 or not (
                0.96 < z < 1.14 and -6.0 < vy < -2.15
            ):
                continue
            if len(bins[slot]) < 2:
                bins[slot].append(
                    {
                        "id": int(i),
                        "x_m": x,
                        "z_m": z,
                        "vx": vx,
                        "vy": vy,
                        "vz": vz,
                        "arrival_s": t,
                    }
                )
            if all(len(group) == 2 for group in bins):
                break
    finally:
        arena.close()
    if not all(len(group) == 2 for group in bins):
        raise ValueError(f"Selection incomplete: {[len(g) for g in bins]}")
    # Interleave the two sides in each test round. All learners get the same releases.
    rows = [bins[i][j] for j in range(2) for k in range(4) for i in (3 - k, 4 + k)]
    _write(
        output,
        {
            "feed_sha256": _hash(feeds),
            "seed": 312809,
            "plane_y_m": plane,
            "lateral_centers_m": lateral_centers.tolist(),
            "rows": rows,
        },
    )


def evaluate(feeds, manifest, checkpoint, output, optical=None):
    data = _read(manifest)
    if _hash(feeds) != data["feed_sha256"]:
        raise ValueError("Feed data do not match the selected openings")
    run(
        feeds,
        output,
        [r["id"] for r in data["rows"]],
        "left+right",
        "lateral",
        checkpoint=checkpoint,
        return_target=1,
        no_memory=True,
        optical_checkpoint=optical,
        optical_outcomes="learn" if optical else None,
        freeze_optical_outcomes=bool(optical),
        censored_optical_outcomes=bool(optical),
        camera_period=2,
        rated=True,
        motor_envelope=True,
        robust_contact=True,
        rally_objective=True,
        recent=True,
        blend=True,
        fast_partner=True,
        net_rescue=True,
        cached_stereo=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--optical",
        type=Path,
        help="Frozen placement memory paired with the movement checkpoint",
    )
    p.add_argument(
        "--ready",
        type=Path,
        help="Use one fixed calibrated starting posture throughout the exam",
    )
    args = p.parse_args()
    if bool(args.checkpoint) != bool(args.output):
        p.error("Provide both checkpoint and output for evaluation")
    if args.checkpoint:
        from .tt31_ready import centered

        with centered(_read(args.ready)["ready"]) if args.ready else nullcontext():
            evaluate(
                args.feeds, args.manifest, args.checkpoint, args.output, args.optical
            )
        if args.ready:
            result = _read(args.output)
            result["ready_sha256"] = _hash(args.ready)
            result["ready"] = _read(args.ready)["ready"]
            _write(args.output, result)
    else:
        select(args.feeds, args.manifest)
