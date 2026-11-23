"""Use a tool attitude estimated from solo balls to transport existing motor goals between grips."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .tt4_transfer_trial import _read, _write
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_public_body import PublicDecoder
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import OUT, PRIOR, Grip, ReferenceReceiver, load_match
from .tt26_receiving_speed import draw

MOUNTS = {
    "nominal": (0.0, (0, 0, 1)),
    "yaw_plus": (0.22, (0, 0, 1)),
    "yaw_minus": (-0.22, (0, 0, 1)),
    "pitch_plus": (0.26, (1, 0, 0)),
}


def decoder():
    optics = CarriedWristPolicy().calibration[0]
    return PublicDecoder(*(optics[k] for k in ("offset", "normal", "registration", "default")))


def transport(rig, goal, rotation, bounds, *, position_scale=0.008):
    nominal = rig.normal
    changed = rotation @ nominal

    def frame(q, n):
        p, r = rig.links.read(q[None, :])
        return p[0] + r[0] @ rig.offset[3:], r[0] @ n

    p, n = frame(goal, nominal)

    def residual(candidate):
        center, normal = frame(candidate, changed)
        return np.r_[(center - p) / position_scale, (normal - n) / 0.045, (candidate - goal) / 0.70]

    lo, hi = bounds
    fit = least_squares(
        residual, goal, bounds=(np.maximum(lo, goal - 0.70), np.minimum(hi, goal + 0.70)), max_nfev=50
    )
    return fit.x


def run(path):
    shots, (keys, goals, scale) = load_match(path)
    donors = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    recorded = _read(PRIOR / "recorded-progressive.json.gz")["tests"]
    source = _read(OUT / "optical-face.json.gz")["mounts"]
    rig = decoder()
    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    axes = arena._arm_ids[1][2][:5]
    bounds = arena._joint_low[axes], arena._joint_high[axes]
    records = []
    try:
        for name, (angle, axis) in MOUNTS.items():
            if name not in source:
                raise ValueError(f"Incomplete optical source: {name}")
            grip.set(angle, axis)
            rot = np.asarray(source[name]["rotation_from_nominal"])
            if not np.allclose(rot.T @ rot, np.eye(3), atol=1.0e-6) or np.linalg.det(rot) < 0.99:
                raise ValueError("Invalid learned optical rotation")
            banks = {
                "cold": goals,
                "optical": np.stack([transport(rig, g, rot, bounds) for g in goals]),
                "reverse": np.stack([transport(rig, g, rot.T, bounds) for g in goals]),
            }
            rng = np.random.default_rng(4923711)
            cases = [("beginner", draw(shots, donors, rng, i)) for i in range(48)]
            cases += [
                (r["group"], shots[r["id"]]) for r in recorded if r["group"] in ("nearby", "unfiltered")
            ]
            rows = []
            for group, shot in cases:
                blank, key = episode(arena, shot, collect=True)
                scores = {}
                for mode, bank in banks.items():
                    if key is None:
                        score = blank
                    else:
                        i = np.argmin(np.linalg.norm((keys - key) / scale, axis=1))
                        score = episode(arena, shot, bank[i])[0]
                    scores[mode] = score
                rows.append(dict(group=group, id=shot["id"], results=scores))
            records.append(dict(mount=name, goals={k: v.tolist() for k, v in banks.items()}, rows=rows))
            _write(OUT / "optical-transfer.json.gz", dict(records=records))
            print(
                name,
                {
                    group: {
                        k: sum(r["results"][k]["legal"] for r in rows if r["group"] == group) for k in banks
                    }
                    for group in ("beginner", "nearby", "unfiltered")
                },
                flush=True,
            )
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
