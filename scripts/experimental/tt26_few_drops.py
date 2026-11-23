"""Learn a tool adjustment chronologically from unpaired solo drops before any target shot."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_bounce_optics import free_acceleration, impact_candidates, rotation_between, unit
from .tt26_grip_course import Grip, OUT, PRIOR, ReferenceReceiver, load_match
from .tt26_optical_transfer import MOUNTS, decoder, transport
from .tt26_receiving_speed import draw


def public_track(record):
    return {name: np.asarray(record[name], dtype=float) for name in ("ball", "joint_angles")}


def directions(track, gravity, rig):
    return [
        np.asarray(events[0]["raw_normal_link"]) if events else None
        for events in (impact_candidates(row, gravity, rig) for row in track)
    ]


def run(path):
    shots, (keys, goals, scale) = load_match(path)
    base = _read(OUT / "optical-face.json.gz")["mounts"]
    donors = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    rig = decoder()
    axes = arena._arm_ids[1][2][:5]
    bounds = arena._joint_low[axes], arena._joint_high[axes]
    records = []
    try:
        for fold in (0, 1):
            nominal = base["nominal"]["drops"][fold::2]
            renderer = [public_track(row) for row in nominal]
            gravity, _ = free_acceleration(renderer)
            estimates = [v for v in directions(renderer, gravity, rig) if v is not None]
            reference = unit(np.median(estimates, axis=0))
            for name in ("yaw_plus", "yaw_minus", "pitch_plus"):
                changed = base[name]["drops"][1 - fold :: 2]
                if {r["seed"] for r in nominal} & {r["seed"] for r in changed}:
                    raise ValueError("Nominal and changed releases must be independent")
                trajectory = [public_track(r) for r in changed]
                measured = directions(trajectory, gravity, rig)
                banks = {}
                rotations = {}
                for budget in (0, 1, 4, 8):
                    usable = [v for v in measured[:budget] if v is not None]
                    rot = (
                        rotation_between(reference, unit(np.median(usable, axis=0)))
                        if usable
                        else np.eye(3)
                    )
                    rotations[str(budget)] = dict(matrix=rot.tolist(), events=len(usable))
                    banks[str(budget)] = (
                        np.stack([transport(rig, g, rot, bounds) for g in goals]) if usable else goals
                    )
                grip.set(*MOUNTS[name])
                rng = np.random.default_rng(3112945)
                rows = []
                for tick in range(32):
                    shot = draw(shots, donors, rng, tick)
                    _, key = episode(arena, shot, collect=True)
                    index = int(np.argmin(np.linalg.norm((keys - key) / scale, axis=1)))
                    results = {k: episode(arena, shot, bank[index])[0] for k, bank in banks.items()}
                    rows.append(dict(i=tick, donor=shot["id"], results=results))
                records.append(
                    dict(
                        fold=fold,
                        mount=name,
                        reference_seeds=[r["seed"] for r in nominal],
                        changed_seeds=[r["seed"] for r in changed],
                        fits=rotations,
                        rows=rows,
                    )
                )
                _write(OUT / "few-drops.json.gz", dict(records=records))
                print(
                    fold, name, {k: sum(r["results"][k]["legal"] for r in rows) for k in banks}, flush=True
                )
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
