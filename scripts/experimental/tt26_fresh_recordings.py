"""Confirm tool transfer on recorded feeds outside the preceding TT25/26 test rosters."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_nearby_lessons import features
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import OUT, PRIOR, Grip, ReferenceReceiver, load_match
from .tt26_optical_transfer import MOUNTS, decoder, transport


def course(shots):
    base = _read(PRIOR / "recorded-nearby-lessons.json.gz")
    exclusion = {s["id"] for s in json.loads((PRIOR / "recorded-reach.json").read_text())["cases"]}
    for group in _read(PRIOR / "recorded-block-confirmation.json.gz")["cohorts"]:
        exclusion.update(s["id"] for s in group)
    exclusion.update(s["id"] for s in _read(PRIOR / "recorded-progressive.json.gz")["tests"])
    pool = [s for s in shots.values() if s["id"] not in exclusion and s["pos_y"] > 0 and s["vel_y"] < 0]
    field = np.vstack([features(s) for s in pool])
    rng = np.random.default_rng(764983)
    nearby = []
    for sid in base["donors"]:
        dist = np.linalg.norm((field - features(shots[sid])) / base["metric"], axis=1)
        eligible = [
            pool[j]["id"] for j in np.argsort(dist) if dist[j] > 1.0e-5 and pool[j]["id"] not in exclusion
        ][:40]
        ids = rng.choice(eligible, 12, replace=False).tolist()
        exclusion.update(ids)
        nearby.extend(ids)
    broad = rng.choice([s["id"] for s in pool if s["id"] not in exclusion], 64, replace=False).tolist()
    if len(set(nearby + broad)) != 136:
        raise ValueError("Recorded feed repeated")
    return [("nearby", sid) for sid in nearby] + [("unrestricted", sid) for sid in broad]


def run(path):
    shots, (keys, goals, scale) = load_match(path)
    cohort = course(shots)
    body = ReferenceReceiver()
    grip = Grip(body.model)
    axes = body._arm_ids[1][2][:5]
    bounds = body._joint_low[axes], body._joint_high[axes]
    optics = _read(OUT / "optical-face.json.gz")["mounts"]
    rig = decoder()
    records = []
    try:
        for name in ("yaw_minus", "yaw_plus", "pitch_plus"):
            grip.set(*MOUNTS[name])
            rotation = np.asarray(optics[name]["rotation_from_nominal"])
            banks = {
                "cold": goals,
                "optical": np.stack([transport(rig, g, rotation, bounds) for g in goals]),
            }
            if name.startswith("yaw"):
                adjustment = next(
                    r["selected"]
                    for r in _read(OUT / "grip-source.json.gz")["records"]
                    if r["angle"] == MOUNTS[name][0]
                )
                banks["reward"] = goals + np.array([0.0, 0.0, 0.0, 0.0, adjustment])
            rows = []
            for group, sid in cohort:
                blank, key = episode(body, shots[sid], collect=True)
                results = {}
                for label, bank in banks.items():
                    if key is None:
                        result = blank
                    else:
                        goal = bank[np.argmin(np.linalg.norm((keys - key) / scale, axis=1))]
                        result = episode(body, shots[sid], goal)[0]
                        if result["legal"] and episode(body, shots[sid], goal)[0] != result:
                            raise ValueError("New recorded return failed replay")
                    results[label] = result
                rows.append(dict(group=group, id=sid, results=results))
            records.append(dict(mount=name, rows=rows))
            _write(OUT / "fresh-recordings.json.gz", dict(seed=764983, records=records))
            print(
                name,
                {
                    g: {k: sum(r["results"][k]["legal"] for r in rows if r["group"] == g) for k in banks}
                    for g in ("nearby", "unrestricted")
                },
                flush=True,
            )
    finally:
        body.close()


def expanded_without_bounce(path):
    shots, _ = load_match(path)
    reference = _read(OUT / "match-broader.json.gz")
    played = next(c for c in reference["assessment"] if c["mount"] == "yaw_plus")
    with np.load(OUT / "match-broader.npz", allow_pickle=False) as archive:
        memory = [archive[k] for k in ("keys", "goals", "scale")]
    arena = ReferenceReceiver()
    Grip(arena.model).set(*MOUNTS["yaw_plus"])
    rows = []
    try:
        for row in played["rows"]:
            shot = shots[row["id"]]
            score, _ = episode(arena, shot, memory=memory)
            if score["legal"] and episode(arena, shot, memory=memory)[0] != score:
                raise ValueError("Expanded cold receiver failed replay")
            rows.append(
                dict(id=row["id"], group=row["group"], cold=score, optical=row["results"]["expanded"])
            )
        _write(OUT / "expanded-no-bounce.json.gz", dict(rows=rows))
        print({k: sum(x[k]["legal"] for x in rows) for k in ("cold", "optical")}, flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("--expanded-without-bounce", action="store_true")
    args = p.parse_args()
    if args.expanded_without_bounce:
        expanded_without_bounce(args.feeds)
    else:
        run(args.feeds)
