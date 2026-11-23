"""A focused practice course of distinct real shots near the six demonstrations."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def features(row):
    return np.array([row[f"{prefix}_{a}"] for prefix in ("pos","vel","w_vel") for a in "xyz"])


def run(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != json.loads((OUTPUT/"recorded-secondary-reference.json").read_text())["sha256"]:
        raise ValueError("Unexpected shot file")
    shots = {r["id"]:r for r in json.loads(raw)}
    lesson = _read(OUTPUT/"recorded-pose-imitation.json.gz")
    donors = [r["id"] for r in lesson["teachers"] if any(x["legal"] for x in r["trials"])]
    with np.load(OUTPUT/"recorded-pose-imitation.npz",allow_pickle=False) as file:
        keys, goals, scales = (file[name] for name in ("image_keys","joint_goals","scale"))
    if len(donors) != len(keys):
        raise ValueError("Inconsistent demonstrated set")
    used = {r["id"] for r in json.loads((OUTPUT/"recorded-reach.json").read_text())["cases"]}
    candidates = [s for s in shots.values() if s["id"] not in used and s["pos_y"]>0 and s["vel_y"]<0]
    vectors = np.stack([features(r) for r in candidates])
    # Course construction only. The camera policy does not see these values,
    # donor assignment, or the launch ID. Position, velocity and spin each count.
    metric = np.array([.22,.30,.15,1.,1.5,.75,45.,45.,45.])
    ordered = []
    for donor in donors:
        target = features(shots[donor])
        distance = np.linalg.norm((vectors-target)/metric,axis=1)
        slots = 0
        for j in np.argsort(distance):
            row = candidates[j]
            if row["id"] in used or np.allclose(target,vectors[j],atol=1e-5,rtol=0):
                continue
            used.add(row["id"])
            ordered.append(dict(id=row["id"],donor=donor,rank=slots+1,
                                start_displacement_m=float(np.linalg.norm(vectors[j,:3]-target[:3])),
                                delta_velocity_mps=float(np.linalg.norm(vectors[j,3:6]-target[3:6]))))
            slots += 1
            if slots == 8:
                break
    arena=ReferenceReceiver()
    memory=(keys,goals,scales)
    shuffled=(keys,np.roll(goals,1,axis=0),scales)
    try:
        for case in ordered:
            shot=shots[case["id"]]
            conditions={}
            for name,mem in [("camera",memory),("wrong_assignment",shuffled),("hold",None)]:
                stats,_=episode(arena,shot,memory=mem)
                conditions[name]=stats
                if stats["legal"]:
                    repeated,_=episode(arena,shot,memory=mem)
                    if repeated!=stats:
                        raise ValueError("Learned return changed on replay")
            case["results"]=conditions
        _write(OUTPUT/"recorded-nearby-lessons.json.gz",dict(cases=ordered, donors=donors, metric=metric.tolist()))
        print({name:sum(c["results"][name]["legal"] for c in ordered) for name in conditions})
    finally:
        arena.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind",type=Path)
    run(p.parse_args().deepmind)
