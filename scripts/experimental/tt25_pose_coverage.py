"""How often could any of the demonstrated postures return a new incoming ball?"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def run(path):
    raw = path.read_bytes()
    if (
        sha256(raw).hexdigest()
        != json.loads((OUTPUT / "recorded-secondary-reference.json").read_text())["sha256"]
    ):
        raise ValueError("Unexpected DeepMind input")
    shots = {r["id"]: r for r in json.loads(raw)}
    original = _read(OUTPUT / "recorded-pose-imitation.json.gz")
    with np.load(OUTPUT / "recorded-pose-imitation.npz", allow_pickle=False) as file:
        goals, keys = file["joint_goals"], file["image_keys"]
    arena = ReferenceReceiver()
    try:
        cases = []
        for sid in original["held"]:
            row = [episode(arena, shots[sid], goal)[0] for goal in goals]
            cases.append(dict(id=sid, contact=[s["contact"] for s in row], legal=[s["legal"] for s in row]))
        report = dict(
            cases=cases,
            demonstrated=len(goals),
            unique_early_rgb_features=len(np.unique(keys, axis=0)),
            any_legal=sum(any(r["legal"]) for r in cases),
            any_contact=sum(any(r["contact"]) for r in cases),
        )
        _write(OUTPUT / "recorded-pose-coverage.json.gz", report)
        print({k: v for k, v in report.items() if k != "cases"})
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    run(p.parse_args().deepmind)
