"""Evaluate a frozen four-point optical landing calibration on the common rally starts."""

from pathlib import Path
import argparse

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_rally import ReturnMemory
from .tt22_rally_trial import CONFIG, OUTPUT, Station, replay

SEED = 220381117


def evaluate(checkpoint, output):
    memory = ReturnMemory(_read(checkpoint)["evidence"])
    station = Station(0.30, 0.82, "anchor")
    data = dict(
        schema="tt22-frozen-optical-memory-v1",
        seed=SEED,
        spread=0.30,
        pace=0.82,
        checkpoint=_hash(checkpoint),
        rival="anchor",
        privileged=False,
        arena=CONFIG,
        camera="native stereo",
        rows=[],
    )
    try:
        for seed in np.random.default_rng(SEED).integers(2**31, size=12):
            row = station.point(int(seed), memory=memory, update_memory=False)
            data["rows"].append(row)
            print(seed, row["longest"], flush=True)
            _write(output, data)
    finally:
        station.close()
    data["verified"] = replay(data["rows"], 0.30, 0.82, "anchor")
    _write(output, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT / "rally-learning-checkpoint.json.gz")
    parser.add_argument("--output", type=Path, default=OUTPUT / "benchmark-learned.json.gz")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output)
