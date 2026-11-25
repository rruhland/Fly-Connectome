"""Compare the same learned early motor response with its time-lag inputs removed."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt43_temporal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write

OUT = servo.ROOT / "instant.npz"


def fit():
    x, y, groups = servo.packets(servo.SOURCES)
    trained = servo.Readout(x, y, False)
    np.savez_compressed(OUT, **trained.__dict__)
    _write(
        OUT.with_suffix(".json"),
        {
            "inputs": {str(p): _hash(p) for p in servo.SOURCES},
            "model": _hash(OUT),
            "frames": len(x),
            "releases": len(np.unique(groups)),
        },
    )


def run(a):
    read = servo.reload
    with patch.object(servo, "reload", lambda path: read(OUT)):
        servo.run(a)
    d = _read(a.out)
    d["temporal_sha256"] = _hash(OUT)
    d["treatment"] = "same feedback fit using only current ball and encoder observation"
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "play"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--n", type=int, default=6)
    p.add_argument(
        "--carry", type=Path, default=servo.course.ROOT / "learning-12.json.gz"
    )
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(mode="frozen", deck="assessment", cap=240, resume=False)
    a = p.parse_args()
    fit() if a.phase == "fit" else run(a)
