"""Use the arm's estimated current visual position to encode the observed ball relative to the body."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt34_latency import OUT, PUPIL, Map
from .tt34_translate import LearnedBody

BASE = joint.features
BANK = OUT / "lag-bank.npz"


@contextmanager
def body_coordinates(mode):
    with np.load(BANK) as source:
        seen = LearnedBody(source)
    points = seen.p
    if mode == "shuffled":
        points = points[np.random.default_rng(351127).permutation(len(points))]
    position = Map(seen.q, points, seen.scale)

    def observed(x, option, q=None, home=None):
        raw = BASE(x, option, q, home)
        if option != "raw-proprioception":
            return raw
        # Refer coordinates to the body's displacement from its observed home position.
        poses = np.reshape(q[..., :5], (-1, 5))
        motion = np.asarray([position(pose) for pose in poses]) - position(home[:5])
        relative = raw.copy()
        relative[..., :3] -= motion.reshape(q.shape[:-1] + (3,)) / (0.10, 0.25, 0.12)
        return relative

    with patch.object(joint, "features", observed):
        yield


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("learned", "shuffled", "play"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--feeds", type=Path)
    parser.add_argument("--course", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "play":
        model = _read(args.checkpoint)
        if model["body_bank_sha256"] != _hash(BANK):
            raise ValueError("The observation bank differs from the fitted model")
        with body_coordinates(model["coordinates"]):
            joint.evaluate(
                args.feeds,
                args.course,
                args.checkpoint,
                args.output,
                cap=16,
                limit=None,
                trace=False,
            )
    else:
        with body_coordinates(args.mode):
            joint.fit(args.raw, args.checkpoint, _read(PUPIL)["options"].copy())
        model = _read(args.checkpoint)
        _write(
            args.checkpoint,
            model | {"coordinates": args.mode, "body_bank_sha256": _hash(BANK)},
        )
