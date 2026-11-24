"""Align the direct pupil's training observations with its deployed visual estimator."""

import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt37_joint_capture
from .tt4_transfer_trial import _hash, _read, _write
from .tt33_joint import Vision
from .tt37_joint_deep import collect
from .tt37_visual_inertia import LearnedOptics


def aligned_eye(checkpoint):
    def build(*args, **kwargs):
        eye = Vision(*args, **kwargs)
        eye.eye = LearnedOptics(eye.eye, checkpoint)
        return eye

    return build


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("sources", nargs="+", type=Path)
    a = p.parse_args()
    with patch.object(tt37_joint_capture, "Vision", aligned_eye(a.checkpoint)):
        collect(a.sources, a.out)
    metadata = a.out.with_suffix(".json")
    _write(
        metadata,
        _read(metadata)
        | {"visual_sha256": _hash(a.checkpoint), "visual": str(a.checkpoint)},
    )
