"""Evaluate the optical readout with the frozen camera-to-joint field on identical cards."""

import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt37_joint_deep as pupil
from .tt33_joint import Vision
from .tt37_visual_inertia import LearnedOptics


def evaluate(feeds, archive, plan, deck, out, checkpoint):
    def build(*args, **kwargs):
        eye = Vision(*args, **kwargs)
        eye.eye = LearnedOptics(eye.eye, checkpoint)
        return eye

    with patch.object(pupil, "Vision", build):
        pupil.run(
            feeds,
            archive,
            plan,
            deck,
            "local",
            out,
            cap=200,
            details={"optics": str(checkpoint)},
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deck")
    p.add_argument("out", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--archive",
        type=Path,
        default=Path("artifacts/tt37/joint-deep-observations.npz"),
    )
    p.add_argument(
        "--plan", type=Path, default=Path("artifacts/tt37/opponent-plan.json")
    )
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    evaluate(a.feeds, a.archive, a.plan, a.deck, a.out, a.checkpoint)
