"""Return to the familiar cooperative opponent with the same point seed and motor limits."""

import argparse
from contextlib import nullcontext
from pathlib import Path

from .tt4_transfer_trial import _read
from .tt29_action_memory import run
from .tt31_curriculum import ERRORS, MOVEMENTS
from .tt31_ready import centered


def retain(feeds, folder, learner):
    checkpoint = (
        folder / "ready-staged-checkpoint.json.gz"
        if learner in {"trained", "restored"}
        else MOVEMENTS
    )
    optical = checkpoint if learner in {"trained", "restored"} else ERRORS
    pose = (
        centered(_read(folder / "left-ready.json")["ready"])
        if learner in {"posture", "trained"}
        else nullcontext()
    )
    with pose:
        run(
            feeds,
            folder / f"cooperative-{learner}.json.gz",
            [7271],
            "left+right",
            "lateral",
            checkpoint=checkpoint,
            optical_checkpoint=optical,
            optical_outcomes="learn",
            freeze_optical_outcomes=True,
            censored_optical_outcomes=True,
            no_memory=True,
            seed_offset=41157,
            return_target=24,
            rated=True,
            motor_envelope=True,
            robust_contact=True,
            rally_objective=True,
            recent=True,
            blend=True,
            fast_partner=True,
            cached_stereo=True,
            camera_period=2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("learner", choices=("original", "posture", "trained", "restored"))
    a = p.parse_args()
    retain(a.feeds, a.folder, a.learner)
