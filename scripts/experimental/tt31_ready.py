"""Calibrate a collision-screened ready posture for a known practice region."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_arm_clearance import ArmClearance
from .tt29_rally_physics import DEEP_STRIKE_READY, BladeIK, RallyArena
from .tt31_curriculum import ERRORS, MOVEMENTS, play_card


@contextmanager
def centered(ready):
    original = RallyArena._arm_home

    def home(self, side):
        if (
            self.fixture == "table-edge"
            and self.strike_ready == "lateral"
            and side == 1
        ):
            return ready, ()
        return original(self, side)

    with patch.object(RallyArena, "_arm_home", home):
        yield


def body_ready(folder, region):
    """Choose a common pose close in joint space to playable poses on both sides."""
    a = RallyArena(strike_ready="lateral", detail="cad")
    try:
        ik = BladeIK(a)
        guard = ArmClearance(a.model)
        joints = qmc.scale(
            qmc.Sobol(5, scramble=True, seed=311298).random_base2(16), ik.low, ik.high
        )
        descriptors = np.array([np.r_[*ik.pose(q)] for q in joints])
        valid = {}

        def clear(index):
            if index not in valid:
                q = tuple(joints[index]) + (-0.10,)
                with centered(q):
                    a.reset(render=False, server=-1)
                valid[index] = (
                    min(
                        (d[2] for d in guard.distances(a.data, horizon=0.025)),
                        default=0.025,
                    )
                    >= 0.008
                )
            return valid[index]

        def neighbors(target, count):
            distance = np.linalg.norm(
                descriptors[:, :3] - [target, -1.29, 1.045], axis=1
            )
            distance += 0.08 * np.linalg.norm(
                descriptors[:, 3:] - [0.0, 0.97, 0.24], axis=1
            )
            close = []
            for j in np.argsort(distance, kind="stable"):
                if clear(j):
                    close.append(j)
                    if len(close) == count:
                        return joints[close]
            raise ValueError("No clear body postures")

        sides = [neighbors(x, 20) for x in [-0.22, -0.10, 0.10, 0.22]]
        middle = neighbors(0.0 if region == "centered" else -0.19, 48)
        # Lower bound on joint travel at a common target-rate limit.
        travel = np.stack(
            [
                np.min(
                    np.max(abs(middle[:, None, :] - poses[None, :, :]), axis=2), axis=1
                )
                for poses in sides
            ]
        )
        costs = travel if region == "centered" else travel[:2]
        selected = int(np.argmin(costs.max(axis=0) + 0.25 * costs.mean(axis=0)))
        ready = tuple(middle[selected]) + (-0.10,)
        with centered(DEEP_STRIKE_READY):
            a.reset(render=False, server=-1)
        prior_gap = min(guard.distances(a.data, horizon=0.035), key=lambda d: d[2])
        _write(
            folder / f"{region}-ready.json",
            {
                "ready": ready,
                "selected_rad": travel[:, selected].tolist(),
                "target_x_m": [-0.22, -0.10, 0.10, 0.22],
                "unsafe_legacy_pose_gap": prior_gap,
            },
        )
        return ready
    finally:
        a.close()


def run(folder, feeds, region):
    deck = _read(folder / "course-plan.json.gz")
    # The initial two periodic cards and one additional novel delivery on each side.
    cards = [c for c in deck["staged"] if c["mode"] == "periodic"][:2]
    for side in ("left", "right"):
        cards.append(
            next(
                d["card"]
                for d in deck["neutral_practice"]
                if d["card"]["mode"] == "continuous"
                and d["first_reply"]["observed_lane"] == side
            )
        )
    ready = body_ready(folder, region)
    with centered(ready):
        cad = RallyArena(strike_ready="lateral", detail="cad")
        try:
            cad.launch(_read(feeds)[0])
            nearest = min(
                ArmClearance(cad.model).distances(cad.data, horizon=0.03),
                key=lambda d: d[2],
            )
            if nearest[2] <= 0:
                raise ValueError(f"Initial centered stance intersects scene: {nearest}")
        finally:
            cad.close()
        result = {
            "ready": list(ready),
            "nearest_initial_pair": nearest,
            "cards": cards,
            "opening_side": 1,
            "movement_sha256": _hash(MOVEMENTS),
            "optical_sha256": _hash(ERRORS),
            "rows": [],
        }
        with TemporaryDirectory() as d:
            for card in cards:
                row, _ = play_card(
                    feeds,
                    card,
                    MOVEMENTS,
                    ERRORS,
                    Path(d) / "temp.json.gz",
                    practice=False,
                    cap=2,
                )
                result["rows"].append(row)
                _write(folder / f"{region}-calibration.json.gz", result)
                print(card["key"], row["score"]["learner"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("feeds", type=Path)
    p.add_argument("--region", choices=("centered", "left"), default="centered")
    a = p.parse_args()
    run(a.folder, a.feeds, a.region)
