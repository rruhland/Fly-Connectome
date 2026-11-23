"""Transfer a camera-estimated arm displacement between regulation-table mounting positions."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .tt4_camera import triangulate
from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_optical_transfer import decoder
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT, SOURCE, Eyes, Player, load_bank

GAINS = (0.0, 0.5, 1.0, 1.5)
PRACTICE = (-0.105, 0.075, -0.055)
TARGET = (0.12, -0.085, 0.045, -0.13, 0.0)


def paddle_origin(arena):
    frames = arena.render_cameras()
    views = Eyes(arena).cameras
    centers = []
    for k in ("left", "right"):
        r, g, b = frames[k].astype(float).transpose(2, 0, 1)
        mask = (r > 65) & (r > 2.1 * g + 12) & (r > 2 * b + 12)
        mask[: mask.shape[0] // 2] = False
        y, x = np.nonzero(mask)
        if len(x) < 12:
            raise ValueError("Paddle silhouette could not be located")
        centers.append([np.mean(x), np.mean(y)])
    xyz = triangulate(views, *centers)
    if xyz is None:
        raise ValueError("Paddle stereo estimates did not intersect")
    return np.array(xyz)


class Receiver(Player):
    def __init__(self, arena):
        super().__init__()
        self.rig = decoder()
        axes = arena._arm_ids[1][2][:5]
        self.bounds = (arena._joint_low[axes], arena._joint_high[axes])

    @lru_cache(maxsize=2048)
    def target(self, index, dx):
        goal = self.templates[index].goal
        if dx == 0:
            return goal

        def pose(q):
            p, rot = self.rig.links.read(q[None, :])
            return p[0] + rot[0] @ self.rig.offset[3:], rot[0] @ self.rig.normal

        p, n = pose(goal)
        desired = p - np.array([dx, 0.0, 0.0])

        def error(q):
            new, normal = pose(q)
            return np.r_[(new - desired) / 0.01, (normal - n) / 0.05, (q - goal) / 0.70]

        lo, hi = self.bounds
        return least_squares(
            error, goal, bounds=(np.maximum(lo, goal - 0.9), np.minimum(hi, goal + 0.9)), max_nfev=35
        ).x


def run(path):
    lookup, _ = load_match(path)
    arena = ReferenceReceiver()
    player = Receiver(arena)
    donors = [r["id"] for r in load_bank()["rows"]]
    raw = [r["id"] for r in _read(SOURCE / "fresh-recordings.json.gz")["records"][0]["rows"]]
    rows = []
    try:
        arena.reset(seed=501, render=False)
        nominal = paddle_origin(arena)
        for i, shift in enumerate((*PRACTICE, *TARGET)):
            for part in ("learner_base", "tt20_table_end_clamp"):
                arena.model.body(part).pos[0] = shift
            arena.reset(seed=501, render=False)
            measured = float(paddle_origin(arena)[0] - nominal[0])
            split = "practice" if i < len(PRACTICE) else "held"
            rng = np.random.default_rng(716989 + i)
            shots = [("nearby", draw(lookup, donors, rng, t)) for t in range(28)]
            if split == "held":
                shots.extend(("recorded", lookup[j]) for j in rng.choice(raw, 24, replace=False))
            results = []
            for group, shot in shots:
                blank, k = episode(arena, shot, collect=True)
                idx = player.choose(k) if k is not None else None
                trials = []
                for gain in GAINS:
                    r = (
                        blank
                        if idx is None
                        else episode(arena, shot, goal=player.target(idx, gain * measured))[0]
                    )
                    trials.append(r)
                results.append(dict(group=group, shot=shot["id"], origin=idx, trials=trials))
            rows.append(dict(split=split, dx=shift, measured=measured, rows=results))
            print(
                split,
                shift,
                round(measured, 4),
                {
                    g: [sum(r["trials"][a]["legal"] for r in results if r["group"] == g) for a in range(4)]
                    for g in ("nearby", "recorded")
                },
                flush=True,
            )
            _write(OUT / "mount-position.json.gz", dict(gains=GAINS, blocks=rows))
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
