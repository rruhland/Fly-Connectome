"""Adapt which previous outgoing ball shapes to seek, using the current and following legal returns."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.cluster.vq import kmeans2

from . import tt56_scored_motor_atlas as atlas
from . import tt57_opponent_memory as experience
from .tt4_transfer_trial import _hash, _read, _write

mpc = atlas.mpc
ROOT = Path("artifacts/tt57_rally_strategy")
MODEL = ROOT / "shapes.npz"
SCALE = np.array([0.18, 0.24])


def fit():
    arc, _, _, _, rates, release = experience.assemble()
    # Prototypes describe diverse feasible effects of paid movements. Their
    # previous scores do not determine which prototype the actor will select.
    loc = arc[:, :2]
    centers, _ = kmeans2(loc / SCALE, 5, iter=25, minit="++", seed=570148)
    centers *= SCALE
    order = np.argsort(centers[:, 1])
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL, centers=centers[order])
    _write(
        ROOT / "fit.json",
        dict(
            centers=centers[order].tolist(),
            previously_scored_shapes=len(arc),
            releases=len(np.unique(release)),
            data=_hash(atlas.ROOT / "fit.json"),
            model=_hash(MODEL),
        ),
    )


class Strategy:
    def __init__(self, centers, seed):
        self.centers = centers
        self.rng = np.random.default_rng(seed)
        self.a = np.full((len(centers) + 1, 4), 4.8)
        self.b = np.full_like(self.a, 1.2)
        self.count = np.zeros_like(self.a)
        self.fresh()

    def fresh(self):
        self.current = None
        self.previous = None

    def choose(self, x):
        if self.current is None:
            side = np.clip(0.5 + x[0] / 0.45, 0, 1)
            lift = np.clip(0.5 + x[5] / 4, 0, 1)
            weights = np.array(
                [
                    (1 - side) * (1 - lift),
                    side * (1 - lift),
                    (1 - side) * lift,
                    side * lift,
                ]
            )
            shared_a = 0.3 * self.a.mean(1) + 0.7 * (self.a @ weights)
            shared_b = 0.3 * self.b.mean(1) + 0.7 * (self.b @ weights)
            mode = int(np.argmax(self.rng.beta(shared_a, shared_b)))
            self.current = (mode, weights)
        return self.current[0]

    def outcome(self, success):
        for item, credit in ((self.current, 0.4), (self.previous, 0.6)):
            if item is None:
                continue
            mode, w = item
            self.a[mode] += credit * w * success
            self.b[mode] += credit * w * (1 - success)
            self.count[mode] += credit * w
        self.previous = self.current if success else None
        self.current = None


class Player(atlas.season.Player):
    def __init__(self, *args, strategy, **kw):
        super().__init__(*args, **kw)
        self.strategy = strategy
        keep = self.priors.keep

        def earn(tape, tag):
            self.strategy.outcome(True)
            return keep(tape, tag)

        self.priors.keep = earn

    def begin(self):
        super().begin()
        self.strategy.fresh()
        self.target = None

    def missed(self, tape):
        self.strategy.outcome(False)

    def search(self, tick, x, q, ref, base):
        new = self.strategy.current is None
        mode = self.strategy.choose(x)
        self.target = self.strategy.centers[mode - 1] if mode else None
        self.diag[f"strategy_{mode}_offers"] += int(new)
        super().search(tick, x, q, ref, base)

    def trial(self, goals, *, long=False, traces=False):
        loss, legal, track, y, f = super().trial(goals, long=long, traces=traces)
        if self.target is None:
            return loss, legal, track, y, f
        dep = f[:, 1] > 0
        if not dep.any() and not legal.any():
            return loss, legal, track, y, f
        # Use the existing camera-initialized private dynamics and its nominal
        # response. A searched movement can still earn an ordinary legal return
        # anywhere on the table, including outside the preferred shape.
        v = f[:, 2:5]
        p = f[:, 5:8]
        elapsed = (
            v[:, 2] + np.sqrt(np.maximum(0, v[:, 2] ** 2 + 19.62 * (p[:, 2] - 0.779)))
        ) / 9.81
        land = p[:, :2] + 0.94 * v[:, :2] * np.maximum(0.01, elapsed[:, None])
        land[legal] = y[legal, 3:5]
        distance = np.linalg.norm((land - self.target) / SCALE, axis=1)
        # Bounded adjustment keeps outright misses outside the legal-action class.
        extra = np.where(dep | legal, 1.45 * np.minimum(2.5, distance), 0.0)
        return loss + extra, legal, track, y, f


def play(a):
    data = _read(ROOT / "fit.json")
    if _hash(MODEL) != data["model"]:
        raise ValueError("Changed initial action effects")
    with np.load(MODEL) as f:
        centers = f["centers"]
    strategy = Strategy(centers, 57012927)
    rank = atlas.season.fresh.shot.Coverage(_read(atlas.season.fresh.RISK))

    def make(*args, **kw):
        return Player(*args, strategy=strategy, coverage=rank, **kw)

    with (
        atlas.enabled(),
        patch.object(atlas.season.course, "PLAN", atlas.dose.PLAN),
        patch.object(mpc, "MPC", make),
    ):
        atlas.season.course.run(a)
    learned = ROOT / "adapted.npz"
    np.savez_compressed(
        learned, centers=centers, a=strategy.a, b=strategy.b, counts=strategy.count
    )
    d = _read(a.out)
    d.update(
        initial_effects=_hash(MODEL),
        learned_effect_preference=_hash(learned),
        reward_evidence=strategy.count.sum(1).tolist(),
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit", "play"))
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        out=ROOT / "games.json.gz",
        mode="live",
        deck="practice",
        carry=atlas.season.CARRY,
        cap=240,
        resume=False,
    )
    a = p.parse_args()
    fit() if a.action == "fit" else play(a)
