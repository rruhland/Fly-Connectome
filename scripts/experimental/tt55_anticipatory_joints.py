"""Start the predictable part of a rewarded joint movement while the ball is at the opponent."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt46_continuous_season as season
from . import tt50_forecast_learning as optic
from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

mpc = season.course.mpc
ROOT = Path("artifacts/tt55_anticipatory_joints")
BANK = ROOT / "observed-preparation.npz"
SCALE = np.array([0.12, 0.055, 0.32, 0.55, 0.55])


def crossing(seq, y):
    for a, b in zip(seq, seq[1:]):
        if a[1] > y >= b[1] and a[4] < -0.8 and b[4] < -0.8:
            return a + (b - a) * ((y - a[1]) / (b[1] - a[1]))
    return None


def behavior(seq):
    if len(seq) < 3:
        return None
    early = crossing(seq, -0.48)
    mid = crossing(seq, -1.0)
    future = crossing(seq, -1.1)
    if early is None or mid is None or future is None:
        return None
    return future[[0, 2, 3, 4, 5]], mid[6:11] - early[6:11]


def predict(keys, motions, key):
    d = np.sum(((keys - key) / SCALE) ** 2, axis=1)
    ids = np.argsort(d)[:24]
    w = np.exp(-0.45 * np.minimum(d[ids], 50))
    return (w @ motions[ids] + 2 * np.median(motions, axis=0)) / (w.sum() + 2)


def fit():
    keys = []
    moves = []
    groups = []
    forbidden = {c["id"] for c in _read(dose.PLAN)["practice"]}
    for p in season.fresh.servo.SOURCES:
        meta = _read(p.with_suffix(".json"))
        if _hash(p) != meta["bank_sha256"]:
            raise ValueError("Practice changed")
        with np.load(p) as z:
            states, marks = z["states"], z["marks"]
        pos = 0
        for ep in meta["practice"]:
            count = ep["frames"]
            x = states[pos : pos + count]
            m = marks[pos : pos + count]
            pos += count
            ident = ep["card"]["id"]
            if ident in forbidden:
                raise ValueError("Training release contaminates independent cohort")
            # Each archive group is one physically rewarded movement. Only
            # sensor crossings and joint changes supply its numerical label.
            hit = np.rint((m[:, 0] * mpc.DT - m[:, 1]) * 480).astype(int)
            for label in np.unique(hit):
                use = x[(hit == label) & (m[:, 1] < 0)]
                pair = behavior(use)
                if pair is not None:
                    k, v = pair
                    keys.append(k)
                    moves.append(v)
                    groups.append(ident)
    keys, moves, groups = map(np.array, (keys, moves, groups))
    losses = np.zeros(5)
    constant = np.zeros(5)
    for ids in np.array_split(
        np.random.default_rng(55004).permutation(np.unique(groups)), 5
    ):
        held = np.isin(groups, ids)
        outside = ~held
        for x, v in zip(keys[held], moves[held], strict=True):
            losses += (predict(keys[outside], moves[outside], x) - v) ** 2
            constant += (np.median(moves[outside], axis=0) - v) ** 2
    skill = 1 - losses / np.maximum(constant, 1e-12)
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BANK, keys=keys, motions=moves, groups=groups, reliability=np.clip(skill, 0, 1)
    )
    _write(
        ROOT / "fit.json",
        dict(
            model=_hash(BANK),
            count=len(moves),
            releases=len(np.unique(groups)),
            fractions_of_unconditional_squared_error_removed=skill.tolist(),
        ),
    )


class Player(season.Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with np.load(BANK) as z:
            self.keys = z["keys"]
            self.moves = z["motions"]
            self.skill = z["reliability"]
        self.more = deque(maxlen=192)
        self.reader = optic.Online()
        # Calibrate the perceptual forecast from the camera's later measured
        # trajectories in the existing 48-game practice recording.
        for p in sorted((optic.ROOT / "practice-episodes").glob("game-*.json.gz")):
            for leg in _read(p)["observed"]:
                for d in leg["decisions"]:
                    self.reader.calibrator.observe(
                        d["gate"], np.array(d["x"]), np.array(leg["target"])
                    )
        keep = self.priors.keep

        def store(archive, label):
            rows = []
            newest = None
            for _, x, q, _, _ in archive:
                if x is None:
                    continue
                if x[4] < -0.9:
                    rows.append(np.r_[x, q[:5]])
                elif rows:
                    newest = behavior(rows)
                    rows = []
            if newest is not None:
                self.more.append(newest)
                self.diag["earned_preparations"] += 1
            return keep(archive, label)

        self.priors.keep = store

    def begin(self):
        super().begin()
        self.reader.reset()
        self.delta = np.zeros(5)
        self.early = None

    def watch(self, frames, x, t):
        g = self.reader.update(frames, x, t)
        if x is not None and x[4] < -0.9:
            self.early = None
        if g is None:
            return
        _, pred = g
        if self.more:
            k, v = map(np.asarray, zip(*self.more, strict=True))
            keys = np.vstack((self.keys, k))
            motion = np.vstack((self.moves, v))
        else:
            keys, motion = self.keys, self.moves
        # Only the direction-specific, release-predictable part of a motor
        # response is advanced. The average preparation shared across situations
        # is already provided by the established controller.
        deviation = predict(keys, motion, pred) - np.median(self.moves, axis=0)
        self.early = np.clip(0.85 * self.skill * deviation, -0.5, 0.5)
        self.diag["visual_previews"] += 1

    def step(self, t, eye, x, q, ref, base):
        usual = super().step(t, eye, x, q, ref, base)
        target = np.zeros(5) if self.early is None else self.early
        self.delta += np.clip(target - self.delta, -0.032, 0.032)
        if x is not None and x[4] < -0.9 and x[1] < -0.4:
            self.delta[:] = 0
        goal = usual.copy()
        goal[:5] += self.delta
        self.diag["anticipated_joint_ticks"] += int(np.max(abs(self.delta)) > 0.015)
        return goal


def play(a):
    ordinary = mpc.observed
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    active = []

    def make(*args, **kwargs):
        actor = Player(*args, coverage=rank, **kwargs)
        active.append(actor)
        return actor

    def observe(eye, t, frames, q):
        x = ordinary(eye, t, frames, q)
        active[-1].watch(frames, x, t)
        return x

    with (
        enabled(),
        patch.object(season.course, "PLAN", dose.PLAN),
        patch.object(mpc, "MPC", make),
        patch.object(mpc, "observed", observe),
    ):
        season.course.run(a)
    d = _read(a.out)
    learner = active[-1]
    calibration = learner.reader.calibrator
    carried = ROOT / "learned-preparation.npz"
    words = {}
    for i, g in enumerate(optic.GATES):
        words[f"correction_{i}"] = calibration.correction[g]
        words[f"covariance_{i}"] = calibration.covariance[g]
        words[f"count_{i}"] = calibration.count[g]
    np.savez_compressed(
        carried,
        keys=np.array([v[0] for v in learner.more]),
        motions=np.array([v[1] for v in learner.more]),
        **words,
    )
    d["anticipation_model"] = _hash(BANK)
    d["carried_preparation"] = _hash(carried)
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit", "play"))
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        out=ROOT / "live.json.gz",
        deck="practice",
        mode="live",
        cap=240,
        carry=season.CARRY,
        resume=False,
    )
    a = p.parse_args()
    fit() if a.action == "fit" else play(a)
