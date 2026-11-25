"""Incrementally distill the next few ticks of the arm's own scored movement into its early optical feedback."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt47_predictive")
DT = course.mpc.DT


class Forecast:
    def __init__(self):
        self.encoder = servo.reload(servo.ROOT / "prior.npz")
        # A subset of the original fixed nonlinear features lowers the cost
        # of a full recursive least-squares update after each earned reward.
        self.dim = 48
        self.inverse = np.eye(self.dim) / 3.0
        self.weights = np.zeros((self.dim, 5))
        self.n = 0

    def encode(self, seq):
        f = self.encoder.encode(servo.feature(seq))[0]
        return np.r_[1.0, np.clip(f[1:12], -2.0, 2.0) * 0.4, f[-36:]] / 3.0

    def predict(self, v):
        return v @ self.weights

    def update(self, x, y):
        denom = self.inverse @ x
        gain = denom / (1.0 + x @ denom)
        self.weights += np.outer(gain, y - self.predict(x))
        self.inverse -= np.outer(gain, denom)
        self.n += 1


class Actor(season.Player):
    def __init__(self, *args, learned, **kw):
        super().__init__(*args, **kw)
        self.learned = learned
        self.priors.on_reward = self.hear

    def begin(self):
        super().begin()
        self.experience = []

    def step(self, tick, eye, x, q, ref, base):
        original = super().step(tick, eye, x, q, ref, base)
        if (
            x is None
            or x[4] >= -0.9
            or not -0.46 < x[1] < 1.35
            or len(self.samples) != 7
        ):
            return original
        feature = self.learned.encode(self.samples)
        self.experience.append((tick, feature, original[:5].copy()))
        shift = np.clip(self.learned.predict(feature), -0.16, 0.16)
        self.diag["forecast_motor_uses"] += int(np.max(abs(shift)) > 0.012)
        goal = original.copy()
        goal[:5] += 0.65 * shift
        return goal

    def hear(self, archive):
        # This scorer is called after a real legal return. Restrict credit
        # to the last optical approach and to fully observed signed movements.
        arrivals = [
            t for t, x, *_ in archive if x is not None and x[4] < -0.9 and x[1] < -0.87
        ]
        if arrivals:
            end = arrivals[-1]
            table = {r[0]: r for r in archive}
            kept = [z for z in self.experience if end - 65 < z[0] < end - 4]
            for t, f, original in kept[::3]:
                row = table.get(t + 4)
                if row is None or row[1] is None or row[1][4] >= -0.9:
                    continue
                self.learned.update(f, np.clip(row[-1][:5] - original, -0.32, 0.32))
        self.experience.clear()

    def missed(self, tape):
        self.experience.clear()


class Memory(course.Carried):
    def keep(self, archive, leg):
        self.on_reward(archive)
        return super().keep(archive, leg)


def run(a):
    ink = Forecast()
    ranker = fresh.shot.Coverage(_read(fresh.RISK))
    native = course.mpc.play

    def build(*args, **kw):
        return Actor(*args, learned=ink, coverage=ranker, **kw)

    def play(*args, **kw):
        r = native(*args, **kw)
        r["forecast_samples"] = ink.n
        # The readout is small enough to checkpoint every game as ordinary
        # numeric data in the action receipt.
        r["forecast_weights"] = ink.weights.tolist()
        return r

    with (
        patch.object(course, "PLAN", fresh.PLAN),
        patch.object(course, "Carried", Memory),
        patch.object(course.mpc, "MPC", build),
        patch.object(course.mpc, "play", play),
    ):
        course.run(a)
    d = _read(a.out)
    d.update(
        update=_hash(Path(__file__)),
        inverse=ink.inverse.tolist(),
        features=_hash(servo.ROOT / "prior.npz"),
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", default=24, type=int)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live", deck="assessment", carry=fresh.CARRY, cap=240, resume=False
    )
    a = p.parse_args()
    a.out.parent.mkdir(exist_ok=True, parents=True)
    run(a)
