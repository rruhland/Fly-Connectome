"""Reward-weighted exploration of early direct joint feedback, with exploration carried across real rallies."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from . import tt44_rally_course as fresh
from . import tt45_context_feedback as control
from . import tt45_error_learning as observed
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt46_search")
MOTORS = ROOT / "modes.npz"


def fit(out):
    x, y, g = servo.packets(servo.SOURCES)
    model = servo.reload(servo.ROOT / "prior.npz")
    err = y - model.predict(x)
    _, values, v = np.linalg.svd(err, full_matrices=False)
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out, basis=v[:3], deviations=values / np.sqrt(len(err)))
    _write(
        out.with_suffix(".json"),
        dict(
            model=_hash(out),
            source=_hash(servo.ROOT / "prior.npz"),
            releases=np.unique(g).tolist(),
            samples=len(err),
        ),
    )


class LocalSearch:
    def __init__(self, basis, learn=True):
        self.rows = deque(maxlen=1100)
        self.basis = basis
        self.rng = np.random.default_rng(46093441)
        self.learn = learn

    def act(self, key):
        z = np.zeros(3)
        support = 0.0
        if self.rows:
            distances = observed.dis(key, [r[0] for r in self.rows])
            ids = np.argsort(distances, kind="stable")[:38]
            similarity = np.exp(-3.5 * distances[ids] ** 2)
            quality = np.array([self.rows[i][2] for i in ids])
            # Weight local actions by relative performance of the actual
            # incoming attempts. A null response has four pseudocounts.
            weights = similarity * np.exp(
                4.0 * (quality - np.average(quality, weights=similarity + 1e-9))
            )
            support = similarity.sum()
            z = (
                weights
                @ np.stack([self.rows[i][1] for i in ids])
                / (weights.sum() + 4.0)
            )
        if self.learn:
            spread = 0.11 / np.sqrt(1.0 + support / 10.0)
            z += np.clip(self.rng.normal(size=3), -1.8, 1.8) * spread
        return np.clip(z, -0.23, 0.23)

    def hear(self, key, z, completed, achieved):
        if not self.learn:
            return None
        r = float(completed)
        if not completed and achieved is not None:
            _, miss, deflection = achieved
            r = 0.14 * np.exp(-np.linalg.norm(miss) / 0.09) + 0.13 * float(
                deflection is not None
            )
        self.rows.append((key, z, r))
        return r


class Player(fresh.Hybrid, season.ReuseMPC):
    def __init__(self, *args, local, **kw):
        super().__init__(*args, **kw)
        self.local = local
        self.priors.hear = self.hear

    def begin(self):
        super().begin()
        self.sensor = control.Approach()
        self.exploration = None
        self.attempts = []

    def step(self, tick, eye, x, q, reference, base):
        context = self.sensor.observe(x, q)
        if context is not None:
            z = self.local.act(context)
            self.exploration = (tick, context, z)
        u = super().step(tick, eye, x, q, reference, base)
        if (
            self.exploration is not None
            and x is not None
            and x[4] < -0.9
            and -0.49 < x[1] < 1.35
        ):
            z = self.exploration[2]
            u = u.copy()
            u[:5] += np.clip(z @ self.local.basis, -0.18, 0.18)
        return u

    def hear(self, tape, reward):
        if self.exploration is None:
            return
        start, key, z = self.exploration
        self.exploration = None
        achieved = None if reward else observed.sense(tape, start, self.geometry)
        value = self.local.hear(key, z, reward, achieved)
        self.attempts.append(
            dict(
                tick=start,
                context=key.tolist(),
                coordinates=z.tolist(),
                landed=bool(reward),
                reward=value,
            )
        )

    def missed(self, tape):
        self.hear(tape, False)


class Memory(course.Carried):
    def keep(self, tape, leg):
        n = super().keep(tape, leg)
        self.hear(tape, True)
        return n


def run(a):
    if _hash(MOTORS) != _read(MOTORS.with_suffix(".json"))["model"]:
        raise ValueError("Motor basis differs")
    ids = {r["id"] for r in _read(season.PLAN)[a.deck][: a.n]}
    if ids & set(_read(MOTORS.with_suffix(".json"))["releases"]):
        raise ValueError("Learning and assessment overlap")
    with np.load(MOTORS) as data:
        policy = LocalSearch(data["basis"], a.mode == "live")
    if a.policy:
        earlier = _read(a.policy)
        if ids.intersection(c["id"] for c in earlier["cards"][: len(earlier["rows"])]):
            raise ValueError("Policy has already seen evaluation releases")
        for row in earlier["rows"]:
            for trial in row["early_trials"]:
                policy.rows.append(
                    (
                        np.array(trial["context"]),
                        np.array(trial["coordinates"]),
                        trial["reward"],
                    )
                )
    risk = fresh.shot.Coverage(_read(fresh.RISK))
    ordinary = course.mpc.play

    def play(*args, **kwargs):
        row = ordinary(*args, **kwargs)
        row["early_trials"] = args[10].attempts
        return row

    def factory(*args, **kwargs):
        return Player(*args, coverage=risk, local=policy, **kwargs)

    with (
        patch.object(course, "PLAN", season.PLAN),
        patch.object(course, "Carried", Memory),
        patch.object(course.mpc, "MPC", factory),
        patch.object(course.mpc, "play", play),
    ):
        course.run(a)
    d = _read(a.out)
    d.update(
        basis=_hash(MOTORS),
        local_reward_learning="bounded nearby perturbations of learned joint feedback with observed returns",
        source=_hash(Path(__file__)),
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "live", "frozen"))
    p.add_argument("out", type=Path)
    p.add_argument("--deck", default="practice", choices=("practice", "assessment"))
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--policy", type=Path)
    p.add_argument("--carry", type=Path, default=season.CARRY)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(cap=240, resume=False)
    a = p.parse_args()
    fit(a.out) if a.mode == "fit" else run(a)
