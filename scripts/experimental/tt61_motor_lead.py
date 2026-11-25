"""Learn how far to advance directly observed joint motion on an incoming ball."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt43_temporal_servo as servo
from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt61_motor_lead")
MODEL = ROOT / "sensor-readout.npz"
LEADS = (3, 7, 11)
course = dose.course
season = dose.season


def rows():
    features, moves, releases = [], [], []
    for p in servo.SOURCES:
        meta = _read(p.with_suffix(".json"))
        if meta["bank_sha256"] != _hash(p):
            raise ValueError("Changed paid motor observations")
        with np.load(p) as bank:
            states, controls, marks = [
                bank[k] for k in ("states", "targets_now", "marks")
            ]
        start = 0
        for sample in meta["practice"]:
            end = start + sample["frames"]
            for i in range(start + 6, end - max(LEADS)):
                j = i + max(LEADS)
                if marks[j, 0] - marks[i - 6, 0] != max(LEADS) + 6 or marks[j, 1] >= 0:
                    continue
                if np.any(states[i - 6 : j + 1, 4] > -0.9) or states[i, 1] < -1.18:
                    continue
                features.append(servo.feature(states[i - 6 : i + 1, :11]))
                moves.append(
                    np.concatenate(
                        [controls[i + k, :5] - controls[i, :5] for k in LEADS]
                    )
                )
                releases.append(sample["card"]["id"])
            start = end
    return np.array(features), np.array(moves), np.array(releases)


def fit():
    X, Y, ids = rows()
    loss = np.zeros(Y.shape[1])
    null = np.zeros_like(loss)
    for cohort in np.array_split(
        np.random.default_rng(61412).permutation(np.unique(ids)), 4
    ):
        hold = np.isin(ids, cohort)
        pred = servo.Readout(X[~hold], Y[~hold], True).predict(X[hold])
        loss += np.square(pred - Y[hold]).sum(0)
        null += np.square(Y[hold]).sum(0)
    quality = 1 - loss / np.maximum(null, 1e-9)
    model = servo.Readout(X, Y, True)
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL, **model.__dict__, quality=quality)
    _write(
        ROOT / "fit.json",
        dict(
            model=_hash(MODEL),
            rows=len(Y),
            ids=np.unique(ids).tolist(),
            leads=list(LEADS),
            observed_future_skill=quality.reshape(-1, 5).tolist(),
            sources={str(p): _hash(p) for p in servo.SOURCES},
            evaluation="32 complete broad random games in the first TT52 block, base 48-game motor checkpoint. Ordinary stock actions on the same schedule are available. Three lead choices correspond to demonstrated future joint-reference changes. The fourth choice advances no movement. Only an actual own legal landing earns a success.",
        ),
    )


class Player(season.Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with np.load(MODEL) as m:
            self.lead_model = servo.Readout.__new__(servo.Readout)
            self.lead_model.__dict__.update(
                {k: m[k] for k in m.files if k != "quality"}
            )
            self.quality = np.clip(m["quality"].reshape(-1, 5), 0, 1)
        self.outcomes = np.zeros((4, 3, 2))
        self.choice_rng = np.random.default_rng(610234)
        self.records = []
        keep = self.priors.keep

        def earn(tape, tag):
            self.register(1)
            return keep(tape, tag)

        self.priors.keep = earn

    def begin(self):
        super().begin()
        self.sensory = deque(maxlen=7)
        self.option = None
        self.issue_ticks = 0
        self.record_start = len(self.records)

    def register(self, paid):
        if self.option is not None:
            action, weights = self.option
            self.outcomes[action, :, paid] += weights
            self.records.append([action, weights.tolist(), self.issue_ticks, paid])
        self.option = None
        self.issue_ticks = 0

    def missed(self, tape):
        self.register(0)

    def step(self, t, eye, x, q, ref, base):
        target = super().step(t, eye, x, q, ref, base)
        if x is None or x[4] >= -0.9:
            self.sensory.clear()
            return target
        self.sensory.append(np.r_[x, q[:5]])
        if len(self.sensory) < 7 or not -1.18 < x[1] < 1.1:
            return target
        if self.best is not None and 0 <= t - self.tick < self.lease:
            return target
        if self.option is None:
            weights = np.exp(-0.5 * ((x[0] - np.array([-0.23, 0, 0.23])) / 0.14) ** 2)
            weights /= max(weights.sum(), 1e-12)
            totals = self.outcomes
            local = np.einsum("ajk,j->ak", totals, weights)
            # Share outcomes across receiving positions while retaining local differences.
            posterior = 0.55 * local + 0.45 * totals.sum(axis=1)
            wager = self.choice_rng.beta(4 + posterior[:, 1], 1 + posterior[:, 0])
            self.option = (int(np.argmax(wager)), weights)
        option, _ = self.option
        if option == 0:
            return target
        i = option - 1
        delta = self.lead_model.predict(servo.feature(self.sensory))[0].reshape(3, 5)[i]
        delta = np.clip(0.8 * self.quality[i] * delta, -0.32, 0.32)
        out = target.copy()
        out[:5] += delta
        self.issue_ticks += 1
        return out


def play(a):
    meta = _read(ROOT / "fit.json")
    if _hash(MODEL) != meta["model"] or set(meta["ids"]) & {
        c["id"] for c in _read(dose.PLAN)["practice"]
    }:
        raise ValueError("Changed or overlapping motor training")
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    who = []

    def make(*args, **kw):
        actor = Player(*args, coverage=rank, **kw)
        who.append(actor)
        return actor

    original = course.mpc.play

    def game(*args, **kw):
        result = original(*args, **kw)
        actor = who[-1]
        result["lead_trials"] = actor.records[actor.record_start :]
        result["lead_outcomes"] = actor.outcomes.tolist()
        return result

    with (
        enabled(),
        patch.object(course, "PLAN", dose.PLAN),
        patch.object(course.mpc, "MPC", make),
        patch.object(course.mpc, "play", game),
    ):
        course.run(a)
    _write(
        ROOT / "adaptive.json",
        dict(
            outcomes=who[0].outcomes.tolist(),
            rng=who[0].choice_rng.bit_generator.state,
            primary=_hash(a.out.with_suffix("").with_suffix(".npz")),
            readout=_hash(MODEL),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "play"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live",
        deck="practice",
        cap=240,
        carry=season.CARRY,
        out=ROOT / "live.json.gz",
        resume=False,
    )
    a = p.parse_args()
    fit() if a.phase == "fit" else play(a)
