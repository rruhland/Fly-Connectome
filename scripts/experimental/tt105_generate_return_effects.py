"""Generate a new physical stroke only if the known current return stays robust."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt52_practice_dose as dose
from . import tt102_outgoing_choice as outcome
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt105_generate_effects")
PLAN = ROOT / "protocol.json"
mpc = dose.course.mpc
H = mpc.HORIZON
WIDTH = 7


def alternatives(g, eta, z):
    # Smooth independent joint perturbations and a monotone retiming of the
    # actually chosen movement. All five actuators may change independently.
    t = np.arange(H)
    peak = max(2.0, eta / mpc.DT)
    phase = np.clip((t + 1) / peak, 0.0, 1.0)
    env = phase**2 * (3 - 2 * phase)
    ix = np.clip(t * (1.0 + 0.065 * z[-2]) + 1.4 * z[-1] * env, 0, H - 1)
    out = np.column_stack([np.interp(ix, t, g[:, j]) for j in range(6)])
    out[:, :5] += env[:, None] * z[:5] * [0.045, 0.035, 0.045, 0.065, 0.075]
    return out


def choose(score, radial, ok, baseline=0):
    # Require every modeled incoming perturbation to land with a supported
    # forecast. Reject changes that turn an ordinary face contact into a rim shot.
    if not ok[baseline].all():
        return baseline
    improvement = score - score[baseline]
    allowed = (
        ok.all(axis=1) & (improvement.min(1) > 0.012) & (improvement.mean(1) > 0.046)
    )
    allowed &= radial.max(1) <= min(0.067, max(0.047, radial[baseline].max() + 0.002))
    if not allowed.any():
        return baseline
    return int(
        np.argmax(
            np.where(
                allowed, 0.6 * improvement.mean(1) + 0.4 * improvement.min(1), -np.inf
            )
        )
    )


class Player(dose.season.Player):
    def __init__(self, *a, teacher, **kw):
        super().__init__(*a, **kw)
        self.teacher = teacher

    def begin(self):
        super().begin()
        self.spent = False
        self.surveys = []

    def step(self, tick, eye, x, q, reference, base):
        if x is not None and x[4] > 0.9:
            self.spent = False
        return super().step(tick, eye, x, q, reference, base)

    def predict_effect(self, goals):
        _, legal, trace, y, _ = self.trial(goals, long=True, traces=True)
        value = np.full(len(goals), -0.5)
        radius = np.full(len(goals), 1.0)
        valid = np.zeros(len(goals), bool)
        for i in np.flatnonzero(legal):
            f = outcome.from_trace(trace[i], y[i, 2])
            if f is None:
                continue
            expected = self.teacher.predict(f)
            if expected[3] > 2.25:
                continue
            value[i] = expected[2] - 0.15 * expected[1]
            radius[i] = y[i, 12]
            valid[i] = True
        return value, radius, valid

    def stress(self, goals):
        a = self.engine.twin
        b, v = a.ball.copy(), a.velocity.copy()
        w = a.data.qvel[a._ball_dof + 3 : a._ball_dof + 6]
        spin = w.copy()
        all_ = [self.predict_effect(goals)]
        try:
            for s in (-1.0, 1.0):
                a.ball[:] = b + s * outcome.OFF
                a.velocity[:] = v + s * outcome.DV
                w[:] = spin + [16.0 + 17.0 * s, 0.0, 0.0]
                all_.append(self.predict_effect(goals))
        finally:
            a.ball[:] = b
            a.velocity[:] = v
            w[:] = spin
        return [np.stack(items, axis=1) for items in zip(*all_, strict=True)]

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        if self.spent or not self.notes or self.notes[-1].get("tick") != tick:
            return
        note = self.notes[-1]
        if not note.get("chosen") or note.get("robust") != 2 or not self.engine.bounced:
            return
        eta = note["contact_delay_s"]
        if not 0.045 < eta < 0.22:
            return
        self.spent = True
        anchor = self.best
        current, _, good = self.predict_effect([anchor])
        if not good[0] or current[0] > 0.95:
            return
        z = np.vstack([np.zeros(WIDTH), np.eye(WIDTH), -np.eye(WIDTH)])
        trial = [alternatives(anchor, eta, p) for p in z]
        trial[0] = anchor
        val, _, viable = self.predict_effect(trial)
        # A paired finite-difference estimate proposes coordinated changes
        # only along directions that preserved a nominal legal return.
        plus = val[1 : 1 + WIDTH]
        minus = val[1 + WIDTH :]
        slope = np.where(viable[1 : 1 + WIDTH] & viable[1 + WIDTH :], plus - minus, 0.0)
        scale = np.max(abs(slope))
        ranked = list(np.argsort(-val, kind="stable")[:4])
        selected = [0] + [j for j in ranked if j and viable[j]]
        finals = [trial[j] for j in selected]
        codes = [z[j] for j in selected]
        if scale > 0.01:
            for gain in (0.7, 1.4):
                code = gain * slope / scale
                codes.append(code)
                finals.append(alternatives(anchor, eta, code))
        if len(finals) == 1:
            return
        value, radial, ok = self.stress(finals)
        k = choose(value, radial, ok)
        self.surveys.append(
            dict(
                tick=tick,
                tested=len(trial) + len(finals) * 3,
                raw=float(current[0]),
                best_nominal=float(val.max()),
                survived=int(ok.all(1).sum()),
                issued=bool(k),
                change=codes[k].tolist(),
                estimate=value[[0, k]].tolist(),
                radius=radial[[0, k]].tolist(),
            )
        )
        if k:
            self.best = finals[k]
            note["tactical_second_search"] = True


def freeze():
    p = _read(outcome.PLAN)
    with np.load(outcome.MODEL) as m:
        ids = set(map(int, m["ids"]))
    cards = _read(dose.PLAN)["practice"][:32]
    if ids & {c["id"] for c in cards}:
        raise ValueError("Effect teacher learned on the requested release")
    ROOT.mkdir(exist_ok=True)
    _write(
        PLAN,
        dict(
            cards=cards,
            teacher=_hash(outcome.MODEL),
            teacher_design=_hash(outcome.PLAN),
            carry=_hash(dose.season.CARRY),
            control="artifacts/tt52_practice_dose/live-from-48.json.gz",
            rule="One physical launch on every assigned course card. Preserve the full ordinary first search. Only after it finds a legal movement under both sensor perturbations, generate a new motion by finite differences in joint and time space, using estimated next returnability as the search objective. At most one offer per incoming ball. Both arms update their actual scored motor bank online. No candidate gets a physical retry.",
        ),
    )


def run(a):
    d = _read(PLAN)
    if (
        d["teacher"] != _hash(outcome.MODEL)
        or d["cards"] != _read(dose.PLAN)["practice"][:32]
    ):
        raise ValueError("Changed research inputs")
    teacher = outcome.Teacher()
    rank = dose.season.fresh.shot.Coverage(_read(dose.season.fresh.RISK))
    original = mpc.play

    def play(*v, **kw):
        r = original(*v, **kw)
        r["extra_surveys"] = v[10].surveys
        return r

    with (
        enabled(),
        patch.object(dose.course, "PLAN", dose.PLAN),
        patch.object(mpc, "play", play),
        patch.object(
            mpc,
            "MPC",
            lambda *v, **kw: Player(*v, coverage=rank, teacher=teacher, **kw),
        ),
    ):
        dose.course.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "run"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        n=32,
        mode="live",
        deck="practice",
        carry=dose.season.CARRY,
        resume=False,
        cap=240,
        out=ROOT / "live.json.gz",
    )
    a = p.parse_args()
    freeze() if a.stage == "plan" else run(a)
