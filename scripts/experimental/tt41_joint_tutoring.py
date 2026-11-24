"""Offer a scored spatial motor tutor to a direct controller when its own search struggles."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from . import tt37_motor_clinic as clinic
from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_posture import Surface, sample
from .tt39_carry_joints import Carried


class Tutor(mpc.MPC):
    def __init__(self, bank, raw, *, variant, budget, motor, override):
        super().__init__(bank, raw, variant=variant, budget=budget)
        poses = sample(self.geometry)
        self.surface = Surface(self.geometry, poses, -1.56, 0.35)
        self.motor = motor
        self.override = override
        self.cloud = clinic.generate(20)
        self.teacher_pending = None

    def observation(self, tick, depth):
        with patch.object(joint, "RECEIVE_Y", depth):
            return joint.flight_at_plane(self.engine.latest, tick * mpc.DT)

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        note = self.notes[-1] if self.notes and self.notes[-1]["tick"] == tick else {}
        good = bool(note.get("chosen")) and note.get("robust", 0) > 0
        if good and not self.override:
            return
        original = self.best
        rest = None
        if original is not None and 0 <= tick - self.tick < self.lease:
            rest = mpc.tail(original[tick - self.tick :], mpc.HORIZON)
        candidates = []
        for depth in (-1.56, -1.44, -1.62):
            opt = self.observation(tick, depth)
            if opt is None or not 0.035 < opt[0] < 0.225:
                continue
            eta = opt[0]
            context = np.r_[opt[1:6], q[:5]]
            fit = clinic.geometry(self.surface, opt, depth=depth)
            dist = np.linalg.norm((self.contexts - context) / clinic.SCALE, axis=1)
            ids = np.argsort(dist)[:7]
            remembered = self.corrections[ids]
            controls = np.vstack(
                (remembered, self.cloud[::2], remembered[0] + 0.63 * self.cloud[1::2])
            )
            cmds = clinic.schedules(
                *fit, eta, controls, mpc.DT, self.surface.ready, mpc.HORIZON
            )
            candidates.extend(cmds)
        if not candidates:
            return
        self.diag["tutor_consulted"] += 1
        if rest is not None:
            candidates.insert(0, rest)
        short, *_ = self.trial(candidates)
        idx = np.unique(np.r_[0, np.argsort(short)[:10]])
        loss, legal, *_ = self.trial([candidates[j] for j in idx], long=True)
        valid = np.flatnonzero(legal)
        if not len(valid):
            return
        best = valid[np.argsort(loss[valid])[:4]]
        if good and rest is not None and legal[0]:
            best = np.unique(np.r_[best, 0])
        choices = idx[best]
        agreement, risk = self.uncertainty([candidates[j] for j in choices], loss[best])
        safe = np.flatnonzero(agreement >= (2 if good else 1))
        if not len(safe):
            return
        v = safe[np.argmin(risk[safe])]
        if rest is not None and choices[v] == 0:
            return
        incumb = np.flatnonzero(choices == 0) if rest is not None else []
        if (
            good
            and len(incumb)
            and agreement[incumb[0]] >= 1
            and risk[v] + 0.60 >= risk[incumb[0]]
        ):
            return
        self.best, self.tick, self.lease = candidates[choices[v]], tick, mpc.HORIZON
        self.next = tick + min(12, max(5, round(0.65 * mpc.HORIZON)))
        self.diag["teacher_admitted"] += 1
        note.update(
            tutor=True,
            tutor_predicted_quality=float(risk[v]),
            tutor_variants=int(agreement[v]),
        )

    def install(self):
        self.contexts = np.array([r["x"] for r in self.motor])
        self.corrections = np.array([r["change"] for r in self.motor])


def run(a):
    prior = _read(a.motor)
    held = {c["id"] for c in _read(a.plan)[a.deck][: a.limit]}
    if held & {c["id"] for c in prior["cards"]}:
        raise ValueError("Test release was used in motor tutoring")
    instances = []

    def model(file, raw, **kw):
        actor = Tutor(file, raw, **kw, motor=prior["bank"], override=a.compare)
        actor.install()
        instances.append(actor)
        return actor

    def bank(file, engine):
        return Carried(
            file, engine, practice=[mpc.ROOT / "tt37/camera-time-school.npz"], held=held
        )

    with patch.object(mpc, "MPC", model), patch.object(mpc, "Fragments", bank):
        mpc.run(
            a.feeds,
            mpc.BANK,
            a.plan,
            a.deck,
            "live",
            a.out,
            optic=mpc.ROOT / "tt37/rebound.npz",
            limit=a.limit,
            variant="elastic",
            cap=240,
        )
    doc = _read(a.out)
    doc.update(motor_practice_sha256=_hash(a.motor), compare_with_viable=a.compare)
    rows = instances[0].priors.online
    ck = a.out.with_suffix("").with_suffix(".npz")
    np.savez_compressed(
        ck,
        states=[r[0] for r in rows],
        goals=[r[1] for r in rows],
        tags=[repr(r[2]) for r in rows],
    )
    doc["checkpoint_sha256"] = _hash(ck)
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--compare", action="store_true")
    p.add_argument(
        "--motor", type=Path, default=mpc.ROOT / "tt37/independent-cold-value.json.gz"
    )
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
