"""Choose between acquired direct joint sequences using camera cues and a private motor model."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried
from .tt41_rising_school import FOLDER


class Movement:
    def __init__(self, data, geometry):
        self.phase = np.array(data["phase"])
        self.refs = np.array(data["reference"])
        self.contact = np.array(data["contact"])
        q = np.asarray(data["q"])[int(np.argmin(abs(self.phase)))]
        _, _, jac, normal = geometry.pose(q[:5], jacobian=True)
        self.inverse = np.linalg.pinv(np.r_[jac, 0.085 * normal, 0.024 * np.eye(5)])

    def sequence(self, phase, px, pz, vx):
        delta = np.r_[
            px - self.contact[0],
            0.0,
            pz - self.contact[2],
            -0.035 * (vx - self.contact[3]),
            0.0,
            0.0,
            np.zeros(5),
        ]
        adapt = np.r_[np.clip(self.inverse @ delta, -1.1, 1.1), 0.0]
        return (
            np.stack(
                [np.interp(phase, self.phase, series) for series in self.refs.T],
                axis=-1,
            )
            + adapt
        )

    def cue(self, flight, now):
        with patch.object(joint, "RECEIVE_Y", self.contact[1]):
            return joint.flight_at_plane(flight, now)


class Modes(mpc.MPC):
    def __init__(self, bank, raw, *, variant, budget, examples, standby):
        super().__init__(bank, raw, variant=variant, budget=budget)
        self.moves = [Movement(ex, self.geometry) for ex in examples]
        self.standby = standby
        self.center = min(self.moves, key=lambda m: abs(m.contact[0]))

    def begin(self):
        super().begin()
        self.active = None
        self.last_cue = None
        self.past_first = False

    def wait_pose(self, base):
        move = self.center
        ready = move.sequence(-15.0, 0.0, move.contact[2], 0.0)
        return self.standby * ready + (1 - self.standby) * base

    def step(self, tick, eye, x, q, reference, base):
        now = tick * mpc.DT
        self.engine.observe(tick, eye, q, reference.copy())
        if x is not None and x[4] > 0.9:
            self.past_first = True
            self.active = None
            self.best = None
            self.last_cue = None
            return self.wait_pose(base)
        f = eye.latest
        cues = [(m, m.cue(f, now)) for m in self.moves]
        valid = [(m, c) for m, c in cues if c is not None]
        with patch.object(joint, "RECEIVE_Y", -1.53):
            late = joint.flight_at_plane(f, now)
        if self.active is None and late is not None and valid and late[0] > 0.12:
            if late[2] < 1.015:
                self.active = True
                self.next = tick
                self.best = None
            elif late[0] < 0.29:
                self.active = False
        if not self.active:
            if self.active is None and not valid:
                base = self.wait_pose(base) if self.past_first else base
            with patch.object(self.engine, "observe", lambda *args: None):
                return super().step(tick, eye, x, q, reference, base)
        if valid:
            self.last_cue = (tick, valid)
        elif self.last_cue is not None:
            old, rows = self.last_cue
            if tick - old < 10:
                valid = [(m, c.copy()) for m, c in rows]
                for _, c in valid:
                    c[0] -= (tick - old) * mpc.DT
        if self.best is not None and 0 <= tick - self.tick < self.lease:
            return self.best[tick - self.tick]
        if not valid:
            return self.wait_pose(base)

        def cost(row):
            m, c = row
            at = m.sequence(-13.0, c[1], c[2], c[3])
            return (
                ((c[1] - m.contact[0]) / 0.26) ** 2
                + ((c[2] - m.contact[2]) / 0.13) ** 2
                + ((c[3] - m.contact[3]) / 2) ** 2
                + 0.28 * np.mean((q[:5] - at[:5]) ** 2)
            )

        preferred, cue = min(valid, key=cost)
        phase = -cue[0] / mpc.DT
        if (
            tick >= self.next
            and x is not None
            and 0.04 < cue[0] < 0.27
            and self.engine.sync()
        ):
            self.next = tick + 4
            moves = []
            for m, c in valid:
                for shift in (0.0, -1.8, 1.8):
                    ticks = -c[0] / mpc.DT + np.arange(mpc.HORIZON) + shift
                    moves.append(m.sequence(ticks, c[1], c[2], c[3]))
            moves.extend(seq for seq, _ in self.priors.priors(x, q, reference, 5))
            bases = len(moves)
            for j in range(30):
                moves.append(
                    mpc.structured(moves[j % bases], self.rng.normal(size=10), 0.64)
                )
            short, *_ = self.trial(moves)
            order = np.argsort(short)[:10]
            loss, legal, *_ = self.trial([moves[j] for j in order], long=True)
            ranked = np.flatnonzero(legal)[np.argsort(loss[legal])[:4]]
            self.diag["shallow_trials"] += len(moves)
            if len(ranked):
                ids = order[ranked]
                win, agree, _ = self.robust([moves[j] for j in ids], ids, loss[ranked])
                if agree:
                    self.best, self.tick, self.lease = moves[win], tick, mpc.HORIZON
                    self.diag["shallow_commits"] += 1
                    return self.best[0]
        return preferred.sequence(np.clip(phase, -18, 8), cue[1], cue[2], cue[3])


def run(a):
    files = [
        FOLDER / (s + "-example.json") for s in ("stereo", "center", "right", "flip")
    ]
    ex = [_read(f) for f in files]
    held = {c["id"] for c in _read(a.plan)[a.deck][: a.limit]}
    if held & {r["id"] for r in ex}:
        raise ValueError("Test release appears in motor demonstrations")

    def factory(file, raw, **kw):
        return Modes(file, raw, **kw, examples=ex, standby=a.standby)

    def bank(file, engine):
        return Carried(
            file, engine, practice=[mpc.ROOT / "tt37/camera-time-school.npz"], held=held
        )

    with patch.object(mpc, "MPC", factory), patch.object(mpc, "Fragments", bank):
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
    doc.update(practice={p.name: _hash(p) for p in files}, standby=a.standby)
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--standby", type=float, default=0.4)
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
