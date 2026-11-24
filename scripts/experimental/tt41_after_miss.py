"""Learn counterfactual direct commands from an observed miss, after the point is over."""

import argparse
from collections import deque
from pathlib import Path
from time import process_time
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried
from .tt39_turn_time import turn_time


class CounterfactualMemory(Carried):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.inferred = deque(maxlen=120)
        self.recalled = 0

    def priors(self, x, q, reference, limit):
        proven = super().priors(x, q, reference, limit)
        key = mpc.query(x, q, reference)
        ranked = sorted(
            ((np.linalg.norm(s - key), i) for i, (s, _) in enumerate(self.inferred)),
            key=lambda r: r[0],
        )
        for dist, i in ranked[:2]:
            if dist < 1.7:
                proven.append((self.inferred[i][1], False))
                self.recalled += 1
        return proven


def observed_arc(track, tick, lower, upper, sign):
    seen = [(t, p) for t, p in track if lower <= t <= upper]
    if len(seen) < 3:
        return None
    times, p = map(np.asarray, zip(*seen, strict=True))
    dt = (times - tick) * mpc.DT
    basis = np.c_[np.ones(len(dt)), dt]
    g = -0.5 * 9.81 * dt**2
    y = p.copy()
    y[:, 2] -= g
    fitted = np.linalg.lstsq(basis, y, rcond=None)[0]
    if (
        sign * fitted[1, 1] < 0.6
        or np.sqrt(np.mean(np.sum((y - basis @ fitted) ** 2, axis=1))) > 0.022
    ):
        return None
    return fitted


def desired_loss(v, p):
    t = (
        v[:, 2] + np.sqrt(np.maximum(0.0, v[:, 2] ** 2 + 19.62 * (p[:, 2] - 0.779)))
    ) / 9.81
    duration = np.maximum(0.02, t * 0.94)
    xy = p[:, :2] + duration[:, None] * v[:, :2]
    net = -p[:, 1] / np.maximum(0.4, 0.94 * v[:, 1])
    height = p[:, 2] + net * v[:, 2] - 4.905 * net**2
    return (
        3.8 * np.maximum(abs(xy[:, 0]) - 0.45, 0)
        + abs(xy[:, 1] - 0.66)
        + 6 * np.maximum(0.965 - height, 0)
        + np.maximum(1.8 - v[:, 1], 0)
    )


class ReplayMotor(mpc.MPC):
    def begin(self):
        super().begin()
        self.shadows = deque(maxlen=7)
        self.points = deque(maxlen=120)

    def step(self, tick, eye, x, q, reference, base):
        history = getattr(eye.filter, "track", ())
        if history:
            time, xyz = history[-1]
            stamp = round(time / mpc.DT)
            if not self.points or self.points[-1][0] < stamp:
                self.points.append((stamp, np.asarray(xyz).copy()))
        return super().step(tick, eye, x, q, reference, base)

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        twin = self.engine.twin
        if -1.43 < x[1] < -0.63 and self.engine.bounced and self.engine.sync():
            shadow = mujoco.MjData(twin.model)
            mujoco.mj_copyData(shadow, twin.model, twin.data)
            self.shadows.append(
                (tick, mpc.query(x, q, reference), shadow, twin._received_bounce)
            )

    def missed(self, archive):
        start = process_time()
        if not self.shadows:
            return
        near = [
            (t, x, q)
            for t, x, q, *_ in archive
            if x is not None and x[4] < -0.9 and x[1] < -0.85
        ]
        if not near:
            return
        turn = turn_time([(t, x) for t, x, *_ in archive if x is None or x[1] < 0.40])
        near = near[-18:]
        if turn is None or turn < near[0][0] - 6:
            turn = min(
                near,
                key=lambda r: np.linalg.norm(
                    self.geometry.pose(r[2][:5])[0] - r[1][:3]
                ),
            )[0]
        good = [s for s in self.shadows if 4 <= turn - s[0] <= 18]
        if not good:
            return
        tick, key, shadow, bounced = min(good, key=lambda s: abs(turn - s[0] - 11))
        issued = np.array([r[-1] for r in archive if tick <= r[0] < tick + mpc.HORIZON])
        if not len(issued):
            return
        actual = mpc.tail(issued, mpc.HORIZON)
        twin = self.engine.twin
        mujoco.mj_copyData(twin.data, twin.model, shadow)
        twin._received_bounce = bounced
        before = observed_arc(self.points, tick, tick - 3, turn - 2, -1)
        if before is not None:
            twin.ball[:] += np.clip(before[0] - twin.ball, -0.028, 0.028)
            twin.velocity[:] += np.clip(before[1] - twin.velocity, -0.40, 0.40)
            mujoco.mj_forward(twin.model, twin.data)
        _, _, trace, _, baseline = self.trial([actual], traces=True)
        anchors = [actual]
        for face in (False, True):
            delta = self.transport(trace[0], face=face)
            if delta is not None:
                anchors.append(actual + delta)
        moves = anchors.copy()
        for i in range(48):
            moves.append(
                mpc.structured(anchors[i % len(anchors)], self.rng.normal(size=10), 1.0)
            )
        loss, _, _, _, f = self.trial(moves)
        out = observed_arc(self.points, turn, turn + 1, turn + 9, 1)
        if out is not None and baseline[0, 1] > 0.5:
            bias = np.clip(out[1] - baseline[0, 2:5], -1.8, 1.8)
            contact = f[:, 1] > 0.5
            loss[contact] = (
                desired_loss(f[contact, 2:5] + bias, f[contact, 5:8])
                + 2 * f[contact, 8]
            )
        rank = np.argsort(loss)[:11]
        nominal, legal, *_ = self.trial([moves[i] for i in rank], long=True)
        valid = np.flatnonzero(legal)
        self.diag["miss_replayed"] += 1
        if len(valid):
            selected = valid[np.argsort(nominal[valid] + 0.60 * loss[rank[valid]])[:4]]
            agree, risk = self.uncertainty(
                [moves[rank[i]] for i in selected], nominal[selected]
            )
            for i in np.argsort(risk):
                j = rank[selected[i]]
                if agree[i] == 2 and j != 0:
                    self.priors.inferred.append((key, moves[j].copy()))
                    self.diag["inferred_corrections"] += 1
                    break
        self.diag["miss_practice_cpu_s"] += process_time() - start


def run(a):
    held = {c["id"] for c in _read(a.plan)[a.deck][: a.limit]}
    models = []

    def bank(file, engine):
        m = CounterfactualMemory(
            file, engine, practice=[mpc.ROOT / "tt37/camera-time-school.npz"], held=held
        )
        models.append(m)
        return m

    with patch.object(mpc, "MPC", ReplayMotor), patch.object(mpc, "Fragments", bank):
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
    p = a.out.with_suffix("").with_suffix(".npz")
    m = models[0]
    np.savez_compressed(
        p,
        estimated_states=np.array([r[0] for r in m.inferred]),
        estimated_actions=np.array([r[1] for r in m.inferred]),
        earned_states=np.array([r[0] for r in m.online]),
        earned_actions=np.array([r[1] for r in m.online]),
        earned_tags=[repr(r[2]) for r in m.online],
    )
    doc.update(
        recalled=m.recalled, inferred=len(m.inferred), checkpoint_sha256=_hash(p)
    )
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
