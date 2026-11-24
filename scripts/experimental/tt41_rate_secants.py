"""Improve a prospective direct action by measuring signed precontact effects in motor-rate space."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried


def rate_space(goals, prev, stride, lo, hi):
    rates = []
    ref = prev.copy()
    for g in goals:
        r = np.clip((np.clip(g, lo, hi) - ref) / stride, -0.8, 0.8)
        rates.append(r)
        ref = np.clip(ref + r * stride, lo, hi)
    return np.array(rates)


def apply(rates, ref, stride, lo, hi, basis, z):
    offset = np.einsum("th,bhj->btj", basis, np.asarray(z).reshape(-1, 2, 5))
    result = np.broadcast_to(rates, (len(offset), *rates.shape)).copy()
    result[:, :, :5] = np.clip(result[:, :, :5] + offset, -0.8, 0.8)
    position = np.broadcast_to(ref, (len(offset), len(ref))).copy()
    for i in range(len(rates)):
        position = np.clip(position + result[:, i] * stride, lo, hi)
        result[:, i] = position
    return result


def departure_error(f):
    v = f[:, 2:5]
    p = f[:, 5:8]
    # Many outbound velocities can continue the rally. Choose the nearest
    # flight reaching an interior part of the opposite surface with net clearance.
    goals = []
    for duration in (0.42, 0.49, 0.57, 0.65, 0.75):
        aim = np.c_[
            np.clip(p[:, 0] + 0.93 * duration * v[:, 0], -0.34, 0.34),
            np.clip(p[:, 1] + 0.93 * duration * v[:, 1], 0.30, 1.08),
            np.full(len(p), 0.78),
        ]
        want = (aim - p) / (0.93 * duration)
        want[:, 2] += 0.5 * 9.81 * duration
        t = -p[:, 1] / np.maximum(0.5, 0.93 * want[:, 1])
        h = p[:, 2] + want[:, 2] * t - 4.905 * t * t
        want[:, 2] += np.maximum(0.0, 1.025 - h) / np.maximum(0.09, t)
        goals.append(want - v)
    ds = np.stack(goals)
    error = np.linalg.norm(ds * [0.8, 1.0, 1.2], axis=2)
    j = np.argmin(error, axis=0)
    return ds[j, np.arange(len(f))], error.min(axis=0) + 5 * np.maximum(
        0.0, f[:, 8] - 0.035
    )


class Secants(mpc.MPC):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        a = self.engine.twin
        axes = self.engine.actuators
        self.stride = a._joint_increments[axes] * self.engine.predict.micro
        self.lo = a._joint_low[axes]
        self.hi = a._joint_high[axes]

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        note = self.notes[-1] if self.notes and self.notes[-1]["tick"] == tick else {}
        if not self.engine.bounced or x[1] > -0.76 or x[1] < -1.48:
            return
        if note.get("chosen") and note.get("robust", 0) >= 2:
            return
        if not self.engine.sync():
            return
        goals = [s for s, _ in self.priors.priors(x, q, reference, 5)]
        if self.best is not None and 0 <= tick - self.tick < self.lease:
            goals.insert(0, mpc.tail(self.best[tick - self.tick :], mpc.HORIZON))
        if not goals:
            goals = [np.broadcast_to(base, (mpc.HORIZON, 6)).copy()]
        _, _, tracks, _, f = self.trial(goals, traces=True)
        _, loss = departure_error(f)
        order = np.argsort(np.where(f[:, 1] > 0.5, loss, 20 + 20 * f[:, 0]))
        options = []
        for i in order[:2]:
            trace = tracks[i]
            visible = np.flatnonzero(np.isfinite(trace[:, 0]))
            if len(visible) < 3:
                continue
            rel = [
                np.linalg.norm(trace[k, :3] - self.geometry.pose(trace[k, 6:11])[0])
                for k in visible
            ]
            eta = int(visible[np.argmin(rel)])
            if eta < 3:
                continue
            t = np.arange(mpc.HORIZON)
            width = max(2.0, eta * 0.45)
            basis = np.stack(
                [
                    np.exp(-0.5 * ((t - eta * c) / width) ** 2) * (t <= eta + 2)
                    for c in (0.12, 0.82)
                ],
                axis=1,
            )
            rates = rate_space(goals[i], reference, self.stride, self.lo, self.hi)
            z = np.vstack([np.zeros(10), np.eye(10) * 0.24, -np.eye(10) * 0.24])
            near = apply(rates, reference, self.stride, self.lo, self.hi, basis, z)
            _, _, _, _, fb = self.trial(near)
            delta, penalty = departure_error(fb)
            good = fb[:, 1] > 0.5
            if not np.any(good):
                continue
            sorted_idx = np.argsort(np.where(good, penalty, 1e6))
            options.extend(near[sorted_idx[:2]])
            # Around each observed contact mode, learn how independent motor
            # rate differences change its outgoing velocity. Weight by closeness
            # in both center and response to avoid mixing opposite blade faces.
            for anchor in sorted_idx[: min(3, int(good.sum()))]:
                d = z[good] - z[anchor]
                effect = fb[good, 2:5] - fb[anchor, 2:5]
                weight = np.exp(
                    -0.25 * np.sum(effect**2, axis=1)
                    - 320 * np.sum((fb[good, 5:8] - fb[anchor, 5:8]) ** 2, axis=1)
                )
                j = np.linalg.solve(
                    d.T @ (d * weight[:, None]) + 0.009 * np.eye(10),
                    d.T @ (effect * weight[:, None]),
                ).T
                change = j.T @ np.linalg.solve(
                    j @ j.T + 0.13 * np.eye(3), delta[anchor]
                )
                zz = z[anchor] + np.array([0.55, 1.0, 1.5])[:, None] * np.clip(
                    change, -0.62, 0.62
                )
                options.extend(
                    apply(rates, reference, self.stride, self.lo, self.hi, basis, zz)
                )
        if not options:
            return
        val, legal, *_ = self.trial(options, long=True)
        finite = np.flatnonzero(legal)
        self.diag["rate_searches"] += 1
        if not len(finite):
            return
        idx = finite[np.argsort(val[finite])[:5]]
        agreement, loss = self.uncertainty([options[j] for j in idx], val[idx])
        safe = np.flatnonzero(agreement >= 1)
        if not len(safe):
            return
        k = safe[np.argmin(loss[safe] - 1.7 * agreement[safe])]
        if (
            note.get("chosen")
            and agreement[k] <= note.get("robust", 0)
            and loss[k] + 0.3 >= note["score"]
        ):
            return
        self.best, self.tick, self.lease = options[idx[k]], tick, mpc.HORIZON
        self.next = tick + min(10, max(5, round(0.6 * mpc.HORIZON)))
        self.diag["rate_admitted"] += 1
        note.update(
            rate_refined=True,
            rate_robust=int(agreement[k]),
            rate_quality=float(loss[k]),
        )


def run(a):
    held = {c["id"] for c in _read(a.plan)[a.deck][: a.limit]}
    actors = []

    def bank(file, engine):
        obj = Carried(
            file, engine, practice=[mpc.ROOT / "tt37/camera-time-school.npz"], held=held
        )
        actors.append(obj)
        return obj

    with patch.object(mpc, "MPC", Secants), patch.object(mpc, "Fragments", bank):
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
    ck = a.out.with_suffix("").with_suffix(".npz")
    rows = actors[0].online
    np.savez_compressed(
        ck,
        states=[r[0] for r in rows],
        goals=[r[1] for r in rows],
        tags=[repr(r[2]) for r in rows],
    )
    d = _read(a.out)
    d.update(method="signed rate finite differences", checkpoint_sha256=_hash(ck))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
