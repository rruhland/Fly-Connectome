"""Refine direct joint references by moving the face's interior toward a virtual contact."""

import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt38_motor_value import MotorValue


def interior(radius):
    return 3.2 * (np.maximum(0.0, radius - 0.027) / 0.040) ** 2


def adjustment(geometry, app, response, incoming, length):
    eta = app[0]
    delta = app[1:4] - app[4:7]
    if not np.isfinite(eta) or eta < 0.012 or np.linalg.norm(delta) > 0.25:
        return None
    _, n, jac, angle = geometry.pose(app[10:15], jacobian=True)
    flight = np.clip((0.70 - app[2]) / 3.9, 0.4, 0.72)
    target = (np.array([0.0, 0.70, 0.78]) - app[1:4]) / flight
    target[2] += 4.905 * flight
    impulse = target - incoming
    desired = impulse / max(0.1, np.linalg.norm(impulse))
    # Keep the ball on the incident face. Sliding the face beneath the ball leaves
    # the nominal impact time nearly unchanged, while reducing rim sensitivity.
    tangential = delta - n * (delta @ n)
    if app[18]:
        mismatch = target - response[2:5]
        tilt = (mismatch - n * (mismatch @ n)) / max(3.9, np.linalg.norm(impulse))
        forward = np.clip((mismatch @ n) * 0.38, -0.75, 1.1) * n
        shift = tangential
    else:
        tilt = desired - n
        forward = 0.28 * n
        shift = tangential + n * np.clip((delta @ n) - 0.025, -0.05, 0.05)
    mat = np.vstack([jac, 0.125 * angle])
    inverse = mat.T @ np.linalg.inv(mat @ mat.T + 0.019**2 * np.eye(6))
    slide = inverse @ np.r_[shift, np.zeros(3)]
    rotation = inverse @ np.r_[np.zeros(3), 0.125 * np.clip(tilt, -0.38, 0.38)]
    speed = inverse @ np.r_[forward, np.zeros(3)]
    t = np.arange(1, length + 1) * mpc.DT
    ramp = np.minimum(1.0, t / max(0.022, eta * 0.53))
    slope = np.clip(t - eta + 0.043, -0.065, 0.13) * ramp
    field = lambda x: np.c_[ramp[:, None] * np.clip(x, -0.46, 0.46), np.zeros(len(t))]
    return (
        field(slide),
        field(rotation),
        np.c_[slope[:, None] * np.clip(speed, -4.0, 4.0), np.zeros(len(t))],
    )


class Centered(mpc.MPC):
    def __init__(self, *args, judge, **kw):
        super().__init__(*args, **kw)
        self.judge = judge
        self.collect = False

    def trial(self, goals, *, long=False, traces=False):
        if not self.collect:
            return super().trial(goals, long=long, traces=traces)
        g = np.asarray(
            [mpc.tail(seq, mpc.HORIZON + (mpc.TAIL if long else 0)) for seq in goals]
        )
        f = np.zeros((len(g), 9))
        trace = np.full((*g.shape[:2], 17), np.nan) if traces else None
        approach = np.full((len(g), 19), np.nan)
        _, y = self.engine.predict.trajectories(
            g,
            mpc.AIM,
            mpc.HORIZON * mpc.DT,
            feedback=f,
            traces=trace,
            approaches=approach,
        )
        self.diag["native_candidates"] += len(g)
        loss, legal = mpc.cost(y, f)
        if legal.any():
            loss[legal] = (
                0.45 * loss[legal]
                + 0.55 * (self.judge.cost(y[legal]) - 5.0)
                + interior(y[legal, 12])
            )
        for seq, a, b, o, l, ok in zip(goals, approach, f, y, loss, legal, strict=True):
            if np.isfinite(a[0]):
                self.exposures.append((seq[: mpc.HORIZON], a, b, o, l, bool(ok)))
        return loss, legal, trace, y, f

    def search(self, tick, x, q, reference, base):
        self.collect = True
        self.exposures = []
        try:
            super().search(tick, x, q, reference, base)
            if not self.notes or self.notes[-1]["tick"] != tick:
                return
            note = self.notes[-1]
            cold = self.tick != tick or not note.get("chosen")
            incumbent = None if cold else self.best
            if not cold and note.get("robust") == 2:
                measurements = [
                    r
                    for r in self.exposures
                    if r[-1] and np.array_equal(r[0], incumbent)
                ]
                if measurements and max(r[3][12] for r in measurements) < 0.039:
                    return
            before = process_time()
            additions = []
            for stage in range(2):
                candidates = [r for r in self.exposures if r[1][0] > 0.014]
                if not candidates:
                    break
                selected = []
                for r in sorted(candidates, key=lambda item: item[4]):
                    if any(np.max(np.abs(r[0] - v[0])) < 0.024 for v in selected):
                        continue
                    selected.append(r)
                    if len(selected) >= 4:
                        break
                self.exposures = []
                tests = []
                for seq, a, f, _, _, _ in selected:
                    terms = adjustment(self.geometry, a, f, x[3:6], len(seq))
                    if terms is None:
                        continue
                    slide, rot, speed = terms
                    for u, v, w in [
                        (1, 0, 0),
                        (0.6, 0.35, 0),
                        (1, 0.7, 0.4),
                        (0.9, 1.1, 0.8),
                        (0.3, 1.3, 1.2),
                        (0.8, 0.65, -0.35),
                        (0.45, 0, 0.9),
                    ]:
                        tests.append(seq + u * slide + v * rot + w * speed)
                if not tests:
                    break
                _, valid, _, _, _ = self.trial(tests, long=True)
                additions.extend(r for r in self.exposures if r[-1])
                if valid.any():
                    break
            if additions:
                bets = sorted(additions, key=lambda r: r[4])[:6]
                options = [r[0] for r in bets]
                if incumbent is not None:
                    options.insert(0, incumbent)
                loss, legal, _, y, _ = self.trial(options, long=True)
                ids = np.flatnonzero(legal)
                if len(ids):
                    # Perturbed estimates check whether the change survives sensor error.
                    viable = [options[j] for j in ids]
                    agree, risk = self.uncertainty(viable, loss[ids])
                    rank = risk + 2.25 * (2 - agree)
                    chosen = int(np.argmin(rank))
                    j = ids[chosen]
                    if (incumbent is None or j != 0) and agree[chosen] >= 1:
                        self.best = options[j]
                        self.tick = tick
                        self.lease = mpc.LEASE
                        self.diag["interior_replacements"] += 1
                        note.update(
                            chosen="interior_translation",
                            robust=int(agree[chosen]),
                            center_radius=float(y[j, 12]),
                            replacement=int(not cold),
                        )
            self.diag["search_cpu_s"] += process_time() - before
        finally:
            self.collect = False


def run(a):
    judge = MotorValue(_read(a.value))
    cards = _read(a.plan)
    if a.deck != "development" and {c["id"] for c in cards[a.deck][: a.limit]} & set(
        _read(a.value)["release"]
    ):
        raise ValueError("Movement quality learned on evaluation releases")
    with patch.object(
        mpc, "MPC", lambda *args, **kw: Centered(*args, judge=judge, **kw)
    ):
        mpc.run(
            a.feeds,
            a.bank,
            a.plan,
            a.deck,
            "live",
            a.output,
            optic=a.optics,
            cap=200,
            limit=a.limit,
            variant="elastic",
        )
    d = _read(a.output)
    d["interior_search"] = _hash(Path(__file__))
    d["value"] = _hash(a.value)
    _write(a.output, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deck")
    p.add_argument("output", type=Path)
    p.add_argument("--plan", type=Path, default=mpc.PLAN)
    p.add_argument("--bank", type=Path, default=mpc.BANK)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument(
        "--value", type=Path, default=Path("artifacts/tt37/motor-value.json.gz")
    )
    p.add_argument("--optics", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
