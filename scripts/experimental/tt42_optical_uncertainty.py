"""Learn uncertainty of each optical approach from subsequent images of the same flight."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt39_carry_joints as carry
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_rebound import ReboundReadout, fit

SCALE = np.array([0.17, 0.24, 0.080, 0.85, 0.85, 1.5])
LIMIT = np.array([0.014, 0.022, 0.020, 0.21, 0.25, 0.31])
NORM = np.array([0.002, 0.004, 0.003, 0.035, 0.065, 0.08])
PRIOR = Path("artifacts/tt37/visual-observations-development.npz")
OPTIC = Path("artifacts/tt37/rebound.npz")


class ForecastErrors:
    """Hindsight labels are reconstructed only when four further RGB observations exist."""

    def __init__(self, x=(), delta=()):
        self.states = deque(x, maxlen=2400)
        self.errors = deque(delta, maxlen=2400)
        self.reset()

    def reset(self):
        self.window = deque(maxlen=5)
        self.last = None

    def observe(self, t, point, x):
        if self.last is not None and t - self.last < 0.001 and t >= self.last:
            return
        if self.last is None or not 0.030 < t - self.last < 0.036:
            self.window.clear()
        self.last = t
        if point is None or not np.isfinite(point).all():
            self.window.clear()
            return
        self.window.append((t, point, x))
        if len(self.window) < 5:
            return
        times, p, states = zip(*self.window, strict=True)
        source = states[0]
        if source is None or source[4] > -0.95 or not -1.6 < source[1] < 0.1:
            return
        tt = np.asarray(times) - times[0]
        pp = np.asarray(p)
        pos, vel, acc = fit(tt, pp)
        residual = pp - pos - tt[:, None] * vel - 0.5 * tt[:, None] ** 2 * acc
        # Reject windows containing a paddle/table impulse, occlusions and false blobs.
        if np.max(np.linalg.norm(residual, axis=1)) > 0.004 or vel[1] > -1.2:
            return
        self.states.append(np.asarray(source).copy())
        self.errors.append(np.clip(np.r_[pos, vel] - source, -LIMIT, LIMIT))

    def scenarios(self, state):
        if len(self.states) < 12:
            return None
        x = np.asarray(self.states)
        d = np.sum(((x - state) / SCALE) ** 2, axis=1)
        k = min(32, len(d))
        ids = np.argpartition(d, k - 1)[:k]
        if np.min(d[ids]) > 5:
            return None
        w = np.exp(-0.55 * d[ids])
        if w.sum() < 2:
            return None
        w /= w.sum()
        examples = np.asarray(self.errors)[ids]
        mean = w @ examples
        white = (examples - mean) / NORM
        cov = (white * w[:, None]).T @ white
        eig, vec = np.linalg.eigh(cov)
        axis = vec[:, -1] * np.sqrt(max(0, eig[-1])) * NORM
        return np.clip(np.stack([mean - axis, mean + axis]), -LIMIT, LIMIT)


def train(out):
    tracker = ForecastErrors()
    with np.load(PRIOR) as z:
        sensors = z["sensor"]
        groups = z["groups"]
        last = None
        for row, group in zip(sensors, groups, strict=True):
            if last is None or not np.array_equal(group, last):
                tracker.reset()
                filter_ = ReboundReadout(OPTIC)
            last = group
            current = row[4:10] if np.isfinite(row[4:10]).all() else None
            point = row[1:4] if np.isfinite(row[1:4]).all() else None
            state = filter_.update(row[0], point, current)
            tracker.observe(row[0], point, state)
    np.savez_compressed(
        out,
        states=tracker.states,
        errors=tracker.errors,
        practice_ids=np.unique(groups[:, 0]),
    )
    _write(
        out.with_suffix(".json"),
        {
            "samples": len(tracker.states),
            "camera_source": _hash(PRIOR),
            "optics": _hash(OPTIC),
            "checkpoint": _hash(out),
        },
    )


class Risk(mpc.MPC):
    def __init__(self, *args, belief, **kwargs):
        super().__init__(*args, **kwargs)
        self.belief = belief
        self.x = None

    def begin(self):
        super().begin()
        self.belief.reset()

    def step(self, tick, eye, x, q, reference, base):
        points = eye.filter.track
        if points and abs(points[-1][0] - tick * mpc.DT) < 1e-5:
            self.belief.observe(points[-1][0], points[-1][1], x)
        self.x = x
        return super().step(tick, eye, x, q, reference, base)

    def uncertainty(self, goals, nominal):
        scenarios = self.belief.scenarios(self.x)
        if scenarios is None:
            return super().uncertainty(goals, nominal)
        self.diag["learned_uncertainty_calls"] += 1
        t = self.engine.twin
        p, v = t.ball.copy(), t.velocity.copy()
        spin = t.data.qvel[t._ball_dof + 3 : t._ball_dof + 6]
        original = spin.copy()
        outcomes = []
        try:
            for n, delta in enumerate(scenarios):
                t.ball[:] = p + delta[:3]
                t.velocity[:] = v + delta[3:]
                spin[:] = original + [-1 if n == 0 else 33, 0, 0]
                loss, legal, *_ = self.trial(goals, long=True)
                outcomes.append((loss, legal))
        finally:
            t.ball[:] = p
            t.velocity[:] = v
            spin[:] = original
        agreement = np.sum([legal for _, legal in outcomes], axis=0)
        risk = np.mean(
            [
                np.where(legal, loss, 2.75 + 0.12 * np.minimum(loss, 10))
                for loss, legal in outcomes
            ],
            axis=0,
        )
        return agreement, 0.43 * nominal + 0.57 * risk


def run(a):
    with np.load(a.model) as z:
        belief = ForecastErrors(z["states"], z["errors"])
        ids = set(z["practice_ids"])
    held = {c["id"] for c in _read(a.plan)[a.deck][: a.limit]}
    if held & ids:
        raise ValueError("Optical uncertainty trained on assessment release")

    def actor(*args, **kwargs):
        return Risk(*args, belief=belief, **kwargs)

    with patch.object(mpc, "MPC", actor):
        carry.run(a)
    doc = _read(a.out)
    doc.update(
        optical_uncertainty_sha256=_hash(a.model),
        error_samples_at_end=len(belief.states),
    )
    state = a.out.with_name(a.out.name.split(".")[0] + "-observer.npz")
    np.savez_compressed(state, states=belief.states, errors=belief.errors)
    doc["observer_sha256"] = _hash(state)
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "play"))
    p.add_argument("out", type=Path)
    p.add_argument(
        "--model", type=Path, default=Path("artifacts/tt42/optical-uncertainty.npz")
    )
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        practice=[mpc.ROOT / "tt37/camera-time-school.npz"], late=False, capacity=768
    )
    a = p.parse_args()
    if a.mode == "train":
        train(a.out)
    else:
        run(a)
