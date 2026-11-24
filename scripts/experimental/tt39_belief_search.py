"""Search visual uncertainty when every joint movement fails in the nominal private model."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _write
from .tt37_motor_clinic import AIM, landing_error
from .tt37_rebound import ReboundReadout, accel
from .tt37_visual_inertia import future_velocity

SCALE = np.array([0.22, 0.28, 0.12, 0.72, 1.0, 1.3])


def train(source, optics, output):
    with np.load(source) as z:
        s = z["sensor"]
        groups = z["groups"]
        dt = float(z["period"]) if "period" in z else 1 / 30
    estimator = ReboundReadout(optics)
    x = []
    residual = []
    for i, r in enumerate(s):
        if i == 0 or not np.array_equal(groups[i], groups[i - 1]):
            estimator.samples.clear()
            estimator.last_time = None
        base = r[4:10] if np.isfinite(r[4:10]).all() else None
        state = estimator.update(r[0], r[1:4], base)
        if (
            state is None
            or state[4] > -0.9
            or not -1.72 < state[1] < 0.5
            or i + 4 > len(s)
        ):
            continue
        ts = s[i : i + 4, 0]
        pos = s[i : i + 4, 1:4]
        if not np.all(groups[i : i + 4] == groups[i]) or not np.allclose(
            np.diff(ts), dt
        ):
            continue
        v = future_velocity(pos, dt)
        if v is None or v[1] > -0.9:
            continue
        t = (ts - ts[0])[:, None]
        p = np.mean(pos - t * v - 0.5 * t * t * accel(v), axis=0)
        change = np.r_[p, v] - state
        if np.linalg.norm(change[:3]) > 0.05 or np.linalg.norm(change[3:]) > 1.65:
            continue
        x.append(state.copy())
        residual.append(change)
    np.savez_compressed(output, x=np.stack(x), residual=np.stack(residual))
    _write(
        output.with_suffix(".json"),
        {
            "frames": len(x),
            "source": _hash(source),
            "optics": _hash(optics),
            "policy_signals": "Triangulated stereo, filtered current image state, three later images used only for practice.",
            "residual_quantiles": np.quantile(
                residual, [0.1, 0.5, 0.9], axis=0
            ).tolist(),
            "sha256": _hash(output),
        },
    )


class BeliefSearch:
    def __init__(self, path):
        with np.load(path) as z:
            self.residual = z["residual"]
            self.tree = cKDTree(z["x"] / SCALE)
        self.examples = []

    def start(self):
        self.examples = []

    def observe(self, choices, goals, feedback):
        for u, g, f in zip(choices, goals, feedback, strict=True):
            self.examples.append((u, g, f))

    def shifts(self, state):
        _, ix = self.tree.query(state / SCALE, min(len(self.residual), 65))
        cloud = self.residual[np.atleast_1d(ix)]
        # Uncertainty in vertical arrival and progress along the flight dominate
        # short-horizon contact. Preserve correlations by drawing actual observed residuals.
        futures = cloud[:, :3] + 0.13 * cloud[:, 3:]
        rows = []
        for axis, levels in [(2, [0.06, 0.25, 0.5, 0.75, 0.94]), (1, [0.12, 0.88])]:
            order = np.argsort(futures[:, axis])
            for q in levels:
                rows.append(cloud[order[round(q * (len(order) - 1))]])
        return np.unique(rows, axis=0)

    def select(self, engine, eta):
        if not self.examples:
            return None
        f = np.asarray([r[2] for r in self.examples])
        contacted = f[:, 1] > 0.5
        cost = np.where(
            contacted, np.minimum(landing_error(f), 10.0), 10 + f[:, 0] * 12
        )
        ids = []
        for i in np.r_[np.argsort(cost), np.argsort(f[:, 0])[:4]]:
            if not any(
                np.max(np.abs(self.examples[j][0] - self.examples[i][0])) < 0.035
                for j in ids
            ):
                ids.append(i)
            if len(ids) >= 20:
                break
        actions = np.array([self.examples[j][1] for j in ids])
        t = engine.twin
        p0, v0 = t.ball.copy(), t.velocity.copy()
        deltas = self.shifts(np.r_[p0, v0])
        scores = []
        contacts = 0
        try:
            for d in deltas:
                t.ball[:] = p0 + d[:3]
                t.velocity[:] = v0 + d[3:]
                response = np.zeros((len(actions), 9))
                _, y = engine.trajectories(actions, AIM, eta + 0.12, feedback=response)
                contacts += int(np.sum(response[:, 1] > 0.5))
                scores.append(y)
        finally:
            t.ball[:] = p0
            t.velocity[:] = v0
        y = np.array(scores)
        ok = np.isfinite(y[:, :, 0])
        counts = ok.sum(axis=0)
        note = {
            "trials": len(ids) * len(deltas),
            "legal": int(ok.sum()),
            "hypotheses": len(deltas),
            "max_support": int(counts.max()),
            "contacts": contacts,
            "chosen": False,
        }
        plausible = np.flatnonzero(counts >= max(2, 0.36 * len(deltas)))
        if not len(plausible):
            return None, None, note
        rank = []
        for i in plausible:
            valid = y[ok[:, i], i, :]
            model = getattr(engine, "motor_value", None)
            quality = np.mean(model.cost(valid) if model else valid[:, 0])
            rank.append(quality - 1.7 * counts[i])
        winner = plausible[np.argmin(rank)]
        good = y[ok[:, winner], winner, :]
        representative = good[np.argsort(good[:, 0])[len(good) // 2]]
        note.update(chosen=True, support=int(counts[winner]))
        return self.examples[ids[winner]][0].copy(), representative.tolist(), note


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("optics", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    train(a.source, a.optics, a.output)
