"""Practice uncertain regions using diverse, previously scored motor corrections."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_motor_clinic import AIM, schedules
from .tt37_reuse import options

ROOT = Path("artifacts/tt37")
NAMES = [
    "reuse-cold",
    "reuse-warm",
    "reuse-within",
    "value-economy",
    "value-closed",
    "value-settled",
    "memory-transfer-carry",
    "memory-transfer-learn",
    "independent-cold-value",
    "independent-online-value",
    "independent-value",
    "rally-funded",
    "rally-cold",
    "mix-rollin",
    "mix-fallback-bank",
    "sensor-school",
    "stream-learn",
    "stream-blank",
    "hz60-sensor",
    "hz60-blank",
    "risk-warm",
    "repair-practice",
]
OBS_SCALE = np.r_[0.14, 0.052, 0.78, 0.80, 1.0, 0.85, 0.9, 0.9, 0.9, 0.9]
MOTOR_SCALE = np.array([0.23, 0.33, 0.38, 0.36, 0.5, 2.2, 3.0, 2.8, 3.0, 3.6])


def fit(out, plan, group):
    blocked = {r["id"] for r in _read(plan)[group]}
    rows = {}
    used = set()
    provenance = {}
    for nm in NAMES:
        path = ROOT / (nm + ".json.gz")
        if not path.exists():
            continue
        d = _read(path)
        if (
            not str(d.get("mode", "")).startswith("optical")
            or len(d["rows"]) != len(d["cards"])
            or d.get("camera_stride", 2) != 2
        ):
            continue
        count = 0
        for c, r in zip(d["cards"], d["rows"], strict=True):
            if c["id"] in blocked:
                continue
            near = [
                e["time"]
                for e in r["events"]
                if e["kind"] == "legal_return" and e["side"] == 1
            ]
            late = [e["tick"] for e in r.get("servo", []) if e["status"] == "corrected"]
            moves = r["records"]
            for i, m in enumerate(moves):
                tick = m["tick"]
                end = (
                    moves[i + 1]["tick"]
                    if i + 1 < len(moves)
                    else len(r["actions"]) + 1
                )
                if (
                    m.get("surface_shift", 0) != 0
                    or m.get("camera_rescue")
                    or m.get("impact_margin")
                    or end - tick < 23
                ):
                    continue
                if any(tick < t < min(end, tick + 22) for t in late):
                    continue
                if (m.get("late_repair") or {}).get("status") == "corrected":
                    continue
                if r.get("direct_fallback_ticks"):
                    continue
                success = any(tick / 60 < t < end / 60 for t in near)
                x = np.array(m["context"])
                u = np.array(m["action"])
                key = (
                    c["id"],
                    c["seed"],
                    c["reply_seed"],
                    *x.round(5),
                    *u.round(5),
                    tick,
                )
                rows[key] = (x.tolist(), u.tolist(), int(success))
                used.add(c["id"])
                count += 1
        if count:
            provenance[path.name] = _hash(path)
    x, u, y = zip(*rows.values(), strict=True)
    _write(
        out,
        {
            "x": x,
            "u": u,
            "y": y,
            "n": len(y),
            "positives": sum(y),
            "practice": sorted(used),
            "excluded": sorted(blocked),
            "sources": provenance,
        },
    )


class SkilledMemory:
    def __init__(self, path):
        d = _read(path)
        self.x = np.array(d["x"])
        self.u = np.array(d["u"])
        self.y = np.array(d["y"])
        self.scale = OBS_SCALE
        self.tree = cKDTree(self.x / self.scale)
        self.pos = np.flatnonzero(self.y)
        self.positive = cKDTree(self.x[self.pos] / self.scale)
        self.checks = self.active = 0

    def alternatives(self, x):
        state = x / self.scale
        d, ii = self.tree.query(state, min(44, len(self.x)))
        w = np.exp(-0.5 * (d / max(0.38, np.median(d))) ** 2)
        expected = (2.0 + w @ self.y[ii]) / (3 + w.sum())
        # Action changes are confined to places where similar movements struggled.
        if expected >= 0.80:
            return None, expected
        dist, ii = self.positive.query(state, min(96, len(self.pos)))
        ix = self.pos[np.atleast_1d(ii)]
        a = self.u[ix]
        wt = np.exp(-0.5 * (dist / max(0.38, np.median(dist[:12]))) ** 2)
        weight = wt / (max(1e-12, wt.sum()))
        mu = weight @ a
        centered = (self.x[ix] - x) / self.scale
        dx = centered - weight @ centered
        slope = np.linalg.solve(
            dx.T @ (weight[:, None] * dx) + 0.30 * np.eye(10),
            dx.T @ (weight[:, None] * (a - mu)),
        )
        local = mu - np.clip(
            (weight @ centered) @ slope, -0.5 * MOTOR_SCALE, 0.5 * MOTOR_SCALE
        )
        proposals = [local, mu]
        for i, item in enumerate(a):
            if min(np.linalg.norm((item - p) / MOTOR_SCALE) for p in proposals) < 0.17:
                continue
            proposals.extend([item, 0.55 * item + 0.45 * local])
            # Transfer the local generalization of this active motor branch to today's state.
            proposals.append(
                item
                - np.clip(centered[i] @ slope, -0.38 * MOTOR_SCALE, 0.38 * MOTOR_SCALE)
            )
            if len(proposals) >= 52:
                break
        return np.stack(proposals), expected

    def propose(self, engine, bank, x, fit, eta, dt, home):
        self.checks += 1
        previous = options(bank, x)
        extra, expected = self.alternatives(x)
        if extra is not None:
            self.active += 1
            actions = np.vstack([previous, extra])
        else:
            actions = previous
        f = np.zeros((len(actions), 9))
        goals = schedules(*fit, eta, actions, dt, home, round((eta + 1.05) / dt))
        _, y = engine.trajectories(goals, AIM, eta + 0.12, feedback=f)
        ok = np.flatnonzero(np.isfinite(y[:, 0]))
        note = {
            "trials": len(actions),
            "legal": len(ok),
            "memory": len(bank),
            "skill_support": expected,
            "extra": 0 if extra is None else len(extra),
        }
        if not len(ok):
            return None, None, note
        model = getattr(engine, "motor_value", None)
        cost = y[ok, 0] if model is None else model.cost(y[ok])
        if extra is not None:
            # Only in hard contexts, favor a strike with a margin to both rim and net.
            cost += 2.6 * (
                np.maximum(0.0, y[ok, 12] - 0.032) / 0.045
            ) ** 2 + 12 * np.maximum(0.0, 0.052 - y[ok, 5])
        j = ok[np.argmin(cost)]
        note["skill_selected"] = bool(j >= len(previous))
        return actions[j], y[j].tolist(), note


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("plan", type=Path)
    p.add_argument("group")
    a = p.parse_args()
    fit(a.output, a.plan, a.group)
