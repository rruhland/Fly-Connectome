"""Use scored motor experience to spend fewer simulations on familiar sensory contexts."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_motor_clinic import AIM, SCALE, schedules


def options(bank, x):
    if not bank:
        return np.zeros((1, 10))
    states = np.asarray([r["x"] for r in bank])
    values = np.asarray([r["change"] for r in bank])
    distance = np.linalg.norm((states - x) / SCALE, axis=1)
    idx = np.argsort(distance)[:12]
    raw = values[idx]
    # Preserve discrete local solutions, with nearby interpolation as extra proposals.
    return np.vstack([raw, 0.65 * raw + 0.35 * raw[0], np.zeros(10)])


def propose(engine, bank, x, fit, eta, dt, home):
    vals = options(bank, x)
    goals = schedules(*fit, eta, vals, dt, home, round((eta + 1.05) / dt))
    f = np.zeros((len(vals), 9))
    _, outcomes = engine.trajectories(goals, AIM, eta + 0.12, feedback=f)
    legal = np.flatnonzero(np.isfinite(outcomes[:, 0]))
    report = {
        "trials": len(vals),
        "legal": len(legal),
        "contacts": int(f[:, 1].sum()),
        "memory": len(bank),
    }
    if not len(legal):
        return None, None, report
    model = getattr(engine, "motor_value", None)
    cost = outcomes[legal, 0] if model is None else model.cost(outcomes[legal])
    i = legal[np.argmin(cost)]
    return vals[i], outcomes[i].tolist(), report


def pooled(paths, out):
    seen = set()
    rows = []
    sources = {}
    for path in paths:
        d = _read(path)
        if d["mode"] != "optical-learn" or len(d["cards"]) != len(d["rows"]):
            raise ValueError(
                "Only complete physical sensor-based learning points qualify"
            )
        for r in d["bank"]:
            key = tuple(r["x"] + r["change"])
            if key not in seen:
                seen.add(key)
                rows.append(r)
        sources[path.name] = _hash(path)
    _write(out, {"bank": rows, "source_sha256": sources})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    a = p.parse_args()
    pooled(a.sources, a.out)
