"""Learn the distribution of the partner's reply after watching our own ball rebound."""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve

from . import tt50_opponent_vision as video
from . import tt51_afterstate as response
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt94_optical_rebound")
G = 9.81
SURFACE = 0.778


def estimate(samples, after):
    """Use only samples already delivered; back/forward-project to the table plane."""
    estimates = []
    for _, state in samples:
        z, vz = state[2], state[5]
        disc = vz * vz + 2 * G * (z - SURFACE)
        if disc <= 0:
            continue
        u = (vz + (-1 if after else 1) * np.sqrt(disc)) / G
        if abs(u) > 0.32:
            continue
        xyz = state[:2] + u * state[3:5]
        estimates.append(np.r_[xyz, state[3:5], vz - G * u])
    return np.median(estimates, axis=0) if estimates else None


class Rebound:
    def __init__(self):
        self.before = []
        self.after = []
        self.ready = False

    def update(self, tick, state):
        if (
            self.ready
            or state is None
            or not np.isfinite(state).all()
            or state[4] < 0.8
        ):
            return None
        if state[2] < 1.16 and state[5] < -0.9 and state[1] > -0.15:
            self.before = (self.before + [(tick, state.copy())])[-3:]
            self.after = []
        if not self.before or state[5] < 0.5 or tick - self.before[-1][0] > 12:
            return None
        self.after.append((tick, state.copy()))
        if len(self.after) < 2 or len(self.before) < 2:
            return None
        pre, post = estimate(self.before, False), estimate(self.after, True)
        if pre is None or post is None:
            return None
        self.ready = True
        return np.r_[pre, post]


class Model:
    def __init__(self, x, y):
        self.center = np.median(x, axis=0)
        self.scale = np.maximum(x.std(0), 0.035)
        self.u = np.clip((x - self.center) / self.scale, -5, 5)
        self.average = y.mean(0)
        self.coef = solve(
            self.kernel(self.u) + 0.32 * np.eye(len(x)),
            y - self.average,
            assume_a="pos",
        )

    def kernel(self, q):
        dist = cdist(q, self.u, "sqeuclidean") / self.u.shape[1]
        return np.exp(-0.60 * dist) + 0.20 * np.exp(-2.0 * dist)

    def predict(self, x):
        return (
            self.average
            + self.kernel(np.clip((x - self.center) / self.scale, -5, 5)) @ self.coef
        )


def packet(path, cohort, held):
    with np.load(path) as z:
        release = int(z["id"])
        tick, stream, ball, label, contacts = (
            z[k] for k in ("tick", "stream", "ball", "label", "contact")
        )
    examples = []
    for i, y in enumerate(label):
        # Difficult replays were chosen via their first reply, so exclude that draw.
        if (
            (cohort == "difficult" and i == 0)
            or not np.isfinite(y).all()
            or y[4] > -0.9
        ):
            continue
        eye = Rebound()
        for j in np.flatnonzero(stream == i):
            feat = eye.update(tick[j], ball[j])
            if feat is None:
                continue
            if tick[j] >= contacts[i]:
                raise ValueError("Forecast after receiver acted")
            examples.append((feat, response.projected(y), (contacts[i] - tick[j]) / 60))
            break
    if not examples:
        return
    x, y, lead = map(np.asarray, zip(*examples, strict=True))
    return dict(
        id=release,
        file=str(path),
        cohort=cohort,
        held=held,
        x=x[0].tolist(),
        features=x.tolist(),
        heights=y[:, 0].tolist(),
        y=y[:, 0].mean().item(),
        risk=np.mean(y[:, 0] < 0.99).item(),
        lead=lead.tolist(),
        n=len(y),
        hash=_hash(path),
    )


def study():
    root = video.ROOT
    roster = _read(root / "difficult-plan.json")["cases"]
    held = {c["id"] for c in roster if c["split"] == "held"}
    files = sorted((root / "observations").glob("case-*.npz"))
    for p in files[16:]:
        with np.load(p) as z:
            held.add(int(z["id"]))
    coverage = {}
    rows = []
    for group in ("observations", "difficult"):
        for p in sorted((root / group).glob("case-*.npz")):
            with np.load(p) as z:
                release = int(z["id"])
                labels = z["label"][(1 if group == "difficult" else 0) :]
                role = group if release in held else "train-" + group
                n = coverage.setdefault(role, dict(traces=0, labeled=0, forecast=0))
                n["traces"] += len(labels)
                n["labeled"] += int(
                    (np.isfinite(labels).all(1) & (labels[:, 4] < -0.9)).sum()
                )
            r = packet(p, group, release in held)
            if r:
                rows.append(r)
                n["forecast"] += r["n"]
    train = [r for r in rows if not r["held"]]
    trainx = np.array([r["x"] for r in train])
    trainy = np.array([[r["y"], r["risk"]] for r in train])
    reports = {}
    for role, pool in [
        ("all", [r for r in rows if r["held"]]),
        ("ordinary", [r for r in rows if r["held"] and r["cohort"] == "observations"]),
        ("difficult", [r for r in rows if r["held"] and r["cohort"] == "difficult"]),
    ]:
        x = np.concatenate([r["features"] for r in pool])
        groups = np.repeat(np.arange(len(pool)), [r["n"] for r in pool])
        y = np.array([[r["y"], r["risk"]] for r in pool])
        out = {}
        for stage, idx in [
            ("prior", None),
            ("before", slice(0, 5)),
            ("after", slice(None)),
        ]:
            pred = (
                np.tile(trainy.mean(0), (len(x), 1))
                if idx is None
                else Model(trainx[:, idx], trainy).predict(x[:, idx])
            )
            pred = np.array([pred[groups == i].mean(0) for i in range(len(pool))])
            out[stage] = dict(
                height_mae_mm=float(np.mean(abs(y[:, 0] - pred[:, 0])) * 1000),
                height_corr=float(np.corrcoef(y[:, 0], pred[:, 0])[0, 1])
                if idx is not None
                else None,
                low_probability_rmse=float(
                    np.sqrt(np.mean((y[:, 1] - pred[:, 1]) ** 2))
                ),
            )
        reports[role] = dict(contexts=len(pool), draws=sum(r["n"] for r in pool), **out)
    ROOT.mkdir(exist_ok=True)
    _write(
        ROOT / "optical.json",
        dict(
            rows=rows,
            reports=reports,
            train=len(train),
            coverage=coverage,
            train_draws=sum(r["n"] for r in train),
            held_ids=sorted(held),
            median_seconds_before_opponent=float(
                np.median(np.concatenate([r["lead"] for r in rows]))
            ),
        ),
    )
    print(reports, flush=True)


if __name__ == "__main__":
    study()
