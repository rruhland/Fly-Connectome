"""Estimate the approaching ball's receiving zone before opponent contact using stereo and past rewards."""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist

from . import tt46_continuous_season as season
from . import tt48_practice_transfer as transfer
from . import tt50_forecast_episodes as clips
from . import tt51_flight_risk as risk
from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt54_earned_low")
MODEL = ROOT / "court-forecast.npz"
SOURCE = Path("artifacts/tt48_transfer/live-plain.json.gz")


def targets(y):
    s = risk.sensor_future(np.atleast_2d(y))
    x, z = s[:, 5], s[:, 6]
    low = z < 1.0
    return np.c_[x, z, low & (x < -0.10), low & (abs(x) <= 0.10), low & (x > 0.10)]


def benchmark(m, points, source):
    by_gate = {g: [] for g in risk.GATES}
    old = _read(source)
    for p in points:
        doc = _read(p)
        r = old["rows"][doc["index"]]
        if digest_json(r["events"]) != doc["point_events"]:
            raise ValueError("Wrong physical assessment point")
        offers = [o for o in r["placements"]["delivered"] if o.get("far_contact")]
        far = np.array([o["far_contact"]["time"] * 60 for o in offers])
        for leg in doc["observed"]:
            j = int(np.searchsorted(far, leg["tick"]) - 1)
            if j < 0 or not offers[j].get("receive_line"):
                continue
            line = offers[j]["receive_line"]
            native = targets(np.r_[np.array(line["xyz"])[[0, 2]], line["velocity"]])[0]
            for d in leg["decisions"]:
                early = (far[j] - d["tick"]) / 60
                if not 0.05 < early < 1.9:
                    continue
                g = d["gate"]
                f = np.array(d["x"])[None, :]
                camera, ball = m[g]
                out = camera.predict(f)[0]
                mu, scale, X, C, b = ball
                u = np.clip((f[:, :6] - mu) / scale, -5, 5)
                compare = np.exp(-cdist(u, X, "sqeuclidean") / 6) @ C + b
                by_gate[g].append((native, out, compare[0], early))
    report = {}
    for g, rows in by_gate.items():
        if not rows:
            continue
        y, vision, ball, t = map(np.asarray, zip(*rows, strict=True))

        def measure(pred):
            out = {}
            for col, label in enumerate(("left", "center", "right"), 2):
                flag = y[:, col] > 0
                order = np.argsort(-pred[:, col], kind="stable")
                ranks = {}
                for proportion in (0.10, 0.25):
                    k = int(np.ceil(len(flag) * proportion))
                    ranks[str(proportion)] = [int(flag[order[:k]].sum()), k]
                out[label] = dict(
                    cases=int(flag.sum()),
                    risk_selections=ranks,
                    brier=float(np.mean((np.clip(pred[:, col], 0, 1) - flag) ** 2)),
                )
            return out

        report[str(g)] = dict(
            n=len(y),
            median_preimpact_s=float(np.median(t)),
            camera=measure(vision),
            outgoing_only=measure(ball),
        )
    return report


def fit():
    reserved = {
        c["id"] for k in ("practice", "assessment") for c in _read(season.PLAN)[k]
    }
    reserved.update(c["id"] for c in _read(transfer.PLAN)["practice"])
    rows = [r for r in risk.load() if not r[1] and r[4] not in reserved]
    models = {}
    arrays = {}
    info = {}
    for i, g in enumerate(risk.GATES):
        use = [r for r in rows if r[0] == g]
        X = np.array([r[5] for r in use])
        Y = targets(np.array([r[6] for r in use]))
        w = np.array([0.55 if r[2] else 1.0 for r in use])
        camera = risk.Kernel.fit(X, Y, w)
        native = X[:, :6]
        mu = np.median(native, axis=0)
        scale = np.maximum(np.percentile(abs(native - mu), 75, axis=0), 0.001)
        u = np.clip((native - mu) / scale, -5, 5)
        b = np.average(Y, axis=0, weights=w)
        C = solve(
            np.exp(-cdist(u, u, "sqeuclidean") / 6) + np.diag(0.22 / w),
            Y - b,
            assume_a="pos",
        )
        ball = (mu, scale, u, C, b)
        models[g] = camera, ball
        info[g] = dict(
            samples=len(Y),
            releases=len({r[4] for r in use}),
            positives=Y[:, 2:].sum(0).tolist(),
        )
        for key, vals in (("camera", camera.data), ("ball", ball)):
            for j, a in enumerate(vals):
                arrays[f"{i}_{key}_{j}"] = a
    np.savez_compressed(MODEL, **arrays)
    data = dict(
        model=_hash(MODEL),
        training=info,
        excluded_release_ids=sorted(reserved),
        official=benchmark(
            models,
            sorted((risk.ROOT / "untrained-episodes").glob("game-*.json.gz")),
            season.ROOT / "learning-24.json.gz",
        ),
    )
    _write(ROOT / "court-forecast.json", data)


def fresh():
    points = sorted((risk.ROOT / "court-confirmation").glob("game-*.json.gz"))
    if len(points) != 32:
        raise ValueError("Incomplete next-course RGB sequence")
    with np.load(MODEL) as z:
        models = {
            g: (
                risk.Kernel(tuple(z[f"{i}_camera_{j}"] for j in range(6))),
                tuple(z[f"{i}_ball_{j}"] for j in range(5)),
            )
            for i, g in enumerate(risk.GATES)
        }
    _write(
        ROOT / "court-confirmation.json",
        dict(model=_hash(MODEL), scores=benchmark(models, points, SOURCE)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "capture", "fresh"))
    a = p.parse_args()
    if a.mode == "fit":
        fit()
    elif a.mode == "fresh":
        fresh()
    else:
        from .tt54_training_speedups import enabled

        with enabled():
            clips.collect(SOURCE, "court-confirmation", 0, 32)
