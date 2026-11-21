"""Estimate incoming velocity by fitting an aerodynamic free-flight arc to fixed-camera RGB."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_model_rollout import observed_ball
from .tt22_bounce_filter import state as bounce_state
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import Station


def acceleration(v, spin):
    return (
        -0.17345556 * np.linalg.norm(v, axis=-1, keepdims=True) * v
        + [0.0, 0.0, -9.81]
        + 0.0003220273 * np.cross(spin, v)
    )


def arc(history, spin):
    now = history[-1][0]
    samples = [(t, np.asarray(p)) for t, p in history[-21:] if p is not None and now - t <= 18]
    if len(samples) < 5 or samples[-1][0] != now:
        return None
    ts, xyz = zip(*samples)
    ts = np.asarray(ts)
    p = np.stack(xyz)
    d = np.diff(p, axis=0) / np.diff(ts)[:, None]
    # Use a single optical arc. Discontinuities of either sign can mark a surface or swing.
    start = 0
    for j in range(1, len(d)):
        if (
            d[j - 1, 1] * d[j, 1] < -0.0003
            or (p[j, 2] < 0.10 and (d[j, 2] - d[j - 1, 2]) > 0.035)
            or ts[j + 1] - ts[j] > 3
        ):
            start = j
    ts, p = ts[start:], p[start:]
    if len(ts) < 5 or ts[-1] - ts[0] < 5:
        return None
    baseline = (ts - now) / 30.0
    X = np.column_stack((np.ones(len(ts)), baseline))
    wt = np.exp(baseline * 0.7)
    inv = np.linalg.pinv(X * wt[:, None]) * wt
    z = np.array([0.0, 0.0, -9.81])
    center, velocity = inv @ (p - 0.5 * baseline[:, None] ** 2 * z)
    for _ in range(4):
        # Third-order local aerodynamic position, including the changing drag direction.
        a = acceleration(velocity, spin)
        speed = max(1e-6, np.linalg.norm(velocity))
        jerk = -0.17345556 * (speed * a + velocity * np.dot(velocity, a) / speed) + 0.0003220273 * np.cross(
            spin, a
        )
        drift = 0.5 * baseline[:, None] ** 2 * a + baseline[:, None] ** 3 * jerk / 6
        center, velocity = inv @ (p - drift)
    residual = np.linalg.norm(X @ np.stack((center, velocity)) + drift - p, axis=1)
    if np.median(residual) > 0.009 or residual[-1] > 0.020 or velocity[1] > -0.35:
        return None
    return center, velocity / 30.0


def observe(history, settings, spin):
    rebound = bounce_state(history, settings)
    if rebound is not None:
        return rebound[:2]
    return arc(history, spin)


class ArcPlayer(RallyPlayer):
    def __init__(self, law, settings):
        super().__init__(law)
        self.settings = settings

    def search(self, estimate, observation, target):
        # A centered spin prior on a smooth reply serves as a weak curvature prior.
        # Keep the existing impact-spin estimate for ranking the candidate hits.
        fitted = observe(self.history, self.settings, observation.spin if not self.shots else np.zeros(3))
        if fitted is not None:
            pos, vel = fitted
            revised = observed_ball(self.history, fitted, self.law)
            observation = replace(observation, center=pos, velocity=30 * vel, bounced=revised.bounced)
            estimate = fitted
        return super().search(estimate, observation, target)


def diagnose(rows, settings):
    errors = {}
    for row in rows:
        hist = []
        for item in row["samples"]:
            est = item["estimates"]["stock"]
            hist.append((item["tick"], est["position"]))
            if not item["active"] or est["velocity"] is None:
                continue
            old = np.array(est["velocity"])
            fitted = observe(hist, settings, np.zeros(3))
            new = old if fitted is None else fitted[1] * 30
            true = np.array(item["velocity"])
            errors.setdefault(item["bounce_phase"], []).append(
                [np.linalg.norm(old - true), np.linalg.norm(new - true)]
            )
    return {
        k: dict(n=len(v), median=np.median(v, axis=0).tolist(), p90=np.quantile(v, 0.90, axis=0).tolist())
        for k, v in errors.items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=8351422)
    p.add_argument("--count", type=int, default=5)
    args = p.parse_args()
    settings = _read(args.calibration)["calibration"]["parameters"]
    diag = diagnose(_read(args.audit)["rows"], settings)
    print(diag, flush=True)
    data = dict(
        calibration=settings,
        diagnostic=diag,
        rows=[],
        spread=0.30,
        pace=0.82,
        rival="anchor",
        seed=args.seed,
        privileged=False,
    )
    _write(args.output, data)
    if not args.count:
        return
    station = Station(0.30, 0.82, "anchor")
    law = station.player.law
    station.player.close()
    station.player = ArcPlayer(law, settings)
    try:
        for seed in np.random.default_rng(args.seed).integers(2**31, size=args.count):
            row = station.point(int(seed))
            data["rows"].append(row)
            _write(args.output, data)
            print(int(seed), row["longest"], flush=True)
    finally:
        station.close()


if __name__ == "__main__":
    main()
