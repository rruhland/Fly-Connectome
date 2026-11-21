"""Resolve the short optical velocity discontinuity at a physical table bounce."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_model_rollout import observed_ball
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import Station

GRAVITY = 9.81 / 30**2
ORIGIN_Z = 0.020


def calibrate(rows):
    samples = []
    for row in rows:
        views = dict(row["optical"])
        for t in views:
            if not all(k in views for k in range(t - 3, t + 4)):
                continue
            p = np.asarray([views[k] for k in range(t - 3, t + 4)])
            if not (
                -0.78 < p[3, 1] < -0.20 and 0.010 < p[3, 2] < 0.075 and np.max(np.diff(p[:, 1])) < -0.020
            ):
                continue
            d = np.diff(p[:, 2])
            if not (max(d[:3]) < 0.0 and min(d[3:]) > 0.0):
                continue
            # Find when the descending and ascending optical arcs reach tabletop height.
            x = np.arange(-3, 4.0)
            incoming = np.polyfit(x[:3], p[:3, 2] + 0.5 * GRAVITY * x[:3] ** 2, 1)
            outgoing = np.polyfit(x[4:], p[4:, 2] + 0.5 * GRAVITY * x[4:] ** 2, 1)
            tau = (outgoing[1] - incoming[1]) / (incoming[0] - outgoing[0])
            vz0 = incoming[0] - GRAVITY * tau
            vz1 = outgoing[0] - GRAVITY * tau
            effective_z = np.polyval(incoming, tau) - 0.5 * GRAVITY * tau * tau
            tangent = np.polyfit(x[4:], p[4:, 1], 1)[0] / np.polyfit(x[:3], p[:3, 1], 1)[0]
            if abs(tau) < 1.3 and -0.005 < effective_z < 0.055 and vz0 < 0 < vz1:
                samples.append([np.clip(-vz1 / vz0, 0.5, 1.0), np.clip(tangent, 0.65, 1.1), effective_z])
    default = np.array([0.85, 0.9, ORIGIN_Z])
    learned = (
        (3 * default + len(samples) * np.median(samples, axis=0)) / (3 + len(samples)) if samples else default
    )
    return dict(parameters=learned.tolist(), samples=len(samples))


def state(history, settings):
    now = history[-1][0]
    observed = [(t, np.asarray(p)) for t, p in history[-11:] if p is not None and now - t <= 10]
    if len(observed) < 5 or observed[-1][0] != now:
        return None
    ts, xyz = zip(*observed)
    t = np.asarray(ts) - now
    p = np.asarray(xyz)
    if p[-1, 1] > -0.24 or np.min(p[-4:, 2]) > 0.15 or np.max(np.diff(p[-5:, 1])) > -0.008:
        return None
    e, friction, height = settings
    # Fit a pre/post impact family on a sub-frame time grid using only past sightings.
    # Before an otherwise ambiguous rebound, a well observed descending arc determines
    # both the time-to-table and the upward speed on the other side of the impulse.
    tau = np.arange(max(-5.0, t[0] + 2), 1.36, 0.035)
    if not len(tau):
        return None
    delta = t[None, :] - tau[:, None]
    lift = np.where(delta < 0, delta, -e * delta)
    weights = np.linspace(0.7, 1, len(t))
    response = p[:, 2] - height + 0.5 * GRAVITY * delta**2
    speed = np.sum(response * lift * weights, axis=1) / np.maximum(
        1e-7, np.sum(lift * lift * weights, axis=1)
    )
    pred = height + speed[:, None] * lift - 0.5 * GRAVITY * delta**2
    mse = np.mean((p[:, 2] - pred) ** 2 * weights, axis=1)
    mse[(speed > -0.016) | (speed < -0.17)] = 1.0
    idx = np.argmin(mse)
    if mse[idx] > 0.009**2:
        return None
    at = tau[idx]
    # The ordinary estimator is accurate on a smooth, distant incoming arc.
    if at > 0.45 or at < -3.7:
        return None
    d = delta[idx]
    horizontal = np.where(d < 0, d, d * friction)
    X = np.column_stack((np.ones(len(d)), horizontal))
    origin, rate = np.linalg.lstsq(X, p[:, :2], rcond=None)[0]
    phase = 1.0 if at > 0 else friction
    elapsed = -at
    pos = np.r_[
        origin + elapsed * phase * rate,
        height + speed[idx] * (elapsed if at > 0 else -e * elapsed) - 0.5 * GRAVITY * elapsed**2,
    ]
    vel = np.r_[phase * rate, (speed[idx] if at > 0 else -e * speed[idx]) - GRAVITY * elapsed]
    if np.linalg.norm(pos - p[-1]) > 0.019:
        return None
    return pos, vel, at <= 0, at


class BouncePlayer(RallyPlayer):
    def __init__(self, law, settings):
        super().__init__(law)
        self.settings = settings

    def search(self, estimate, observation, target):
        corrected = state(self.history, self.settings)
        if corrected is None:
            return super().search(estimate, observation, target)
        p, v, after, tau = corrected
        obs = replace(observed_ball(self.history, (p, v), self.law), bounced=after)
        held = super().search((p, v), obs, target)
        if held:
            held[1]["optical_bounce_offset"] = float(tau)
        return held


def audit(source, settings):
    rows = _read(source)["rows"]
    errors = {}
    for row in rows:
        history = []
        for s in row["samples"]:
            est = s["estimates"]["stock"]
            p = est["position"]
            history.append((s["tick"], p))
            if not s["active"] or est["velocity"] is None:
                continue
            old = est["velocity"][2]
            new = state(history, settings)
            actual = s["velocity"][2]
            change = new is not None
            errors.setdefault(s["bounce_phase"], []).append(
                [abs(old - actual), abs((new[1][2] * 30 if change else old) - actual), change]
            )
    return {
        k: dict(
            n=len(v),
            adjusted=int(np.sum(np.array(v)[:, 2])),
            median=np.median(v, axis=0)[:2].tolist(),
            p90=np.quantile(v, 0.90, axis=0)[:2].tolist(),
        )
        for k, v in errors.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8351422)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    source = _read(args.practice)
    if source["privileged"]:
        raise ValueError("expected optical recording")
    fits = calibrate(source.get("practice", source.get("rows")))
    scores = audit(args.audit, fits["parameters"])
    result = dict(
        calibration=fits,
        optical_velocity=scores,
        spread=0.30,
        pace=0.82,
        rival="anchor",
        seed=args.seed,
        privileged=False,
        rows=[],
    )
    _write(args.output, result)
    print(fits, scores, flush=True)
    if not args.count:
        return
    station = Station(0.30, 0.82, "anchor")
    law = station.player.law
    station.player.close()
    station.player = BouncePlayer(law, fits["parameters"])
    try:
        for s in np.random.default_rng(args.seed).integers(2**31, size=args.count):
            row = station.point(int(s))
            result["rows"].append(row)
            _write(args.output, result)
            print(int(s), row["longest"], flush=True)
    finally:
        station.close()


if __name__ == "__main__":
    main()
