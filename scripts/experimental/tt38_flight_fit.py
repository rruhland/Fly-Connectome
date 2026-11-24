"""Estimate short unbroken flight segments and visible bounce impulses from stereo."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_arena import BALL_INERTIA, BALL_MASS, BALL_RADIUS, SPIN_DECAY
from .tt4_transfer_trial import _write
from .tt37_rebound import ReboundReadout, accel
from .tt37_visual_inertia import score


def smooth(times, positions):
    """Fit velocity and acceleration, shrinking curvature toward nominal free flight."""
    t = times - times.mean()
    span = times[-1] - times[0]
    slope = (t @ positions) / (t @ t)
    a0 = accel(slope)
    c = 0.5 * (t * t - np.mean(t * t))
    c -= t * (t @ c) / (t @ t)
    # A 1.1 mm centroid error gives weak acceleration evidence in very short arcs.
    da = c @ (positions - 0.5 * t[:, None] ** 2 * a0) / (c @ c + 0.0011**2)
    acceleration = a0 + np.clip(da, -3.0, 3.0)
    u = positions - 0.5 * t[:, None] ** 2 * acceleration
    v = t @ u / (t @ t)
    residual = positions - (
        u.mean(axis=0) + t[:, None] * v + 0.5 * t[:, None] ** 2 * acceleration
    )
    p = u.mean(axis=0) + t[-1] * v + 0.5 * t[-1] ** 2 * acceleration
    return (
        p,
        v + t[-1] * acceleration,
        acceleration,
        np.linalg.norm(residual, axis=1).max(),
        span,
    )


class FlightFit(ReboundReadout):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.spin = np.zeros(3)
        self.spin_time = None

    def update(self, now, point, state):
        if (
            self.last_time is None
            or now <= self.last_time
            or now - self.last_time > 0.14
        ):
            self.spin_time = None
        base = super().update(now, point, state)
        if self.turn is not None:
            when, _, post, pre = self.turn
            if 0 <= now - when < 0.15:
                dv = post - pre
                self.spin = np.clip(
                    BALL_MASS
                    * BALL_RADIUS
                    / BALL_INERTIA
                    * np.array([dv[1], -dv[0], 0.0]),
                    -85.0,
                    85.0,
                )
                self.spin_time = when
        if base is None or point is None or len(self.track) < 4:
            return base
        ts = np.array([t - now for t, _ in self.track])
        ps = np.stack([p for _, p in self.track])
        if ts[-1] < -0.001 or not np.isfinite(ps[-1]).all():
            return base
        # Choose the longest recent segment consistent with one smooth flight.
        # An actual rebound or hit ends that segment. All positions come from older or current images.
        start = int(np.searchsorted(ts, -0.24))
        gaps = np.flatnonzero(np.diff(ts) > self.period * 1.6)
        if len(gaps):
            start = max(start, gaps[-1] + 1)
        if self.turn is not None and now >= self.turn[0]:
            start = max(start, int(np.searchsorted(ts, self.turn[0] - now + 0.004)))
        for first in range(start, len(ts) - 3):
            p, v, _, residual, _ = smooth(ts[first:], ps[first:])
            if residual < 0.004 and np.linalg.norm(p - ps[-1]) < 0.0045:
                return np.r_[p, v]
        return base

    def rotation(self, now):
        if self.spin_time is None or not 0 <= now - self.spin_time < 0.45:
            return np.zeros(3)
        return self.spin * np.exp(-SPIN_DECAY * (now - self.spin_time))


def train(source, checkpoint):
    with np.load(source) as z:
        np.savez_compressed(checkpoint, **dict(z), smooth_flight=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "test", "checkpoint", "report"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    train(args.source, args.checkpoint)
    _write(args.report, score(args.test, FlightFit(args.checkpoint)))
