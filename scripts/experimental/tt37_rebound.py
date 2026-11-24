"""Identify repeatable optical impulses and predict the ball during ambiguous rebound frames."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _write
from .tt37_visual_inertia import VisualInertia, score


def accel(v):
    return np.array([0.0, 0.0, -9.81]) - 0.13 * np.linalg.norm(v) * v


def fit(t, p):
    v = (p[-1] - p[0]) / (t[-1] - t[0])
    a = accel(v)
    corrected = p - 0.5 * t[:, None] ** 2 * a
    centered = t - t.mean()
    v = centered @ corrected / (centered @ centered)
    return corrected.mean(axis=0) - v * t.mean(), v, a


def events(observations):
    s = observations["sensor"]
    times = s[:, 0]
    p = s[:, 1:4]
    dt = float(observations["period"]) if "period" in observations else 1 / 30
    incoming = []
    changes = []
    heights = []
    for i in range(4, len(s) - 4):
        t = times[i - 4 : i + 5] - times[i]
        positions = p[i - 4 : i + 5]
        if not np.isfinite(positions).all() or not np.allclose(np.diff(t), dt):
            continue
        if p[i, 2] >= min(p[i - 1, 2], p[i + 1, 2]) or p[i, 2] > 0.87:
            continue
        before, v, a = fit(t[:4], positions[:4])
        after, w, b = fit(t[-4:], positions[-4:])
        if v[2] > -0.9 or w[2] < 0.9 or v[1] * w[1] <= 0.5:
            continue
        instant = (after[2] - before[2]) / (v[2] - w[2])
        if abs(instant) > 0.82 * dt:
            continue
        pos = before + v * instant + 0.5 * a * instant**2
        vel = v + a * instant
        outgoing = w + b * instant
        heights.append(pos[2])
        incoming.append(vel)
        changes.append(outgoing - vel * [1, 1, -1])
    z = np.asarray(heights)
    keep = np.abs(z - np.median(z)) < 0.012
    return {
        "rebound_states": np.asarray(incoming)[keep],
        "rebound_delta": np.asarray(changes)[keep],
        "rebound_z": z[keep],
    }


class ReboundReadout(VisualInertia):
    def __init__(self, path):
        super().__init__(path)
        with np.load(path) as f:
            self.incoming = f["rebound_states"]
            self.changes = f["rebound_delta"]
            self.height = f["rebound_z"]
        self.rtree = cKDTree(self.incoming / [0.8, 1.4, 0.85])
        self.track = deque(maxlen=16)
        self.turn = None

    def update(self, now, point, state):
        if (
            self.last_time is None
            or now <= self.last_time
            or now - self.last_time > 0.14
        ):
            self.track.clear()
            self.turn = None
        clean = super().update(now, point, state)
        if point is None or not np.isfinite(point).all():
            return clean
        self.track.append((now, np.array(point)))
        if clean is None or len(self.track) < 4:
            return clean
        times = np.array([r[0] - now for r in self.track])
        ps = np.array([r[1] for r in self.track])
        t, p = times[-4:], ps[-4:]
        if max(np.diff(t)) > 1.5 * self.period:
            return clean
        pos, vel, a = fit(t, p)
        if self.turn and now > self.turn[0] + 0.19:
            self.turn = None
        height = np.median(self.height)
        if (
            vel[2] < -0.85
            and 0.008 < pos[2] - height < 0.40
            and (self.turn is None or now < self.turn[0] - 0.003)
        ):
            spec = vel[2] ** 2 - 2 * a[2] * (pos[2] - height)
            if spec >= 0:
                eta = (-vel[2] - np.sqrt(spec)) / a[2]
                if 0 <= eta < 0.14:
                    pre = vel + eta * a
                    d, ix = self.rtree.query(
                        pre / [0.8, 1.4, 0.85], k=min(15, len(self.height))
                    )
                    w = np.exp(-0.5 * (d / max(0.22, d[len(d) // 2])) ** 2)
                    w /= w.sum()
                    local = w @ self.height[ix]
                    eta += (local - height) / min(-0.9, pre[2])
                    center = pos + vel * eta + 0.5 * a * eta**2
                    new = pre * [1, 1, -1] + w @ self.changes[ix]
                    self.turn = (now + eta, center, new, pre)
        if self.turn is None:
            return clean
        when, center, post, pre = self.turn
        elapsed = now - when
        if not -0.010 < elapsed < 0.155:
            return clean
        predicted_pos = center + elapsed * post + 0.5 * elapsed**2 * accel(post)
        if elapsed > 0.015 and np.linalg.norm(predicted_pos - point) > 0.05:
            self.turn = None
            return clean
        if elapsed < 0.007:
            return clean
        pred = post + elapsed * accel(post)
        after = times + now > when + 0.006
        if after.sum() >= 2:
            _, measured, _ = fit(times[after], ps[after])
            weight = 0.85 if after.sum() == 2 else 0.96
            pred = weight * measured + (1.0 - weight) * pred
        out = np.r_[point, pred]
        return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("train", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("base", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("receipt", type=Path)
    a = p.parse_args()
    with np.load(a.train) as source, np.load(a.base) as base:
        hits = events(source)
        np.savez_compressed(a.checkpoint, **dict(base), **hits)
    _write(
        a.receipt,
        {
            "practice_bounces": len(hits["rebound_z"]),
            "held": score(a.test, ReboundReadout(a.checkpoint)),
        },
    )
