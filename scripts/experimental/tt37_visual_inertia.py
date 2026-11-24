"""Learn optical velocity corrections by predicting later stereo observations across changes of motion."""

import argparse
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _write
from .tt29_fast_camera import native_ball_pixels
from .tt35_error import triangulate

SCALE = np.array([0.12, 0.85, 0.85, 0.85, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
DT = 1 / 30


def feature(points, base, period=DT):
    p = np.asarray(points)
    v = np.diff(p, axis=0) / period
    # Disagreements among recent velocities distinguish impact, rebound and free flight.
    return np.r_[
        p[-1, 2] - 0.8, base[3:6], v[-1] - base[3:6], v[-1] - v[-2], v[-2] - v[-3]
    ]


def future_velocity(x, period=DT):
    dt = np.arange(4) * period
    if not np.isfinite(x).all():
        return None
    guess = (x[-1] - x[0]) / (3 * period)
    acc = np.array([0.0, 0.0, -9.81]) - 0.13 * np.linalg.norm(guess) * guess
    corrected = x - 0.5 * dt[:, None] ** 2 * acc
    vel = (dt - dt.mean()) @ corrected / np.sum((dt - dt.mean()) ** 2)
    fit = corrected.mean(axis=0) + (dt - dt.mean())[:, None] * vel
    if np.max(np.linalg.norm(fit - corrected, axis=1)) > 0.0035:
        return None
    return vel


def training(data):
    period = float(data["period"]) if "period" in data else DT
    s = data["sensor"]
    g = data["groups"]
    samples = []
    responses = []
    for i in range(3, len(s) - 3):
        if (
            not np.all(g[i - 3 : i + 4] == g[i])
            or not np.isfinite(s[i - 3 : i + 1, 1:4]).all()
            or not np.isfinite(s[i, 4:11]).all()
        ):
            continue
        if not np.allclose(np.diff(s[i - 3 : i + 4, 0]), period) or abs(s[i, 8]) < 0.9:
            continue
        label = future_velocity(s[i : i + 4, 1:4], period)
        if label is None or np.linalg.norm(label - s[i, 7:10]) > 9:
            continue
        samples.append(feature(s[i - 3 : i + 1, 1:4], s[i, 4:10], period))
        responses.append(label - s[i, 7:10])
    return np.array(samples), np.array(responses)


class VisualInertia:
    def __init__(self, checkpoint):
        z = np.load(checkpoint)
        self.period = float(z["period"]) if "period" in z else DT
        self.y = z["updates"]
        self.tree = cKDTree(z["features"] / SCALE)
        self.gated = bool(z["gated"]) if "gated" in z else False
        self.samples = deque(maxlen=4)
        self.last_time = None

    def update(self, now, point, state):
        if self.last_time is None or not np.isclose(
            now - self.last_time, self.period, atol=0.001
        ):
            self.samples.clear()
        self.last_time = now
        if point is None or not np.isfinite(point).all():
            self.samples.clear()
            return state
        self.samples.append(point)
        if state is None or len(self.samples) < 4:
            return state
        x = feature(self.samples, state, self.period) / SCALE
        if self.gated:
            # Use the learned impulse correction only after a deviation has appeared in the images.
            # This prevents a future-frame label at a bounce from anticipating it during descent.
            secants = np.diff(np.asarray(self.samples), axis=0) / self.period
            change = np.linalg.norm(
                np.diff(secants, axis=0) - np.array([0.0, 0.0, -9.81]) * self.period,
                axis=1,
            )
            latest = secants[-1] + 0.5 * self.period * (
                np.array([0.0, 0.0, -9.81])
                - 0.13 * np.linalg.norm(secants[-1]) * secants[-1]
            )
            if np.max(change) < 0.46 and np.linalg.norm(latest - state[3:6]) < 0.26:
                return state
        d, idx = self.tree.query(x, k=12)
        scale = max(0.22, d[3])
        w = np.exp(-0.5 * ((d - d[0]) / scale) ** 2)
        strength = 1 / (1 + (d[1] / 2.0) ** 4)
        delta = (w @ self.y[idx]) / w.sum() * strength
        out = state.copy()
        out[3:6] += delta
        # Stereo gives a current independent location when the older fit spans a reversal.
        gain = np.clip(np.linalg.norm(delta) / 0.4, 0.0, 1.0)
        if np.linalg.norm(point - state[:3]) < 0.13:
            out[:3] += gain * (point - state[:3])
        return out

    def flight(self, now, frames, flight, rig):
        balls = [native_ball_pixels(frames[n]) for n in ("left", "right")]
        pt = (
            None
            if any(b is None or b.clipped for b in balls)
            else triangulate([b.center for b in balls], rig)
        )
        state = (
            None
            if flight is None or now - flight.time > 0.07
            else np.r_[*flight.advance(now)]
        )
        updated = self.update(now, pt, state)
        if updated is None or pt is None:
            return flight
        return replace(flight, position=updated[:3], velocity=updated[3:], time=now)


class LearnedOptics:
    def __init__(self, tracker, checkpoint):
        self.tracker = tracker
        if isinstance(checkpoint, VisualInertia):
            self.filter = checkpoint
        else:
            with np.load(checkpoint) as archive:
                rebound = "rebound_states" in archive
                smooth = "smooth_flight" in archive
            if smooth:
                from .tt38_flight_fit import FlightFit

                self.filter = FlightFit(checkpoint)
            elif rebound:
                from .tt37_rebound import ReboundReadout

                self.filter = ReboundReadout(checkpoint)
            else:
                self.filter = VisualInertia(checkpoint)
        self.latest = None

    def observe(self, now, frames, q):
        self.tracker.observe(now, frames, q)
        self.latest = self.filter.flight(
            now, frames, self.tracker.latest, self.tracker.rig
        )
        return self.latest


def score(data, model):
    d = np.load(data)
    s = d["sensor"]
    g = d["groups"]
    ev = d["evaluation"]
    after = np.full((len(s), 6), np.nan)
    for i, row in enumerate(s):
        if i == 0 or not np.array_equal(g[i], g[i - 1]):
            model.samples.clear()
            model.last_time = None
        base = None if not np.isfinite(row[4:10]).all() else row[4:10]
        x = model.update(row[0], row[1:4], base)
        if x is not None:
            after[i] = x
    before = s[:, 4:10]
    use = (
        (ev[:, 4] < -1)
        & (ev[:, 1] < -0.55)
        & (ev[:, 1] > -1.65)
        & (s[:, 10] < 0.08)
        & np.isfinite(after).all(axis=1)
    )
    result = {}
    for key, mask in [
        ("near", use),
        ("postbounce", use & (ev[:, 10] > 0) & (ev[:, 5] > 0.35)),
    ]:
        result[key] = {}
        for name, pred in [("original", before), ("trained", after)]:
            error = np.abs(pred[mask] - ev[mask, :6])
            result[key][name] = {
                "n": int(mask.sum()),
                "mae": error.mean(axis=0).tolist(),
                "p90": np.quantile(error, 0.9, axis=0).tolist(),
                "velocity_catastrophes": int(
                    (np.linalg.norm(error[:, 3:], axis=1) > 0.6).sum()
                ),
            }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("train", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("score", type=Path)
    p.add_argument("--gated", action="store_true")
    a = p.parse_args()
    with np.load(a.train) as observed:
        x, y = training(observed)
        period = float(observed["period"]) if "period" in observed else DT
    np.savez_compressed(a.weights, features=x, updates=y, gated=a.gated, period=period)
    res = score(a.test, VisualInertia(a.weights))
    _write(a.score, {"training_frames": len(x), "held": res})
    print(res)
