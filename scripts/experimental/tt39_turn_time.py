"""Recover the time of a completed movement reversal from both observed flight segments."""

import itertools

import numpy as np


def turn_time(rows):
    """Rows pair control ticks with optical position/velocity; times returned in control ticks."""
    xs = [(t, np.asarray(x)) for t, x in rows if x is not None]
    pairs = [
        (a, b)
        for a, b in itertools.pairwise(xs)
        if a[1][4] < -0.9 and b[1][4] > 0.9 and b[0] - a[0] <= 9
    ]
    if not pairs:
        return None
    a, b = pairs[-1]
    detected = (a[0] + b[0]) * 0.5
    # Camera filtering often extrapolates the incoming flight while it gathers
    # the first few outgoing exposures. Fit unambiguous observations on each side.
    pre = [(t, x) for t, x in xs if detected - 17 <= t <= detected - 5 and x[4] < -0.9]
    post = [(t, x) for t, x in xs if b[0] + 1 <= t <= b[0] + 12 and x[4] > 0.9]
    if len(pre) < 3 or len(post) < 3:
        return detected
    times = detected + np.arange(-9.0, 2.26, 0.25)

    def forecast(group):
        t, x = zip(*group, strict=True)
        z = np.asarray(x)
        dt = (times[:, None] - np.array(t)[None, :]) / 60
        v = z[:, 3:6]
        acc = (
            np.array([0.0, 0.0, -9.81])
            - 0.13 * np.linalg.norm(v, axis=1, keepdims=True) * v
        )
        projected = (
            z[None, :, :3]
            + dt[:, :, None] * v[None, :, :]
            + 0.5 * dt[:, :, None] ** 2 * acc[None, :, :]
        )
        return np.median(projected, axis=1)

    # Center-to-center flight extrapolations meet near the moving surface. A
    # short physical impulse gives finite, rather than exact, intersection.
    gaps = np.linalg.norm(forecast(pre) - forecast(post), axis=1)
    i = int(np.argmin(gaps))
    return times[i] if gaps[i] < 0.085 else detected
