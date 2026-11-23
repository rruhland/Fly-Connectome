"""Predict the observable flight through a bounce and refit its response during play."""

import numpy as np


def fit_arc(track, gravity=-9.81):
    a = np.asarray(track)
    dt = (a[:, 0] - a[-1, 0]) / 30.0
    v = np.column_stack((np.ones(len(dt)), dt))
    points = a[:, 1:].copy()
    points[:, 2] -= 0.5 * gravity * dt**2
    intercept, velocity = np.linalg.lstsq(v, points, rcond=None)[0]
    return intercept, velocity


def observed_bounce(track):
    a = np.asarray(track)
    if len(a) < 8:
        return None
    for i in range(3, len(a) - 3):
        if a[i, 3] > 0.12 or not (a[i - 1, 3] >= a[i, 3] <= a[i + 1, 3]):
            continue
        before = a[max(0, i - 6) : i]
        after = a[i + 1 : i + 7]
        if before[-1, 0] - before[0, 0] < 2 or after[-1, 0] - after[0, 0] < 2:
            continue
        p, v = fit_arc(before)
        z = 0.021
        disc = v[2] ** 2 + 2 * 9.81 * (p[2] - z)
        if disc <= 0:
            continue
        impact = (v[2] + np.sqrt(disc)) / 9.81
        inbound = v[2] - 9.81 * impact
        aft, w = fit_arc(after)
        reb = w[2] + 9.81 * ((after[-1, 0] - before[-1, 0]) / 30 - impact)
        if inbound > -0.6 or reb < 0.25 or abs(v[1]) < 0.6:
            continue
        result = np.array([reb / -inbound, w[1] / v[1], w[0] - v[0]])
        if 0.18 < result[0] < 1.45 and 0.3 < result[1] < 1.6 and abs(result[2]) < 3.0:
            return result
    return None


def predict(track, target_y, material, g=-9.81):
    if len(track) < 3:
        return None
    sample = np.asarray(track[-5:])
    if sample[-1, 0] - sample[0, 0] < 2:
        return None
    p, v = fit_arc(sample, g)
    if v[1] > -0.4:
        return None
    duration = (target_y - p[1]) / v[1]
    if not -0.09 < duration < 1.3:
        return None
    # A just-observed bounce supersedes the prior for the rest of this flight.
    a = np.asarray(track)
    min_i = np.argmin(a[:, 3])
    if a[min_i, 3] < 0.12 and len(a) - min_i >= 3 and a[-1, 3] > a[min_i, 3] + 0.015:
        p, v = fit_arc(a[max(min_i + 1, len(a) - 5) :], g)
        if v[1] > -0.4:
            return None
        duration = (target_y - p[1]) / v[1]
        return np.r_[
            p[0] + v[0] * duration,
            p[2] + v[2] * duration + 0.5 * g * duration**2,
            v[0],
            v[1],
            v[2] + g * duration,
        ]
    z = 0.021
    disc = v[2] ** 2 - 2 * g * (p[2] - z)
    if disc <= 0:
        return None
    tb = (-v[2] - np.sqrt(disc)) / g
    if tb < 0 or tb > duration:
        return np.r_[
            p[0] + v[0] * duration,
            p[2] + v[2] * duration + 0.5 * g * duration**2,
            v[0],
            v[1],
            v[2] + g * duration,
        ]
    end = p + v * tb
    u = np.array([v[0] + material[2], v[1] * material[1], -(v[2] + g * tb) * material[0]])
    rest = (target_y - end[1]) / u[1]
    return np.r_[end[0] + u[0] * rest, z + u[2] * rest + 0.5 * g * rest**2, u[0], u[1], u[2] + g * rest]


class MaterialMemory:
    def __init__(self, source, capacity=5):
        self.source = np.array(source)
        self.history = []
        self.capacity = capacity

    @property
    def current(self):
        if not self.history:
            return self.source.copy()
        # Side spin varies shot-to-shot. Preserve the prior lateral mean.
        value = np.median(np.asarray(self.history[-self.capacity :]), axis=0)
        value[2] = self.source[2]
        return value

    def observe(self, track):
        estimated = observed_bounce(track)
        if estimated is not None:
            self.history.append(estimated)
        return estimated
