"""Learn the normal visual response to ordinary commands without rendering a second arm."""

from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .tt20_latency import identify_delay
from .tt21_game_delay import MAX_DELAY


def features(commands):
    rates = np.asarray(commands)[:, :5]
    previous = np.zeros_like(rates[0])
    filtered = []
    for sample in rates:
        previous = 0.45 * previous + 0.55 * sample
        filtered.append(previous.copy())
    window = np.pad(rates, ((4, 0), (0, 0)))
    recent = np.concatenate(
        [window[i : i + len(rates)] * weight for i, weight in enumerate((0.18, 0.35, 0.6, 0.8, 1.0))], axis=1
    )
    posture = np.clip(np.cumsum(rates, axis=0), -8, 8)
    return np.column_stack((recent, filtered, 0.09 * posture))


class MotorTraceMemory:
    """Use similar recent commands from earlier points to forecast visual motion.

    Align subsequent experience to the chosen relative delay. The first moving
    point establishes an implicit nominal installation. No initial absolute lag
    can be recovered from an unknown body and camera through this assumption.
    """

    def __init__(self):
        self.examples = []
        self.delay = 0
        self.seen = 0
        self.loss = np.zeros(MAX_DELAY + 1)

    def observe(self, commands, measured):
        start = perf_counter()
        n = len(measured)
        result = {"previous": self.delay, "delay": self.delay, "status": "no_motion", "rendered": 0}
        if n < 12:
            result["wall_seconds"] = perf_counter() - start
            return result
        x = features(commands[:n])
        y = np.asarray(measured, dtype=float)
        y = np.maximum(y - np.quantile(y, 0.12), 0.0)
        active = np.flatnonzero(np.max(abs(commands[:n]), axis=1) > 0.02)
        if len(active) >= 3 and np.linalg.norm(y) > 0.1:
            start_frame = max(0, active[0] - 1)
            stop = min(n, active[-1] + MAX_DELAY + 8)
            if self.examples:
                train_x = np.concatenate([xx for xx, _ in self.examples])
                train_y = np.concatenate([yy for _, yy in self.examples])
                tree = cKDTree(train_x)
                distance, ids = tree.query(x[start_frame:stop], k=min(7, len(train_x)))
                weight = 1.0 / (0.035 + distance) ** 2
                prediction = np.sum(weight * train_y[ids], axis=1) / weight.sum(axis=1)
                try:
                    match = identify_delay(prediction, y[start_frame:stop], maximum=MAX_DELAY)
                except ValueError:
                    match = None
                result["status"] = "ambiguous"
                if match is not None:
                    loss = np.asarray(match["losses"])
                    result["match"] = match
                    if loss.min() < 0.82 and match["margin"] >= 0.018:
                        self.loss = 0.20 * self.loss + loss - loss.min()
                        self.delay = int(np.argmin(self.loss))
                        result.update(status="accepted", delay=self.delay)
            else:
                result["status"] = "first_reference"
            if result["status"] in ("accepted", "first_reference"):
                lo, hi = start_frame, stop - self.delay
                if hi > lo:
                    self.examples.append((x[lo:hi].copy(), y[lo + self.delay : hi + self.delay].copy()))
                    self.examples = self.examples[-24:]
                    self.seen += 1
        result.update(examples=self.seen, wall_seconds=perf_counter() - start)
        return result


class OnsetDelay:
    """Measure the lag between ordinary movement onset and visible motion.

    The optical noise floor comes from the same point before movement. When a
    stroke begins immediately at reset, or the image never clearly moves, the
    last estimate carries forward. This estimates an end-to-end response lag.
    """

    def __init__(self):
        self.delay = 0

    def observe(self, commands, measured):
        start = perf_counter()
        signal = np.asarray(measured)
        moving = np.flatnonzero(np.max(np.abs(commands[: len(signal)]), axis=1) > 0.04) if len(signal) else []
        result = {"previous": self.delay, "delay": self.delay, "status": "insufficient_motion", "rendered": 0}
        if len(moving) and moving[0] >= 3:
            first = int(moving[0])
            quiet = signal[max(0, first - 8) : first]
            baseline = np.median(quiet)
            mad = np.median(np.abs(quiet - baseline))
            response = signal[first : first + MAX_DELAY + 5]
            if len(response) >= 3:
                threshold = max(
                    1e-4,
                    float(np.max(response)) * 0.015,
                    1.5 * baseline + 4 * mad,
                    1.2 * float(np.quantile(quiet, 0.8)),
                )
                rising = response > threshold
                edges = np.flatnonzero(rising[:-1] & rising[1:])
                if len(edges) and edges[0] <= MAX_DELAY and response.max() > 2.5 * threshold:
                    self.delay = int(edges[0])
                    result.update(
                        delay=self.delay,
                        status="accepted",
                        frame=first + self.delay + 2,
                        threshold=float(threshold),
                    )
        result["wall_seconds"] = perf_counter() - start
        return result
