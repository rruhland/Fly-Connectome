"""Infer changing table restitution from the two visible sides of a single bounce."""
from collections import deque

import numpy as np

from .tt5_player import DRAG, GRAVITY, TABLE_END, TABLE_HALF_WIDTH
from .tt6_continuous import ContinualPlayer


def _vertical_arc(times, heights, depth_rate):
    centered = times - times.mean()
    spread = centered @ centered
    squares = .5 * times**2
    speed = centered @ (heights + GRAVITY * squares) / spread
    # A local constant-acceleration approximation gives the drag term limited
    # influence over these short, partly occluded segments.
    acceleration = -GRAVITY - .5 * DRAG * np.hypot(depth_rate, speed) * speed
    corrected = heights - acceleration * squares
    speed = centered @ corrected / spread
    height = corrected.mean() - speed * times.mean()
    return np.array([height, speed, acceleration]), corrected - height - speed * times


def optical_bounce(rows):
    """Estimate one approaching bounce from recent stereo ball centers."""
    if len(rows) < 6:
        return None
    times, points = zip(*rows)
    times, points = np.asarray(times, dtype=float), np.asarray(points)
    gaps = np.diff(times)
    if np.any((gaps <= 0.) | (gaps > 3.)):
        return None
    depth_rates = np.diff(points[:, 1]) / gaps
    depth_rate = depth_rates.mean()
    if np.any(depth_rates >= -.005) or not -.24 < depth_rate < -.020:
        return None

    best = None
    for split in range(3, len(times) - 2):
        left, right = times[split - 1:split + 1]
        near = points[split - 1:split + 1]
        if near[:, 2].min() > .15:
            continue
        middle = (left + right) / 2.
        offsets = times - middle
        if np.max(np.abs(offsets)) > 5.:
            continue
        incoming, error_in = _vertical_arc(offsets[:split], points[:split, 2], depth_rate)
        outgoing, error_out = _vertical_arc(offsets[split:], points[split:, 2], depth_rate)
        difference = incoming - outgoing
        if difference[1] > -.03:
            continue
        # Match the arcs to one another. Fixing the intercept at the ball's
        # radius would turn pixel bias or contact compression into speed bias.
        step = -difference[0] / difference[1]
        if abs(step) > (right - left) / 2. + .45:
            continue
        step -= ((difference @ [1., step, .5 * step**2]) /
                 (difference[1] + difference[2] * step))
        impact = middle + step
        height = incoming @ [1., step, .5 * step**2]
        if not left - .35 < impact < right + .35 or not -.020 < height < .070:
            continue
        x, depth = (np.interp(impact, times, axis) for axis in points[:, :2].T)
        if abs(x) > TABLE_HALF_WIDTH or not -TABLE_END < depth < -.03:
            continue
        before = incoming[1] + incoming[2] * step
        after = outgoing[1] + outgoing[2] * step
        if not (-.24 < before < -.032 and .016 < after < .22):
            continue
        ratio = -after / before
        error = np.sqrt(np.mean(np.r_[error_in, error_out]**2))
        if .35 < ratio < 1.22 and error < .006 and (best is None or error < best[0]):
            best = float(error), float(ratio), float(impact)
    return best


class SurfaceBelief:
    def __init__(self, observations=()):
        self.values = deque(observations, maxlen=5)

    def predict(self, initial):
        if not self.values:
            return initial
        weights = .59**np.arange(len(self.values) - 1, -1, -1)
        return float(np.clip((weights @ self.values + .27*initial)/(weights.sum()+.27), .50, 1.02))


class SurfacePlayer(ContinualPlayer):
    def __init__(self, *args, belief, adaptive, **kwargs):
        self.belief = belief
        self.adaptive = adaptive
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.sightings = deque(maxlen=7)
        self.bounces = []
        self.last_bounce = -100.
        self.start_ratio = self._surface_ratio()

    def _surface_ratio(self):
        initial = super()._surface_ratio()
        return self.belief.predict(initial) if self.adaptive else initial

    def _reachable_plane(self, plane, ball):
        # Low bounces can land again before the ball reaches the normal strike
        # plane. Move earlier when the predicted contact there is also low.
        if self.adaptive:
            advance = np.clip(1.1 * (.83 - self._surface_ratio()), 0., .19)
            plane += advance * np.clip((.24 - ball[2]) / .08, 0., 1.)
        return super()._reachable_plane(plane, ball)

    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        frame = self.trace[-1]
        if frame['ball'] is None:
            return command
        self.sightings.append((frame['tick'], frame['ball']))
        if frame['tick'] - self.last_bounce > 9:
            estimate = optical_bounce(self.sightings)
            if estimate is not None:
                error, ratio, tick = estimate
                self.belief.values.append(ratio)
                self.bounces.append(dict(tick=tick, observed_at=frame['tick'], ratio=ratio, fit_rmse=error))
                self.last_bounce = tick
                self.sightings.clear()
        return command

    def finish(self, reward):
        result = super().finish(reward)
        result['table_bounce'] = dict(samples=self.bounces, starting_ratio=self.start_ratio,
                                      ratio=self._surface_ratio())
        return result
