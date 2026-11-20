"""Learn a wall's velocity response from visible motion on both sides of a bounce."""

from collections import deque
from dataclasses import asdict, dataclass
import math


@dataclass
class WallMemory:
    """Combine independent camera measurements by their estimated precision."""

    default: float = 1.
    count: int = 0
    precision: float = 0.
    weighted_total: float = 0.

    @property
    def coefficient(self):
        estimate = self.weighted_total / self.precision if self.precision else self.default
        return min(1., max(0., estimate))

    @property
    def uncertainty(self):
        return 1 / math.sqrt(self.precision) if self.precision else None

    def observe(self, ratio, uncertainty):
        if not math.isfinite(ratio) or not 0 < uncertainty < math.inf:
            raise ValueError('a finite bounce measurement and positive uncertainty are required')
        weight = 1 / uncertainty**2
        self.precision += weight
        self.weighted_total += ratio * weight
        self.count += 1

    def to_state(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state):
        return cls(**state)


@dataclass(frozen=True)
class Bounce:
    frame: int
    coefficient: float
    uncertainty: float


def _slope(points, axis):
    center = sum(point[0] for point in points) / len(points)
    variance = sum((point[0] - center)**2 for point in points)
    return sum((point[0] - center) * point[axis] for point in points) / variance, variance


class WallFlight:
    """Track an object in public coordinates where y faces the two side walls.

    Flight geometry uses the camera coordinate system. Positions can be missing
    on occlusion. Time defaults to camera-frame units, or callers can supply
    the positive elapsed time in seconds on each transition.
    """

    def __init__(self, memory, bounds, *, position_noise=.004, window=8, hints=False):
        self.memory = memory
        self.bounds = bounds
        self.noise = position_noise
        self.window = window
        self.hints = hints
        self.guard = 2
        self.history = deque(maxlen=2 * window + 2 * self.guard + 1)
        self.reset()

    def reset(self):
        self.history.clear()
        self.frame = 0
        self.time = 0.
        self.last_bounce = None
        self.last_hint = None
        self.velocity = None
        self.position = None

    def observe(self, position, elapsed=1., *, learn=True):
        if not math.isfinite(elapsed) or elapsed <= 0.:
            raise ValueError('positive camera duration required')
        self.frame += 1
        self.time += elapsed
        self.position = position
        self.velocity = None
        self.last_bounce = None
        self.last_hint = None
        if position is None:
            self.history.clear()
            return None
        self.history.append((self.time, *map(float, position)))
        if len(self.history) < 4:
            return None

        recent = list(self.history)
        self._velocity(recent)
        if len(recent) == self.history.maxlen:
            self.last_bounce = self._measure_bounce(recent)
        if self.last_bounce is not None:
            # Leave the outgoing samples available for immediate forecasting.
            for _ in range(self.window + self.guard + 1):
                self.history.popleft()
            if learn:
                self.memory.observe(self.last_bounce.coefficient, self.last_bounce.uncertainty)
        elif self.hints and len(recent) >= self.window + 2 * self.guard + 6:
            early_width = self.window + 2 * self.guard + 6
            self.last_hint = self._measure_bounce(recent[-early_width:], outgoing=5)
        return self.last_bounce

    def _velocity(self, recent):
        # A short window responds promptly to changes from a paddle or wall.
        # Use a longer estimate on steady motion to reduce pixel quantization.
        samples = recent[-min(10, len(recent)):]
        short = samples[-4:]
        s_elapsed = short[-1][0] - short[0][0]
        l_elapsed = samples[-1][0] - samples[0][0]
        velocity = []
        for axis in (1, 2):
            local = (short[-1][axis] - short[0][axis]) / s_elapsed
            trend = (samples[-1][axis] - samples[0][axis]) / l_elapsed
            changed = abs(local - trend) * s_elapsed > 2.5 * self.noise
            velocity.append(_slope(short if changed else samples, axis)[0])
        self.velocity = tuple(velocity)

    def _measure_bounce(self, samples, outgoing=None):
        pivot = self.window + self.guard
        middle = samples[pivot]
        wall = min(self.bounds, key=lambda bound: abs(bound - middle[2]))
        if abs(wall - middle[2]) > 4 * self.noise:
            return None
        before, after = samples[:self.window], samples[-(outgoing or self.window):]
        vin, pre_precision = _slope(before, 2)
        vout, post_precision = _slope(after, 2)
        wall_direction = 1 if wall == self.bounds[1] else -1
        if wall_direction * vin <= 0. or wall_direction * vout >= 0.:
            return None
        pre_error = self.noise / math.sqrt(pre_precision)
        post_error = self.noise / math.sqrt(post_precision)
        if abs(vin) < 3. * pre_error or abs(vout) < 1.5 * post_error:
            return None

        # A rail collision preserves longitudinal velocity. Vehicle/puck
        # contact can also turn the y coordinate, often through a new x speed.
        x_in, _ = _slope(before, 1)
        x_out, _ = _slope(after, 1)
        if abs(x_out - x_in) > 3.5 * math.hypot(pre_error, post_error):
            return None

        turning_times = []
        for points, speed in ((before, vin), (after, vout)):
            turning_time = sum(point[0] + (wall - point[2]) / speed for point in points) / len(points)
            turning_times.append(turning_time)
        offset = abs(turning_times[0] - turning_times[1]) / (1 / abs(vin) + 1 / abs(vout))
        if offset > 2.5 * self.noise:
            return None

        ratio = abs(vout / vin)
        error = max(.017, math.hypot(post_error, ratio * pre_error) / abs(vin))
        if ratio > 1. + 3. * error:
            return None
        return Bounce(self.frame - (len(samples) - 1 - pivot), ratio, error)


def project_crossing(position, velocity, plane, bounds, restitution):
    """Forecast y when a free flight reaches the requested x plane."""
    if position is None or velocity is None or not velocity[0]:
        return None
    remaining = (plane - position[0]) / velocity[0]
    if remaining < 0.:
        return None
    low, high = bounds
    y = max(low, min(high, position[1]))
    vy = velocity[1]
    while vy:
        wall = high if vy > 0. else low
        travel_time = (wall - y) / vy
        if remaining <= travel_time:
            return y + vy * remaining
        remaining -= travel_time
        y, vy = wall, -restitution * vy
    return y
