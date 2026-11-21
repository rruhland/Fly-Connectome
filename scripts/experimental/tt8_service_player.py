"""Recognize a nearby toss from stereo and intercept its descending trajectory."""
from collections import deque

import numpy as np

from .tt5_player import GRAVITY
from .tt7_flight import FlightPlayer, outgoing_speed


# Each candidate specifies contact height above the table, forward rate in
# metres per camera frame and blade inclination. The calibrated body controller
# has to discover which commanded channels produce that movement.
STROKES = (
    (.32, .080, .13),
    (.33, .10, -.04),
    (.32, .075, .32),
    (.39, .095, .11),
    (.28, .10, .20),
    (.37, .064, -.07),
)


def falling_crossing(rows, height):
    """Fit a ballistic arc in camera coordinates and find its descending crossing."""
    if len(rows) < 3:
        return None
    ticks, points = zip(*rows)
    t = np.asarray(ticks, dtype=float) - ticks[-1]
    pos = np.asarray(points)
    adjusted = pos.copy()
    adjusted[:, 2] += .5 * GRAVITY * t**2
    centered = t - t.mean()
    velocity = centered @ adjusted / (centered @ centered)
    start = adjusted.mean(axis=0) - t.mean() * velocity
    if max(np.linalg.norm(adjusted - start - t[:, None] * velocity, axis=1)) > .055:
        return None
    discriminant = velocity[2]**2 + 2. * GRAVITY * (start[2] - height)
    if discriminant <= 0. or np.linalg.norm(velocity[:2]) > .045:
        return None
    duration = (velocity[2] + np.sqrt(discriminant)) / GRAVITY
    return ticks[-1] + duration, start[:2] + duration * velocity[:2]


class ServiceMemory:
    def __init__(self, state=None):
        self.counts = np.zeros(len(STROKES), dtype=int)
        self.totals = np.zeros(len(STROKES))
        if state is not None:
            self.counts[:] = state['counts']
            self.totals[:] = state['totals']

    def choose(self):
        values = (self.totals + .5) / (self.counts + 1)
        bonus = .6 * np.sqrt(np.log(2. + self.counts.sum()) / (1 + self.counts))
        return int(np.argmax(values + bonus))

    def observe(self, choice, outcome):
        self.counts[choice] += 1
        self.totals[choice] += outcome

    def state(self):
        return dict(counts=self.counts.tolist(), totals=self.totals.tolist())


class ServicePlayer(FlightPlayer):
    def __init__(self, *args, service_memory, fixed=None, **kwargs):
        self.service_memory = service_memory
        self.fixed = fixed
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.toss = deque(maxlen=9)
        self.serve_plan = None
        self.service_choice = None
        self.armed = False

    def _stroke_goals(self, estimate):
        tick = self.scene.tick
        observed = self.scene.tracker.ball
        if (tick <= 20 and observed is not None and observed.observed
                and observed.position[1] < -.98 and (self.serve_plan is None or tick < self.serve_plan[0] - 1.5)):
            self.toss.append((tick, observed.position))
            if self.service_choice is None and len(self.toss) >= 3:
                points = np.asarray([row[1] for row in self.toss])
                ascent = points[-1, 2] - points[0, 2]
                if ascent > .05 and np.ptp(points[:, 1]) < .11:
                    self.service_choice = self.fixed if self.fixed is not None else self.service_memory.choose()
            if self.service_choice is not None:
                height, _, _ = STROKES[self.service_choice]
                crossing = falling_crossing(self.toss, height)
                if crossing is not None and tick - 1 <= crossing[0] <= tick + 23:
                    self.serve_plan = crossing
        if self.serve_plan is None:
            return super()._stroke_goals(estimate)
        deadline, xy = self.serve_plan
        horizon = deadline - tick
        if horizon < -4:
            return super()._stroke_goals(estimate)
        height, rate, pitch = STROKES[self.service_choice]
        centre = np.array([*xy, height]) - .027 * np.array([0., np.cos(pitch), np.sin(pitch)])
        def position(dt):
            point = centre.copy()
            point[1] -= np.clip(rate * dt, -.18, .23)
            return point
        self.armed = True
        self.servo.intercept = horizon
        self.servo.stroke = rate
        return position(horizon), position(horizon - 1.), pitch, horizon

    def finish(self, reward):
        result = super().finish(reward)
        if not self.armed:
            result['serve_stroke'] = None
            return result
        tick, xy = self.serve_plan
        observed = outgoing_speed(self.trace, tick)
        # Public point reward can arrive much later in the rally. A coherent
        # forward flight provides partial credit when learning the first strike.
        score = .70 * ((reward + 1.) / 2.) + .30 * float(observed is not None)
        if self.fixed is None:
            self.service_memory.observe(self.service_choice, score)
        result['serve_stroke'] = dict(choice=self.service_choice, tick=tick, xy=xy.tolist(),
                                      departure=observed is not None, reward=score)
        return result
