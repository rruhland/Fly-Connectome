"""Learn service timing from first-person blade and descending-ball tracks."""
from dataclasses import asdict, dataclass, field

import numpy as np

from .tt4_arena import BALL_RADIUS
from .tt5_player import GRAVITY
from .tt8_service_drive import DrivePlayer


# Contact is possible when the center of the ball reaches the near edge of the
# expanded disc. The RGB blade tracker returns the center of a 12 mm thick disc.
FACE_DISTANCE = BALL_RADIUS + .006
TIMING_LIMITS = (-.35, .80)
REGULARIZATION = np.array([1.5, 4., 3.])


def _toss_arc(trace, deadline):
    rows = [(row['tick'] - deadline, row['ball']) for row in trace
            if deadline - 9 <= row['tick'] <= deadline - 2 and row['ball'] is not None]
    if len(rows) < 4:
        return None
    ticks, positions = zip(*rows)
    times = np.asarray(ticks)
    adjusted = np.asarray(positions) + .5 * times[:, None]**2 * [0., 0., GRAVITY]
    design = np.column_stack((np.ones(len(times)), times))
    origin, velocity = np.linalg.lstsq(design, adjusted, rcond=None)[0]
    residual = np.linalg.norm(adjusted - origin - times[:, None] * velocity, axis=1)
    if residual.max() > .028 or np.linalg.norm(velocity[:2]) > .018 or velocity[2] > -.035:
        return None
    return origin, velocity


def measure_approach(trace, deadline):
    """Interpolate the blade's encounter with the still-falling optical arc."""
    arc = _toss_arc(trace, deadline)
    if arc is None:
        return None
    origin, velocity = arc
    previous = None
    closest = None
    for row in trace:
        time = row['tick'] - deadline
        if not -4. <= time <= 4. or row['blade'] is None or row['blade_pitch'] is None:
            continue
        blade = np.asarray(row['blade'])
        pitch = row['blade_pitch']
        ball = origin + time * velocity - .5 * time**2 * np.array([0., 0., GRAVITY])
        delta = ball - blade
        gap = delta @ [0., np.cos(pitch), np.sin(pitch)] - FACE_DISTANCE
        tangent = delta @ [0., -np.sin(pitch), np.cos(pitch)]
        current = np.array([time, gap, delta[0], tangent, blade[1], pitch])
        if closest is None or abs(gap) < abs(closest[1]):
            closest = current
        if previous is not None:
            interval = time - previous[0]
            if 0. < interval <= 2. and previous[1] > 0. >= gap:
                closure = (previous[1] - gap) / interval
                if closure >= .016:
                    fraction = previous[1] / (previous[1] - gap)
                    lag, _, lateral, vertical, _, angle = previous + fraction * (current - previous)
                    return dict(crossing=True, tick=float(deadline + lag), lag=float(lag),
                                normal_gap=0., lateral=float(lateral), vertical=float(vertical),
                                radius=float(np.hypot(lateral, vertical)),
                                closing_speed=float(closure),
                                forward_speed=float((current[4] - previous[4]) / interval),
                                pitch=float(angle), descent=float(velocity[2] - GRAVITY * lag))
        previous = current
    if closest is None:
        return None
    lag, gap, lateral, vertical, _, pitch = closest
    return dict(crossing=False, tick=float(deadline + lag), lag=float(lag),
                normal_gap=float(gap), lateral=float(lateral), vertical=float(vertical),
                radius=float(np.hypot(lateral, vertical)), closing_speed=None,
                forward_speed=None, pitch=float(pitch),
                descent=float(velocity[2] - GRAVITY * lag))


def context(x, delay):
    """Describe the optical toss and current visual estimate of motor latency."""
    return np.array([1., np.clip((x - .09) / .085, -1.4, 1.4),
                     np.clip((delay - .75) / .68, -1.5, 1.5)])


@dataclass
class ApproachMemory:
    """Fit a short lead correction from signed blade-frame vertical error."""

    samples: list = field(default_factory=list)

    def coefficients(self, contextual=True):
        axes = len(REGULARIZATION) if contextual else 1
        if not self.samples:
            return np.zeros(axes)
        design = np.array([row['context'][:axes] for row in self.samples])
        weights = np.sqrt([row['weight'] for row in self.samples])
        target = np.array([row['lead'] for row in self.samples])
        weighted = weights[:, None] * design
        return np.linalg.solve(weighted.T @ weighted + np.diag(REGULARIZATION[:axes]),
                               weighted.T @ (weights * target))

    def lead(self, features, contextual=True):
        fit = self.coefficients(contextual)
        return float(np.clip(features[:len(fit)] @ fit, *TIMING_LIMITS))

    def observe(self, features, request, measurement, reward):
        if measurement is None or not measurement['crossing']:
            return
        # The falling ball should pass close to the center of the blade rather
        # than grazing the lower or upper edge. An earlier blade crossing sees
        # the ball higher. Preserve successful shots more strongly.
        descent = np.clip(-measurement['descent'], .055, .105)
        shift = (.004 - measurement['vertical']) / descent
        shift = .80 * shift + .20 * (measurement['lag'] - .15)
        weight = .40 if reward > 0 else 1.
        target = request + weight * np.clip(shift, -.75, .75)
        self.samples.append(dict(context=features.tolist(), lead=float(np.clip(target, *TIMING_LIMITS)),
                                 weight=float(weight)))

    def state(self):
        return asdict(self)


class ApproachPlayer(DrivePlayer):
    """Keep the service drive and register its strike with the optical toss."""

    def __init__(self, *args, approach, mode='contextual', learn_approach=False, **kwargs):
        if mode not in ('fixed', 'mean', 'contextual'):
            raise ValueError('unknown optical lead mode')
        if learn_approach and mode == 'fixed':
            raise ValueError('fixed lead cannot learn')
        self.approach = approach
        self.mode = mode
        self.learn_approach = learn_approach
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.features = None
        self.timing_lead = 0.

    def _stroke_goals(self, estimate):
        goal, lead, pitch, horizon = super()._stroke_goals(estimate)
        if self.servo.drive is None:
            return goal, lead, pitch, horizon
        if self.features is None and horizon <= 6.5:
            self.features = context(self.serve_plan[1][0], self.servo.observer.delay)
            if self.mode != 'fixed':
                self.timing_lead = self.approach.lead(self.features, contextual=self.mode == 'contextual')
        if self.timing_lead == 0.:
            return goal, lead, pitch, horizon
        advanced = horizon - self.timing_lead
        windows = np.array([horizon, horizon - 1., advanced, advanced - 1.])
        offsets = np.clip(self.serve_rate * windows, -.107, .185)
        goal[1] += offsets[0] - offsets[2]
        lead[1] += offsets[1] - offsets[3]
        self.servo.intercept = advanced
        return goal, lead, pitch, advanced

    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        blade = self.scene.tracker.actor
        pitch = None
        position = None
        if blade is not None and blade.observed:
            position = list(blade.position)
            pitch = self.scene.tracker.actor_pitch
        self.trace[-1].update(blade=position, blade_pitch=pitch)
        return command

    def finish(self, reward):
        details = super().finish(reward)
        details['precontact'] = None
        if self.serve_plan is not None:
            deadline, _ = self.serve_plan
            measured = measure_approach(self.trace, deadline)
            if self.learn_approach and self.features is not None:
                self.approach.observe(self.features, self.timing_lead, measured, reward)
            details['precontact'] = dict(
                measurement=measured, observations=[
                    {key: row[key] for key in ('tick', 'ball', 'blade', 'blade_pitch')}
                    for row in self.trace if deadline - 9 <= row['tick'] <= deadline + 4
                ], lead=self.timing_lead,
                context=None if self.features is None else self.features.tolist())
        return details
