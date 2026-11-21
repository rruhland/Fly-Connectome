"""Select a reachable meeting point on a returning trajectory from visual evidence."""
from collections import deque

import numpy as np

from tt3_flight import ForecastHistory
from tt3_numeric_transfer import VisualPlayer


class RecoveryForecast(ForecastHistory):
    def __init__(self, owner, evidence, *, require_rebound=True):
        super().__init__(owner, evidence)
        self.require_rebound = require_rebound
        self.image_path = deque(maxlen=11)
        self.flight_sign = 0
        self.receding = 0
        self.inbound = 0
        self.rebound_seen = False
        self.direction = None
        self.trajectory = None

    def update(self, track, frame):
        previous = self.previous_track
        result = super().update(track, frame)
        self.trajectory = None
        if track is None or not track.observed or self.owner.normal is None:
            return result
        direction = self.owner.normal
        point = np.asarray(track.position)
        speed = np.asarray(track.velocity)
        advancing = float(speed @ direction)
        side = float((point - self.owner.ready) @ direction)
        if not self.flight_sign and abs(advancing) > .003 and advancing * side < 0:
            self.flight_sign = np.sign(advancing)
        if not self.flight_sign:
            return result
        if advancing * self.flight_sign < -.003:
            self.inbound = 0
            self.receding += 1
            self.rebound_seen = False
        elif self.receding >= 4 and advancing * self.flight_sign > .003:
            self.inbound += 1
        if track.id != previous or (self.image_path and self.image_path[-1][0] != frame - 1):
            self.image_path.clear()
        gravity = self.evidence.gravity
        accel = np.zeros(3) if gravity is None else np.where(abs(gravity) > .00015, gravity, 0.)
        if self.inbound == 1:
            self.image_path.clear()
        if self.inbound >= 5 and len(self.image_path) >= 6 and np.linalg.norm(accel) > .0001:
            points = np.asarray([old for _, old in self.image_path])
            prior = (points[-3] - points[-6]) / 3
            closing = float((point - points[-3]) @ accel)
            if closing < -.000003 and prior @ accel > .000001:
                self.rebound_seen = True
                for _ in range(max(0, len(self.image_path) - 3)):
                    self.image_path.popleft()
        self.image_path.append((frame, point))
        if self.inbound >= 5 and len(self.image_path) >= 3:
            times = np.array([tick - frame for tick, _ in self.image_path])
            points = np.stack([old for _, old in self.image_path]) - .5 * times[:, None]**2 * accel
            centered = times - times.mean()
            velocity = centered @ points / (centered @ centered)
            position = points.mean(axis=0) - times.mean() * velocity
            self.trajectory = position, velocity, accel
        return result

    def path(self, ready, normal):
        """Predict a finite set of later positions and valid meeting times."""
        if self.trajectory is None:
            return None
        position, velocity, accel = self.trajectory
        closing = float(velocity @ normal)
        if abs(closing) < .003:
            return None
        nominal = -float((position - ready) @ normal) / closing
        if not -4 <= nominal <= 55:
            return None
        times = np.arange(1., 51.)
        predicted = position + times[:, None] * velocity + .5 * times[:, None]**2 * accel
        eligible = np.full(len(times), self.rebound_seen or not self.require_rebound)
        for axis in range(len(position)):
            g = accel[axis]
            if abs(g) < .00015 or abs(normal[axis]) > .5:
                continue
            sign = 1 if g > 0 else -1
            boundary = self.evidence.boundary(axis, sign)
            if boundary is None:
                continue
            wall, restitution = boundary
            discriminant = velocity[axis]**2 + 2 * g * (wall - position[axis])
            if discriminant <= 0:
                continue
            impact = sign * np.sqrt(discriminant)
            bounce = (impact - velocity[axis]) / g
            if not 0 < bounce <= times[-1]:
                continue
            if not self.require_rebound:
                forecast_reflections(predicted[:, axis], times, bounce, wall, restitution, impact, g)
            elif self.rebound_seen:
                eligible &= times < bounce - 1
            else:
                remaining = np.maximum(0., times - bounce)
                after = wall - restitution * impact * remaining + .5 * g * remaining**2
                predicted[remaining > 0, axis] = after[remaining > 0]
                eligible |= (remaining >= 2.) & (remaining < 2 * restitution * abs(impact / g) - 1)
        distance = (predicted - ready) @ normal
        considered = eligible & (np.abs(distance) <= .095)
        return times[considered], predicted[considered], closing


def forecast_reflections(values, times, bounce, wall, restitution, impact, acceleration):
    while bounce < times[-1]:
        remaining = times - bounce
        future = remaining > 0.
        rebound = -restitution * impact
        duration = -2 * rebound / acceleration
        if duration < 1.:
            values[future] = wall
            break
        values[future] = wall + rebound * remaining[future] + .5 * acceleration * remaining[future]**2
        bounce += duration
        impact = -rebound


class _MeetingMemory:
    def __init__(self, carried, owner):
        self.carried = carried
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.carried, name)

    def goal(self, option, actor, position, velocity, ready, *, normal=None, **kwargs):
        base = self.carried.goal(option, actor, position, velocity, ready, normal=normal, **kwargs)
        if normal is None:
            return base
        path = self.owner.flight.path(ready, normal)
        if path is None or not len(path[0]):
            return base
        times, positions, closing = path
        jacobian = self.owner.motor.matrix(actor)
        inverse = np.linalg.pinv(jacobian, rcond=.13)
        difference = positions - actor
        command_time = np.max(np.abs(difference @ inverse.T), axis=1)
        residual = np.linalg.norm(difference - (difference @ inverse.T) @ jacobian.T, axis=1)
        unreachable = np.maximum(0., command_time - times) / (times + 4.)
        depths = (positions - ready) @ normal
        # Prefer additional flight time on the returning branch when selected.
        # Commit to the deeper position early enough to settle before contact.
        preferred = np.clip(self.owner.defer * closing, -.076, .076)
        compromise = np.abs(depths - preferred)
        cost = 2.1 * unreachable + 20. * residual + 11. * compromise + .12 * command_time / (times + 4.)
        best = int(cost.argmin())
        if cost[best] > 1.4:
            return base
        # Shift the visual meeting station only on a visible return leg. The
        # opening retains the source controller and identical motor probes.
        self.owner.recovery_frames += 1
        self.owner.desired_shift = float((positions[best] - ready) @ normal)
        return positions[best]


class RecoveryPlayer(VisualPlayer):
    def __init__(self, *args, defer=0., require_rebound=True, **kwargs):
        kwargs.setdefault('forecast', True)
        super().__init__(*args, **kwargs)
        self.defer = defer
        self.require_rebound = require_rebound
        self.memory = _MeetingMemory(self.memory, self)

    def reset(self, option=None):
        super().reset(option)
        self.flight = RecoveryForecast(self, self.evidence, require_rebound=self.require_rebound)
        self.recovery_frames = 0
        self.desired_shift = 0.

    def finish(self):
        return dict(**super().finish(), recovery_frames=self.recovery_frames)
