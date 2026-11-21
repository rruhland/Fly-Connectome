"""Estimate local table response from earlier stereo ball flights during play."""
from collections import deque

import numpy as np

from .tt5_player import BALL_RADIUS, GRAVITY
from .tt7_surface import optical_bounce
from .tt9_precontact_registration import RegistrationPlayer


class SurfaceMemory:
    """A location-weighted average of measured restitution with a shared prior."""

    def __init__(self, width=None, lifetime=8., samples=()):
        self.width = width
        self.lifetime = lifetime
        self.samples = deque(samples, maxlen=60)

    def predict(self, xy, prior):
        if not self.samples:
            return float(prior)
        rows = np.asarray(self.samples)
        ages = np.arange(len(rows) - 1, -1, -1)
        weights = np.exp(-ages / self.lifetime)
        if self.width is not None:
            distance = np.sum((rows[:, :2] - xy) ** 2, axis=1)
            weights *= np.exp(-.5 * distance / self.width ** 2)
        return float(np.clip((weights @ rows[:, 2] + .5 * prior) / (weights.sum() + .5), .35, 1.15))

    def state(self):
        return dict(width=self.width, lifetime=self.lifetime, samples=list(self.samples))


class SurfacePlayer(RegistrationPlayer):
    """Use the location of the next predicted bounce when planning a stroke."""

    def reset(self):
        super().reset()
        self.surface_sightings = deque(maxlen=7)
        self.surface_events = []
        self.last_contact_tick = -100

    def _surface_ratio(self):
        inherited = super()._surface_ratio()
        if self.surface_mode == 'inherited' or self.flight.state is None:
            return inherited
        position, velocity = self.flight.state[:3], self.flight.state[3:]
        # Estimate where the free ball reaches table height, from current optical state.
        height = max(0., position[2] - BALL_RADIUS)
        dt = (velocity[2] + np.sqrt(velocity[2]**2 + 2 * GRAVITY * height)) / GRAVITY
        xy = position[:2] + np.clip(dt, 0., 35.) * velocity[:2]
        return self.surface.predict(xy, inherited)

    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        row = self.trace[-1]
        if row['ball'] is None:
            return command
        tick = row['tick']
        self.surface_sightings.append((tick, row['ball']))
        if tick - self.last_contact_tick <= 9:
            return command
        evidence = optical_bounce(self.surface_sightings)
        if evidence is None:
            return command
        error, ratio, impact = evidence
        times, points = zip(*self.surface_sightings)
        xy = [float(np.interp(impact, times, axis)) for axis in np.asarray(points)[:, :2].T]
        prior = super()._surface_ratio()
        estimate = self.surface.predict(xy, prior) if self.surface_mode != 'inherited' else prior
        self.surface_events.append(dict(observed_at=tick, impact=impact, xy=xy, ratio=ratio,
                                        predicted=estimate, inherited=prior, fit_error=error))
        self.surface.samples.append([*xy, ratio])
        self.last_contact_tick = impact
        self.surface_sightings.clear()
        return command
