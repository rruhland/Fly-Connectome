"""Opt-in event-centric local prediction with retinotopic competition."""

from collections import deque

import torch
import torch.nn.functional as F


class MotionWithSeparateSurface:
    """Retain observed evidence without activating it as recurrent sources."""

    def __init__(self, motion):
        self.motion = motion
        self.units = motion.units
        self.reset_state()

    def reset_state(self):
        self.motion.reset_state()
        self.latent = torch.zeros((self.units, 32, 64))
        self.observation = torch.zeros((2, 32, 64))

    @torch.no_grad()
    def step(self, input_pair):
        code, surface = input_pair
        self.motion.step(code)
        self.latent = self.motion.latent.clone()
        self.observation = surface.clone()


class CompetitiveLocalForecast:
    """Each winning unit learns its own local future-event distribution."""

    def __init__(self, units, *, horizon=None, next_event=False, eta=.1):
        if (horizon is None) == (not next_event):
            raise ValueError('choose a frame horizon or next event')
        if horizon is not None and horizon < 1:
            raise ValueError('frame horizon must be positive')
        self.weights = torch.zeros((units, 2, 5, 5))
        self.horizon = horizon
        self.next_event = next_event
        self.eta = eta
        self.reset_state()

    def reset_state(self):
        self.pending = deque()

    def sources(self, state):
        amplitude, winner = state.max(0)
        sites = (amplitude >= .5).nonzero(as_tuple=False)
        return [(int(y), int(x), int(winner[y, x]),
                 float(amplitude[y, x])) for y, x in sites]

    def predict(self, sources):
        canvas = torch.zeros((2, 36, 68))
        coverage = torch.zeros((1, 36, 68))
        for y, x, unit, amplitude in sources:
            canvas[:, y:y+5, x:x+5] += amplitude*self.weights[unit]
            coverage[:, y:y+5, x:x+5] += amplitude
        return canvas[:, 2:-2, 2:-2]/coverage[:, 2:-2, 2:-2].clamp(min=1e-6)

    def credit(self, sources, target):
        if not sources:
            return
        padded = F.pad(target, (2, 2, 2, 2))
        totals = {}
        counts = {}
        for y, x, unit, amplitude in sources:
            if unit not in totals:
                totals[unit] = torch.zeros((2, 5, 5))
                counts[unit] = 0.
            totals[unit] += amplitude*padded[:, y:y+5, x:x+5]
            counts[unit] += amplitude
        for unit, total in totals.items():
            self.weights[unit].lerp_(total/counts[unit], self.eta)

    @torch.no_grad()
    def step(self, state, observed_event, *, learn=False,
             credit_target=None):
        event = bool(observed_event.any())
        target = observed_event if credit_target is None else credit_target
        if self.next_event:
            if not event:
                return None
            if self.pending:
                sources = self.pending.popleft()
                if learn:
                    self.credit(sources, target)
        elif len(self.pending) == self.horizon:
            sources = self.pending.popleft()
            if learn:
                self.credit(sources, target)
        sources = self.sources(state)
        prediction = self.predict(sources)
        self.pending.append(sources)
        return prediction
