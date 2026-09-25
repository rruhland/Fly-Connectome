"""Opt-in age-tagged local hidden memory and delayed spatial plasticity."""

from collections import deque

import torch
import torch.nn.functional as F


class RecentLatentMemory:
    def __init__(self, units, *, window):
        if window < 1:
            raise ValueError('memory window must be positive')
        self.units = units
        self.window = window
        self.reset_state()

    def reset_state(self):
        self.activity = torch.zeros((self.units, 32, 64))
        self.age = torch.full((self.units, 32, 64), self.window,
                              dtype=torch.long)

    @torch.no_grad()
    def step(self, state):
        self.age.add_(1).clamp_(max=self.window)
        self.activity.masked_fill_(self.age >= self.window, 0)
        active = state >= .5
        self.activity[active] = state[active]
        self.age[active] = 0
        return self.activity


class LagConditionalForecast:
    """A source's unit and elapsed age select a shared local event synapse."""

    def __init__(self, units, *, window, horizon=4, eta=.1):
        if window < 1 or horizon < 1:
            raise ValueError('memory window and forecast horizon must be positive')
        self.weights = torch.zeros((units, window, 2, 5, 5))
        self.window = window
        self.horizon = horizon
        self.eta = eta
        self.reset_state()

    def reset_state(self):
        self.pending = deque()

    def sources(self, activity, age):
        amplitude, winner = activity.max(0)
        sites = (amplitude >= .5).nonzero(as_tuple=False)
        return [(int(y), int(x), int(winner[y, x]),
                 int(age[winner[y, x], y, x]), float(amplitude[y, x]))
                for y, x in sites]

    def predict(self, sources):
        canvas = torch.zeros((2, 36, 68))
        coverage = torch.zeros((1, 36, 68))
        for y, x, unit, lag, amplitude in sources:
            canvas[:, y:y+5, x:x+5] += amplitude*self.weights[unit, lag]
            coverage[:, y:y+5, x:x+5] += amplitude
        return canvas[:, 2:-2, 2:-2]/coverage[:, 2:-2, 2:-2].clamp(min=1e-6)

    def credit(self, sources, target):
        if not sources:
            return
        padded = F.pad(target, (2, 2, 2, 2))
        totals = {}
        counts = {}
        for y, x, unit, lag, amplitude in sources:
            key = (unit, lag)
            if key not in totals:
                totals[key] = torch.zeros((2, 5, 5))
                counts[key] = 0.
            totals[key] += amplitude*padded[:, y:y+5, x:x+5]
            counts[key] += amplitude
        for (unit, lag), total in totals.items():
            self.weights[unit, lag].lerp_(total/counts[unit, lag], self.eta)

    @torch.no_grad()
    def step(self, activity, age, observed_event, *, learn=False,
             credit_target=None):
        if len(self.pending) == self.horizon:
            sources = self.pending.popleft()
            if learn:
                target = (observed_event if credit_target is None
                          else credit_target)
                self.credit(sources, target)
        sources = self.sources(activity, age)
        prediction = self.predict(sources)
        self.pending.append(sources)
        return prediction
