"""Opt-in local event-due rate and conditional spatial outcome."""

from collections import deque

import torch
import torch.nn.functional as F


def age_bucket(value):
    if value < .5:
        return 0
    if value < 1.5:
        return 1
    if value < 3.5:
        return 2
    if value < 7.5:
        return 3
    return 4


class HazardLocalForecast:
    def __init__(self, units, *, horizon=4, eta=.1):
        if horizon < 1:
            raise ValueError('frame horizon must be positive')
        self.rate = torch.zeros((units, 5))
        self.spatial = torch.zeros((units, 2, 5, 5))
        self.horizon = horizon
        self.eta = eta
        self.reset_state()

    def reset_state(self):
        self.pending = deque()

    def sources(self, state, age):
        amplitude, winner = state.max(0)
        sites = (amplitude >= .5).nonzero(as_tuple=False)
        return [(int(y), int(x), int(winner[y, x]),
                 float(amplitude[y, x]),
                 age_bucket(float(age[winner[y, x], y, x])))
                for y, x in sites]

    def predict(self, sources):
        canvas = torch.zeros((2, 36, 68))
        coverage = torch.zeros((1, 36, 68))
        for y, x, unit, amplitude, bucket in sources:
            canvas[:, y:y+5, x:x+5] += (
                amplitude*self.rate[unit, bucket]*self.spatial[unit])
            coverage[:, y:y+5, x:x+5] += amplitude
        return (canvas[:, 2:-2, 2:-2]
                /coverage[:, 2:-2, 2:-2].clamp(min=1e-6)).clamp_(0, 1)

    def credit(self, sources, target):
        if not sources:
            return
        padded = F.pad(target, (2, 2, 2, 2))
        rates = {}
        spaces = {}
        for y, x, unit, amplitude, bucket in sources:
            patch = padded[:, y:y+5, x:x+5]
            count = float(patch.sum())
            key = (unit, bucket)
            total, weight = rates.get(key, (0., 0.))
            rates[key] = (total+amplitude*count, weight+amplitude)
            if count:
                if unit not in spaces:
                    spaces[unit] = (torch.zeros((2, 5, 5)), 0.)
                total_patch, weight = spaces[unit]
                spaces[unit] = (total_patch+amplitude*patch/count,
                                weight+amplitude)
        for (unit, bucket), (total, weight) in rates.items():
            self.rate[unit, bucket] += self.eta*(total/weight-
                                                  self.rate[unit, bucket])
        for unit, (total, weight) in spaces.items():
            self.spatial[unit].lerp_(total/weight, self.eta)

    @torch.no_grad()
    def step(self, state, age, observed_event, *, learn=False,
             credit_target=None):
        if len(self.pending) == self.horizon:
            sources = self.pending.popleft()
            if learn:
                target = (observed_event if credit_target is None
                          else credit_target)
                self.credit(sources, target)
        sources = self.sources(state, age)
        prediction = self.predict(sources)
        self.pending.append(sources)
        return prediction
