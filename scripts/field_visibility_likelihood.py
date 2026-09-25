"""Opt-in probability of next visible evidence in overlapping local fields."""

import torch
import torch.nn.functional as F


def field_sum(values):
    return F.avg_pool2d(values[None, None], 16, stride=8)[0, 0]*256


def field_events(events):
    return F.max_pool2d((events.sum(0) > 0).float()[None, None],
                        16, stride=8)[0, 0]


class FieldVisibilityHead:
    def __init__(self, *, eta=.05):
        self.eta = eta
        self.weights = torch.zeros(4)
        self.reset_state()

    def reset_state(self):
        self.previous_sites = None
        self.previous_features = None
        self.previous_values = None
        self.probability = torch.zeros((3, 7))

    @torch.no_grad()
    def step(self, transition, events, *, learn=False, credit_target=None):
        if learn and self.previous_sites is not None:
            target = field_events(events if credit_target is None
                                  else credit_target)
            y, x = self.previous_sites.T
            error = target[y, x]-self.previous_values
            self.weights += self.eta*(error[:, None]
                                      *self.previous_features).mean(0)
        pending = transition.pending
        mass = field_sum(pending.sum(0))
        sites = (mass > .1).nonzero(as_tuple=False)
        self.probability = torch.zeros((3, 7))
        self.previous_sites = sites if len(sites) else None
        if len(sites) == 0:
            self.previous_features = None
            self.previous_values = None
            return self.probability
        age_sum = field_sum((transition.pending_age*pending).sum(0))
        evidence = F.max_pool2d((transition.observed.sum(0) > 0).float()[
            None, None], 16, stride=8)[0, 0]
        y, x = sites.T
        age = age_sum[y, x]/mass[y, x].clamp(min=1e-6)
        self.previous_features = torch.stack((
            torch.ones(len(sites)),
            torch.log1p(mass[y, x])/4,
            age/(1+age), evidence[y, x]), dim=1)
        self.previous_values = torch.sigmoid(
            self.previous_features @ self.weights)
        self.probability[y, x] = self.previous_values
        return self.probability
