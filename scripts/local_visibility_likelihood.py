"""Opt-in locally credited probability of seeing a future camera event."""

import torch
import torch.nn.functional as F


def nearby_events(events):
    return F.max_pool2d((events.sum(0) > 0).float()[None, None], 5,
                        stride=1, padding=2)[0, 0]


class LocalVisibilityHead:
    def __init__(self, *, eta=.1):
        self.eta = eta
        self.weights = torch.zeros(4)
        self.reset_state()

    def reset_state(self):
        self.previous_sites = None
        self.previous_features = None
        self.previous_values = None
        self.probability = torch.zeros((32, 64))

    @torch.no_grad()
    def step(self, transition, events, *, learn=False, credit_target=None):
        if learn and self.previous_sites is not None:
            observed = nearby_events(events if credit_target is None
                                     else credit_target)
            y, x = self.previous_sites.T
            error = observed[y, x]-self.previous_values
            self.weights += self.eta*(error[:, None]
                                      *self.previous_features).mean(0)
        pending = transition.pending
        sites = (pending.sum(0) >= .5).nonzero(as_tuple=False)
        self.probability = torch.zeros((32, 64))
        self.previous_sites = sites if len(sites) else None
        if len(sites) == 0:
            self.previous_features = None
            self.previous_values = None
            return self.probability
        y, x = sites.T
        mass = pending.sum(0)
        age = (transition.pending_age*pending).sum(0)/mass.clamp(min=1e-6)
        nearby = F.max_pool2d((transition.observed.sum(0) > 0).float()[
            None, None], 5, stride=1, padding=2)[0, 0]
        self.previous_features = torch.stack((
            torch.ones(len(sites)),
            (mass[y, x]/pending.shape[0]).clamp(max=1),
            age[y, x]/(1+age[y, x]), nearby[y, x]), dim=1)
        self.previous_values = torch.sigmoid(
            self.previous_features @ self.weights)
        self.probability[y, x] = self.previous_values
        return self.probability
