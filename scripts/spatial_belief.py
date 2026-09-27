"""Bounded local episodic associations retaining distinct future outcomes."""

import math
from collections import deque

import torch


class SpatialBelief:
    def __init__(self, capacity=2048, horizon=1):
        self.capacity = capacity
        self.horizon = horizon
        self.keys = torch.empty(0, 8)
        self.values = torch.empty(0, 2)
        self.seen = 0
        self.generator = torch.Generator().manual_seed(191001)
        self.calibration = MarginalCalibration()

    @staticmethod
    def encode(history):
        scale = history.norm(dim=1).mean().clamp(min=.25)
        direction = history.mean(0)
        if direction.norm() < 1e-6:
            moving = (history.norm(dim=1) > 1e-6).nonzero().flatten()
            direction = history[moving[-1]] if len(moving) else torch.tensor([0., 1.])
        direction = direction/direction.norm()
        basis = torch.stack((direction, torch.stack((-direction[1], direction[0]))))
        return (history @ basis.T/scale).flatten(), basis, scale

    @torch.no_grad()
    def observe(self, history, displacement, *, credit=None):
        if credit is not None:
            self.calibration.observe(mixture_pit(credit[0], credit[1], displacement))
        key, basis, scale = self.encode(history)
        value = basis @ displacement/scale
        self.seen += 1
        if len(self.keys) < self.capacity:
            self.keys = torch.cat((self.keys, key[None]))
            self.values = torch.cat((self.values, value[None]))
        else:
            index = int(torch.randint(self.seen, (), generator=self.generator))
            if index < self.capacity:
                self.keys[index], self.values[index] = key, value

    @torch.no_grad()
    def distribution(self, history):
        if not len(self.keys):
            return (self.horizon*history.mean(0))[None], torch.ones(1)
        key, basis, scale = self.encode(history)
        distance = (self.keys-key).square().mean(1)
        nearest, indices = distance.topk(min(32, len(distance)), largest=False)
        weights = torch.softmax(-nearest/nearest[-1].clamp(min=.01), 0)
        return self.values[indices] @ basis*scale, weights

    def predict(self, history):
        centers, weights = self.distribution(history)
        return (centers*weights[:, None]).sum(0)

    def predict_with_credit(self, history):
        centers, weights = self.distribution(history)
        return (centers*weights[:, None]).sum(0), (centers.clone(), weights.clone())

    def log_prob(self, history, target):
        centers, weights = self.distribution(history)
        return mixture_log_prob(centers, weights, target)


def mixture_log_prob(centers, weights, target):
    return torch.logsumexp(weights.log()-.5*(centers-target).square().sum(1)-math.log(2*math.pi), 0)


def mixture_pit(centers, weights, target):
    return ((.5+.5*torch.erf((target-centers)/math.sqrt(2)))*weights[:, None]).sum(0)


def mixture_quantile(centers, weights, probability):
    low, high = centers.min(0).values-10., centers.max(0).values+10.
    for _ in range(32):
        midpoint = (low+high)/2
        below = mixture_pit(centers, weights, midpoint) < probability
        low = torch.where(below, midpoint, low)
        high = torch.where(below, high, midpoint)
    value = (low+high)/2
    value = torch.where(probability <= 0, -torch.inf, value)
    return torch.where(probability >= 1, torch.inf, value)


class MarginalCalibration:
    """Finite-sample marginal rank intervals from observed future endpoints."""

    def __init__(self):
        self.ranks = deque(maxlen=512)

    def observe(self, pit):
        self.ranks.append(pit.clone())

    def bounds(self):
        if not self.ranks:
            return torch.zeros(2), torch.ones(2)
        ranks = torch.stack(list(self.ranks)).sort(dim=0).values
        n = len(ranks)
        low, high = math.floor(.05*(n+1)), math.ceil(.95*(n+1))
        return (ranks[low-1] if low >= 1 else torch.zeros(2),
                ranks[high-1] if high <= n else torch.ones(2))
