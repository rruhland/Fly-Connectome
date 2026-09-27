"""Causal entity-local linear recurrence with rotation-equivariant coefficients."""

from collections import deque

import torch


class OnlineEntityDynamics:
    def __init__(self, *, learn=True):
        self.learn = learn
        self.history = deque(maxlen=5)
        self.correlation = torch.eye(8)
        self.prior = torch.tensor([.25, 0.]*4)
        self.cross = self.prior.clone()
        self.coefficients = self.prior.clone()

    @staticmethod
    def features(history):
        scale = history.norm(dim=1).mean().clamp(min=.25)
        values = history/scale
        # Each lag contributes a*I+b*J; coefficients are shared across axes.
        x = torch.stack((values[:, 0], -values[:, 1], values[:, 1], values[:, 0]))
        matrix = torch.stack((x[:2].T.flatten(), x[2:].T.flatten()))
        return matrix, scale

    @torch.no_grad()
    def observe(self, position):
        if position is None:
            self.history.clear()
            return
        if len(self.history) == 5 and self.learn:
            prior = torch.stack(list(self.history))
            displacement = position-prior[-1]
            x, scale = self.features(prior[1:]-prior[:-1])
            self.correlation.add_(x.T @ x)
            self.cross.add_(x.T @ (displacement/scale))
            self.coefficients = torch.linalg.solve(self.correlation, self.cross)
        self.history.append(position.clone())

    @torch.no_grad()
    def predict(self, horizon):
        if len(self.history) < 5:
            return None
        positions = torch.stack(list(self.history))
        history = positions[1:]-positions[:-1]
        forecast = positions[-1].clone()
        for _ in range(horizon):
            x, scale = self.features(history)
            step = x @ self.coefficients*scale
            forecast += step
            history = torch.cat((history[1:], step[None]))
        return forecast
