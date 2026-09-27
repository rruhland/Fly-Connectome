"""Competitive local linear temporal maps learned from observed correlations."""

import torch

from local_motion_dynamics import LocalMotionDynamics


class LocalGenerativeDynamics(LocalMotionDynamics):
    def __init__(self):
        super().__init__()
        self.correlations = torch.empty(0, 4, 4)
        self.cross = torch.empty(0, 4)
        self.coefficients = torch.empty(0, 4)

    @torch.no_grad()
    def observe(self, history, displacement):
        key, basis, scale = self.encode(history)
        local = key.reshape(4, 2)
        target = basis @ displacement/scale
        if len(self.keys):
            distance, index = (self.keys-key).square().mean(1).min(0)
        else:
            distance, index = torch.tensor(float('inf')), torch.tensor(0)
        if distance > .01 and len(self.keys) < 128:
            index = len(self.keys)
            self.keys = torch.cat((self.keys, key[None]))
            self.counts = torch.cat((self.counts, torch.zeros(1)))
            self.correlations = torch.cat((self.correlations, torch.eye(4)[None]))
            self.cross = torch.cat((self.cross, torch.zeros(1, 4)))
            self.coefficients = torch.cat((self.coefficients, torch.zeros(1, 4)))
        self.counts[index] += 1
        self.keys[index].lerp_(key, 1/self.counts[index])
        self.correlations[index].add_(local @ local.T)
        self.cross[index].add_(local @ target)
        self.coefficients[index] = torch.linalg.solve(self.correlations[index], self.cross[index])

    @torch.no_grad()
    def predict(self, history):
        if not len(self.keys):
            return torch.zeros(2)
        key, basis, scale = self.encode(history)
        distance = (self.keys-key).square().mean(1)
        values, indices = distance.topk(min(4, len(distance)), largest=False)
        weights = 1/(values+1e-6)
        coefficient = (weights[:, None]*self.coefficients[indices]).sum(0)/weights.sum()
        return basis.T @ (coefficient @ key.reshape(4, 2))*scale
