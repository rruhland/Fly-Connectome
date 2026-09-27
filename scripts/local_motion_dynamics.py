"""Opt-in competitive local dynamics associations over observed motion history."""

import torch


class LocalMotionDynamics:
    def __init__(self):
        self.keys = torch.empty(0, 8)
        self.values = torch.empty(0, 2)
        self.counts = torch.empty(0)

    @staticmethod
    def encode(history):
        magnitudes = history.norm(dim=1)
        scale = magnitudes.mean().clamp(min=.25)
        indices = (magnitudes > 1e-6).nonzero().flatten()
        forward = (history[indices[-1]]/magnitudes[indices[-1]] if len(indices)
                   else torch.tensor([0., 1.]))
        lateral = torch.stack((-forward[1], forward[0]))
        basis = torch.stack((forward, lateral))
        return (history @ basis.T/scale).flatten(), basis, scale

    @torch.no_grad()
    def observe(self, history, next_displacement):
        key, basis, scale = self.encode(history)
        value = basis @ next_displacement/scale
        if len(self.keys):
            distance, index = (self.keys-key).square().mean(1).min(0)
        else:
            distance, index = torch.tensor(float('inf')), torch.tensor(0)
        if distance > .01 and len(self.keys) < 128:
            self.keys = torch.cat((self.keys, key[None]))
            self.values = torch.cat((self.values, value[None]))
            self.counts = torch.cat((self.counts, torch.ones(1)))
        else:
            self.counts[index] += 1
            rate = 1/self.counts[index]
            self.keys[index].lerp_(key, rate)
            self.values[index].lerp_(value, rate)

    @torch.no_grad()
    def predict(self, history):
        if not len(self.keys):
            return history[-1].clone()
        key, basis, scale = self.encode(history)
        distance = (self.keys-key).square().mean(1)
        nearest, indices = distance.topk(min(4, len(distance)), largest=False)
        weights = 1/(nearest+1e-6)
        value = (self.values[indices]*weights[:, None]).sum(0)/weights.sum()
        return basis.T @ value*scale

    @torch.no_grad()
    def rollout(self, history, horizon):
        history = history.clone()
        displacement = torch.zeros(2)
        for _ in range(horizon):
            prediction = self.predict(history)
            displacement += prediction
            history = torch.cat((history[1:], prediction[None]))
        return displacement
