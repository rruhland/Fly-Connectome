"""Opt-in local credibility learned from paired event and intensity sensing."""

import torch
import torch.nn.functional as F


class LocalObservationModel:
    """Weight incoming signed events by locally learned sensory agreement."""

    def __init__(self, *, height=32, width=64):
        self.height = height
        self.width = width
        self.weights = torch.zeros((2, 8, 5, 5))
        self.bias = torch.zeros((2, 1, 1))
        self.reset_state()

    def reset_state(self):
        self.fast = torch.zeros((2, self.height, self.width))
        self.slow = torch.zeros_like(self.fast)
        self.contrast = torch.zeros((self.height, self.width))

    def _features(self, raw):
        appearance = torch.stack((self.contrast.clamp(min=0.),
                                  (-self.contrast).clamp(min=0.)))
        return torch.cat((raw, self.fast, self.slow, appearance))

    @torch.no_grad()
    def step(self, raw, *, absolute=None):
        features = self._features(raw)
        credibility = torch.sigmoid(F.conv2d(
            features[None], self.weights, padding=2)[0]+self.bias)
        filtered = raw*credibility
        self.fast.mul_(.65).add_(filtered).clamp_(0., 1.)
        self.slow.mul_(.95).add_(filtered).clamp_(0., 1.)
        self.contrast.add_(filtered[1]-filtered[0]).clamp_(-1., 1.)
        if absolute is not None:
            self.contrast.lerp_(absolute, .35)
        return filtered, features

    @torch.no_grad()
    def credit(self, features, raw, observed_change):
        credibility = torch.sigmoid(F.conv2d(
            features[None], self.weights, padding=2)[0]+self.bias)
        error = (observed_change-credibility)*raw
        patches = F.unfold(features[None], kernel_size=5,
                           padding=2)[0]
        count = raw.sum((1, 2)).clamp(min=1.)
        update = error.reshape(2, -1) @ patches.T
        self.weights.add_(.1*(update/count[:, None]).reshape_as(
            self.weights)).clamp_(-2., 2.)
        self.bias.add_(.1*error.sum((1, 2))[:, None, None]
                       / count[:, None, None]).clamp_(-2., 2.)
