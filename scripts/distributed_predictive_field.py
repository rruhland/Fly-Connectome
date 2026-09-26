"""Opt-in distributed recurrent visual state with local predictive plasticity."""

import torch
import torch.nn.functional as F


class DistributedPredictiveField:
    """Learn local next-event kernels over continuous signed sensory history."""

    def __init__(self, *, height=32, width=64):
        self.height = height
        self.width = width
        self.weights = torch.zeros((2, 8, 5, 5))
        self.reset_state()

    def reset_state(self):
        self.fast = torch.zeros((2, self.height, self.width))
        self.slow = torch.zeros_like(self.fast)
        self.contrast = torch.zeros((self.height, self.width))
        self.last_forecast = torch.zeros_like(self.fast)
        self.state = torch.zeros((8, self.height, self.width))

    @torch.no_grad()
    def forecast(self):
        self.last_forecast = F.conv2d(
            self.state[None], self.weights, padding=2)[0].clamp(0., 1.)
        return self.last_forecast.clone()

    @torch.no_grad()
    def step(self, event, *, absolute=None):
        recurrent = .5*self.last_forecast
        self.fast.mul_(.65).add_(event).clamp_(0., 1.)
        self.slow.mul_(.95).add_(event).clamp_(0., 1.)
        self.contrast.add_(event[1]-event[0]).clamp_(-1., 1.)
        if absolute is not None:
            self.contrast.lerp_(absolute, .35)
        appearance = torch.stack((self.contrast.clamp(min=0.),
                                  (-self.contrast).clamp(min=0.)))
        self.state = torch.cat((self.fast, self.slow, appearance,
                                recurrent))
        self.forecast()
        return self.state.clone()

    @torch.no_grad()
    def credit(self, source, target):
        predicted = F.conv2d(source[None], self.weights,
                             padding=2)[0].clamp(0., 1.)
        error = (target-predicted).reshape(2, -1)
        patches = F.unfold(source[None], kernel_size=5,
                           padding=2)[0]
        exposure = patches.sum(1).clamp(min=1.)
        update = (error @ patches.T)/exposure[None]
        self.weights.add_(.05*update.reshape_as(self.weights))
        self.weights.clamp_(-.5, .5)
