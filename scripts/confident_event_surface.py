"""Opt-in signed event surface with local evidence confidence."""

import torch
import torch.nn.functional as F


class ConfidentEventSurface:
    """Retain ON/OFF contrast while treating unsupported events as uncertain."""

    def __init__(self, *, height=32, width=64, decay=.98):
        self.height = height
        self.width = width
        self.decay = decay
        self.neighborhood = torch.ones((1, 1, 5, 5))
        self.reset_state()

    def reset_state(self):
        self.contrast = torch.zeros((self.height, self.width))
        self.confidence = torch.zeros_like(self.contrast)

    @property
    def confident(self):
        return (self.contrast.abs() > .5) & (self.confidence >= .5)

    @torch.no_grad()
    def step(self, events):
        active = events.sum(0) > 0
        previous_near = F.max_pool2d(
            self.confidence[None, None], 5, stride=1, padding=2)[0, 0]
        neighbors = F.conv2d(active.float()[None, None],
                             self.neighborhood, padding=2)[0, 0]-active.float()
        confirmation = (.25+.75*previous_near).clamp(max=1.)
        confirmation = torch.where(neighbors > 0, .8, confirmation)
        self.contrast.add_(events[1]-events[0]).clamp_(-1., 1.)
        self.confidence.mul_(self.decay)
        self.confidence[active] = torch.maximum(
            self.confidence[active], confirmation[active])
        return self.confident
