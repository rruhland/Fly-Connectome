"""Opt-in local memory of event-camera evidence with explicit unknown state."""

import torch


class LocalObservationSurface:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.known = torch.zeros((32, 64), dtype=torch.bool)
        self.value = torch.zeros((32, 64))

    @torch.no_grad()
    def step(self, events):
        off = events[0] > 0
        on = events[1] > 0
        self.known |= off | on
        self.value[off] = 0
        self.value[on] = 1
        return self.channels

    @property
    def channels(self):
        return torch.stack((self.known & (self.value == 0),
                            self.known & (self.value == 1))).float()
