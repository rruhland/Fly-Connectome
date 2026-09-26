"""Opt-in full-resolution recurrent state with strictly local credit."""

import torch
import torch.nn.functional as F


class RecurrentTemporalState:
    """Signed sensory populations with learned nearby state and event transitions."""

    def __init__(self, *, height=32, width=64):
        self.height = height
        self.width = width
        self.decay = torch.tensor((.55, .8, .95, .55, .8, .95))[:, None, None]
        self.transition = torch.zeros((6, 6, 3, 3))
        for channel in range(6):
            self.transition[channel, channel, 1, 1] = .5
        self.emission = torch.zeros((2, 6, 5, 5))
        self.reset_state()

    def reset_state(self):
        self.state = torch.zeros((6, self.height, self.width))

    @staticmethod
    def _compete(activity):
        activity = activity.clamp(min=0.)
        return activity / activity.sum(0, keepdim=True).clamp(min=1.)

    def _sensory_target(self, source, event):
        sensory = event.repeat_interleave(3, dim=0)
        return self._compete(self.decay*source+sensory)

    @torch.no_grad()
    def step(self, event, *, recurrent=True):
        source = self.state
        sensory_target = self._sensory_target(source, event)
        predicted = (F.conv2d(source[None], self.transition, padding=1)[0]
                     if recurrent else torch.zeros_like(source))
        self.state = self._compete(.5*sensory_target+.5*predicted)
        return self.state.clone()

    @torch.no_grad()
    def forecast(self, source=None):
        source = self.state if source is None else source
        return F.conv2d(source[None], self.emission, padding=2)[0].clamp(0., 1.)

    @torch.no_grad()
    def credit_transition(self, source, next_event):
        target = self._sensory_target(source, next_event)
        predicted = F.conv2d(source[None], self.transition, padding=1)[0]
        patches = F.unfold(source[None], kernel_size=3, padding=1)[0]
        exposure = patches.sum(1).clamp(min=1.)
        update = (target-predicted).reshape(6, -1) @ patches.T
        self.transition.add_(.05*(update/exposure[None]).reshape_as(
            self.transition)).clamp_(-.5, .5)

    @torch.no_grad()
    def credit_emission(self, source, next_event):
        predicted = self.forecast(source)
        patches = F.unfold(source[None], kernel_size=5, padding=2)[0]
        exposure = patches.sum(1).clamp(min=1.)
        update = (next_event-predicted).reshape(2, -1) @ patches.T
        self.emission.add_(.05*(update/exposure[None]).reshape_as(
            self.emission)).clamp_(-.5, .5)
