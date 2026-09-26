"""Opt-in temporal-coherence weighting before local entity association."""

import torch

from event_only_observer_teacher import local_event_support
from local_context_memory import LocalContextMemory


class CorroboratedContextMemory(LocalContextMemory):
    def reset_state(self):
        super().reset_state()
        self.past_events = None

    @torch.no_grad()
    def step(self, sensory, *, learn=False):
        raw = sensory[:2]
        if self.past_events is None:
            self.past_events = torch.zeros_like(raw)
        supported = local_event_support(raw, self.past_events) > 0
        weighted = sensory.clone()
        weighted[:2] *= .1+.9*supported.float()
        self.past_events.mul_(.85).add_(raw)
        return super().step(weighted, learn=learn)
