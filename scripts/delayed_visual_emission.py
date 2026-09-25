"""Opt-in local future-event readout from recurrent visual state."""

from collections import deque

import torch

from local_observation_surface import LocalObservationSurface
from separated_visual_state import active_sources, local_update, scatter_local


class RecentObservationSurface:
    """Persistent signed evidence plus a local decaying event history."""

    def __init__(self, *, decay):
        self.decay = decay
        self.reset_state()

    def reset_state(self):
        self.surface = LocalObservationSurface()
        self.trace = torch.zeros((2, 32, 64))

    @property
    def channels(self):
        return torch.cat((self.surface.channels, self.trace))

    @torch.no_grad()
    def step(self, event):
        self.surface.step(event)
        self.trace.mul_(self.decay).add_(event).clamp_(0, 1)
        return self.channels


class SurfaceMotionEncoder:
    """Keep local camera memory beside the existing learned motion code."""

    def __init__(self, motion, *, observation_channels=2):
        self.motion = motion
        self.units = motion.units+observation_channels
        self.reset_state()

    def reset_state(self):
        self.motion.reset_state()
        self.latent = torch.zeros((self.units, 32, 64))

    @torch.no_grad()
    def step(self, input_pair):
        primitive, surface_channels = input_pair
        self.motion.step(primitive)
        self.latent = torch.cat((self.motion.latent, surface_channels))


class DelayedLocalEmission:
    """A shared local synapse learns when its own forecast becomes observable."""

    def __init__(self, units, *, horizon=None, next_event=False, eta=.5):
        if (horizon is None) == (not next_event):
            raise ValueError('choose a frame horizon or next event')
        if horizon is not None and horizon < 1:
            raise ValueError('frame horizon must be positive')
        self.weights = torch.zeros((2, units, 5, 5))
        self.horizon = horizon
        self.next_event = next_event
        self.eta = eta
        self.reset_state()

    def reset_state(self):
        self.pending = deque()

    @torch.no_grad()
    def step(self, state, observed_event, *, learn=False,
             credit_target=None):
        event = bool(observed_event.any())
        target = observed_event if credit_target is None else credit_target
        if self.next_event:
            if not event:
                return None
            if self.pending:
                sources, prediction = self.pending.popleft()
                if learn:
                    local_update(self.weights, sources,
                                 target-prediction, self.eta)
        elif len(self.pending) == self.horizon:
            sources, prediction = self.pending.popleft()
            if learn:
                local_update(self.weights, sources,
                             target-prediction, self.eta)
        sources = active_sources(state)
        prediction = scatter_local(self.weights, sources)
        self.pending.append((sources, prediction))
        return prediction
