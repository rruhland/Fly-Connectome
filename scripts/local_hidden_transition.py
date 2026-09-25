"""Opt-in local transition for hidden visual state, separate from camera events."""

import torch
import torch.nn.functional as F

from separated_visual_state import active_sources, local_update, scatter_local


class LocalHiddenTransition:
    def __init__(self, encoder, *, eta=.5):
        self.encoder = encoder
        self.eta = eta
        self.weights = torch.zeros((encoder.units, encoder.units, 5, 5))
        self.reset_state()

    def reset_state(self):
        self.encoder.reset_state()
        self.clear_hidden()

    def clear_hidden(self):
        self.observed = torch.zeros((self.encoder.units, 32, 64))
        self.state = torch.zeros_like(self.observed)
        self.pending = None
        self.previous_sources = []

    @torch.no_grad()
    def step(self, primitive, *, learn=False, credit_target=None):
        self.encoder.step(primitive)
        observed = self.encoder.latent.clone()
        evidence = F.max_pool2d((observed.sum(0) > 0).float()[None, None],
                                5, stride=1, padding=2)[0, 0].bool()
        if learn and self.pending is not None:
            target = observed if credit_target is None else credit_target
            error = (target-self.pending)*evidence
            local_update(self.weights, self.previous_sources, error, self.eta)
        imagined = (self.pending if self.pending is not None
                    else torch.zeros_like(observed))
        self.observed = observed
        self.state = torch.where(evidence[None], observed, imagined)
        self.previous_sources = active_sources(self.state)
        self.pending = scatter_local(self.weights, self.previous_sources)
        return self.pending
