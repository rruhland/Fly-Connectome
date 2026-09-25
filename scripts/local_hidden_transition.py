"""Opt-in local transition for hidden visual state, separate from camera events."""

import torch
import torch.nn.functional as F

from separated_visual_state import active_sources, local_update


def scatter_unclamped(weights, sources):
    canvas = torch.zeros((weights.shape[0], 36, 68))
    for y, x, unit, amplitude in sources:
        canvas[:, y:y+5, x:x+5] += amplitude*weights[:, unit]
    return canvas[:, 2:-2, 2:-2]


class LocalHiddenTransition:
    def __init__(self, encoder, *, eta=.5, persistence=0.):
        self.encoder = encoder
        self.eta = eta
        self.persistence = persistence
        self.weights = torch.zeros((encoder.units, encoder.units, 5, 5))
        self.reset_state()

    def reset_state(self):
        self.encoder.reset_state()
        self.clear_hidden()

    def clear_hidden(self):
        self.observed = torch.zeros((self.encoder.units, 32, 64))
        self.state = torch.zeros_like(self.observed)
        self.age = torch.zeros_like(self.observed)
        self.pending = None
        self.pending_age = None
        self.previous_sources = []

    @torch.no_grad()
    def step(self, primitive, *, learn=False, credit_target=None,
             advance=True):
        self.encoder.step(primitive)
        observed = self.encoder.latent.clone()
        if not advance:
            self.observed = observed
            return self.pending
        evidence = F.max_pool2d((observed.sum(0) > 0).float()[None, None],
                                5, stride=1, padding=2)[0, 0].bool()
        if learn and self.pending is not None:
            target = observed if credit_target is None else credit_target
            error = (target-self.pending)*evidence
            local_update(self.weights, self.previous_sources, error, self.eta)
        imagined = (self.pending if self.pending is not None
                    else torch.zeros_like(observed))
        imagined_age = (self.pending_age if self.pending_age is not None
                        else torch.zeros_like(observed))
        if self.persistence:
            held = self.persistence*self.state
            imagined_age = torch.where(imagined >= held,
                                       imagined_age, self.age+1)
            imagined = torch.maximum(imagined, held)
        self.observed = observed
        self.state = torch.where(evidence[None], observed, imagined)
        self.age = torch.where(evidence[None], 0., imagined_age)
        self.age *= self.state > 0
        self.previous_sources = active_sources(self.state)
        aged_sources = [(y, x, unit, amplitude*(1+float(self.age[unit, y, x])))
                        for y, x, unit, amplitude in self.previous_sources]
        pending_raw = scatter_unclamped(self.weights, self.previous_sources)
        age_sum = scatter_unclamped(self.weights, aged_sources)
        self.pending_age = age_sum/pending_raw.clamp(min=1e-6)
        self.pending = pending_raw.clamp_(0, 1)
        return self.pending
