"""Opt-in locally plastic latent transition with separate event emission."""

import torch
import torch.nn.functional as F


def active_sources(activity):
    sites = (activity >= .5).nonzero(as_tuple=False)
    return [(int(y), int(x), int(unit), float(activity[unit, y, x]))
            for unit, y, x in sites]


def scatter_local(weights, sources):
    canvas = torch.zeros((weights.shape[0], 36, 68))
    for y, x, unit, amplitude in sources:
        canvas[:, y:y+5, x:x+5] += amplitude*weights[:, unit]
    return canvas[:, 2:-2, 2:-2].clamp_(0, 1)


def local_update(weights, sources, error, eta):
    if not sources:
        return
    padded = F.pad(error, (2, 2, 2, 2))
    denominator = max(sum(source[3] for source in sources), 1.)
    for y, x, unit, amplitude in sources:
        weights[:, unit] += eta*amplitude/denominator*padded[:, y:y+5, x:x+5]
    weights.clamp_(0, 1)


class SeparatedVisualState:
    def __init__(self, encoder, *, recurrence=True,
                 state_eta=.5, emission_eta=.5,
                 separate_credit=False):
        self.encoder = encoder
        self.recurrence = recurrence
        self.state_eta = state_eta
        self.emission_eta = emission_eta
        self.separate_credit = separate_credit
        units = encoder.units
        self.state_weights = torch.zeros((units, units, 5, 5))
        self.emission_observed = torch.zeros((2, units, 5, 5))
        self.emission_imagined = torch.zeros((2, units, 5, 5))
        self.reset_state()

    def reset_state(self):
        self.encoder.reset_state()
        self.pending_state = None
        self.pending_events = None
        self.pending_observed_events = None
        self.pending_imagined_events = None
        self.previous_state_sources = []
        self.previous_observed_sources = []
        self.previous_imagined_sources = []
        self.observed = torch.zeros((self.encoder.units, 32, 64))
        self.imagined = torch.zeros_like(self.observed)
        self.state = torch.zeros_like(self.observed)

    @torch.no_grad()
    def step(self, primitive, events, *, learn=False):
        self.encoder.step(primitive)
        observed = self.encoder.latent.clone()
        if learn and self.pending_events is not None:
            observed_error = events-(
                self.pending_observed_events if self.separate_credit
                else self.pending_events)
            imagined_error = events-(
                self.pending_imagined_events if self.separate_credit
                else self.pending_events)
            local_update(self.emission_observed,
                         self.previous_observed_sources, observed_error,
                         self.emission_eta)
            local_update(self.emission_imagined,
                         self.previous_imagined_sources, imagined_error,
                         self.emission_eta)
            if self.recurrence:
                evidence = F.max_pool2d((observed.sum(0) > 0).float()[
                    None, None], 5, stride=1, padding=2)[0, 0]
                state_error = (observed-self.pending_state)*evidence
                local_update(self.state_weights,
                             self.previous_state_sources, state_error,
                             self.state_eta)
        imagined = (self.pending_state if self.recurrence and
                    self.pending_state is not None else
                    torch.zeros_like(observed))
        state = torch.maximum(observed, imagined)
        observed_sources = active_sources(observed)
        imagined_sources = active_sources(imagined)
        state_sources = active_sources(state)
        state_prediction = (scatter_local(self.state_weights, state_sources)
                            if self.recurrence else torch.zeros_like(state))
        observed_prediction = scatter_local(self.emission_observed,
                                            observed_sources)
        imagined_prediction = scatter_local(self.emission_imagined,
                                            imagined_sources)
        event_prediction = (observed_prediction
                            + imagined_prediction).clamp_(0, 1)
        self.pending_state = state_prediction
        self.pending_events = event_prediction
        self.pending_observed_events = observed_prediction
        self.pending_imagined_events = imagined_prediction
        self.previous_state_sources = state_sources
        self.previous_observed_sources = observed_sources
        self.previous_imagined_sources = imagined_sources
        self.observed = observed
        self.imagined = imagined
        self.state = state
        return event_prediction
