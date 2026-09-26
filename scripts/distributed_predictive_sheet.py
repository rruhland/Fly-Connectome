"""Opt-in distributed locally plastic recurrent visual sheet."""

import torch
import torch.nn.functional as F


class DistributedPredictiveSheet:
    """Locally competitive graded state with delayed signed sensory credit."""

    def __init__(self, *, fast_units=8, units=16, height=32, width=64,
                 decay=.85, sensory_eta=.02, recurrent_eta=.05):
        self.fast_units = fast_units
        self.units = units
        self.height = height
        self.width = width
        self.decay = decay
        self.sensory_eta = sensory_eta
        self.recurrent_eta = recurrent_eta
        generator = torch.Generator().manual_seed(43)
        self.input_templates = .05*torch.rand(
            (units, 2*fast_units, 7, 7), generator=generator)
        self.output_templates = .02*(torch.rand(
            (units, fast_units, 11, 11), generator=generator)-.5)
        self.transitions = torch.zeros((units, units, 3, 3))
        for unit in range(units):
            self.transitions[unit, unit, 1, 1] = .5
        self.reset_state()

    def reset_state(self):
        self.previous_fast = torch.zeros((self.fast_units, self.height,
                                          self.width))
        self.state = torch.zeros((self.units, self.height//4,
                                  self.width//4))

    @torch.no_grad()
    def predict_state(self, source=None):
        source = self.state if source is None else source
        return F.conv2d(source[None], self.transitions,
                        padding=1)[0].clamp(0., 1.)

    @torch.no_grad()
    def evidence(self, observed, future=None, prior=None):
        weights = self.input_templates / self.input_templates.flatten(
            1).norm(dim=1).clamp(min=1e-6)[:, None, None, None]
        scores = F.conv2d(observed[None], weights,
                          stride=4, padding=3)[0]
        if future is not None:
            weights = self.output_templates / self.output_templates.flatten(
                1).norm(dim=1).clamp(min=1e-6)[:, None, None, None]
            scores += .5*F.conv2d(future[None], weights,
                                   stride=4, padding=5)[0]
        if prior is not None:
            scores += prior
        raw = F.max_pool2d(observed[:self.fast_units].sum(0)[None, None],
                           7, stride=4, padding=3)[0, 0] > 0
        positive = (scores-scores.mean(0, keepdim=True)).clamp(min=0.)
        activity = positive/positive.sum(0, keepdim=True).clamp(min=1e-8)
        if prior is None:
            amplitude = raw.float()
        else:
            nearby = F.max_pool2d(raw.float()[None, None], 3,
                                  padding=1, stride=1)[0, 0] > 0
            amplitude = torch.where(raw, 1., prior.sum(0).clamp(0., 1.))
            amplitude *= nearby
        return activity*amplitude[None]

    @torch.no_grad()
    def step(self, fast):
        if not bool(fast.any()):
            self.state.mul_(self.decay)
            return self.state
        observed = torch.cat((fast, self.previous_fast))
        self.state = self.evidence(observed, prior=self.predict_state())
        self.previous_fast = fast.clone()
        return self.state

    @torch.no_grad()
    def decode(self, activity):
        return F.conv_transpose2d(
            activity[None], self.output_templates,
            stride=4, padding=5, output_padding=3)[0]

    @torch.no_grad()
    def forecast(self, fast_dictionary, baseline_fast):
        combined = (baseline_fast+self.decode(self.state)).clamp(0., 1.)
        return F.conv_transpose2d(combined[None], fast_dictionary,
                                  padding=2)[0].clamp(0., 1.)

    @torch.no_grad()
    def credit_pair(self, observed, future_error, activity):
        current_patches = F.unfold(observed[None], 7, stride=4,
                                   padding=3)[0]
        flat = activity.flatten(1)
        mass = flat.sum(1).clamp(min=1e-8)
        average = (flat @ current_patches.T)/mass[:, None]
        target = average.reshape_as(self.input_templates)
        rate = self.sensory_eta*(flat.sum(1) > 0).float()
        self.input_templates += rate[:, None, None, None]*(
            target-self.input_templates)
        residual = future_error-self.decode(activity)
        future_patches = F.unfold(residual[None], 11,
                                  stride=4, padding=5)[0]
        delta = (flat @ future_patches.T)/mass[:, None]
        self.output_templates += (rate[:, None, None, None]*
                                  delta.reshape_as(self.output_templates))
        self.input_templates.clamp_(0., 1.)
        self.output_templates.clamp_(-1., 1.)

    @torch.no_grad()
    def credit_transition(self, previous, current):
        error = current-self.predict_state(previous)
        patches = F.unfold(previous[None], 3, padding=1)[0]
        delta = (error.flatten(1) @ patches.T)/previous.sum().clamp(min=1.)
        self.transitions += self.recurrent_eta*delta.reshape_as(
            self.transitions)
        self.transitions.clamp_(0., 1.)
