"""Opt-in local temporal transition over predictive-pair assemblies."""

import copy

import torch
import torch.nn.functional as F


class TemporalPairWorldState:
    """Predict sparse future identity and decode a signed sensory correction."""

    def __init__(self, paired, *, recurrent_eta=.05, decoder_eta=.02):
        self.paired = copy.deepcopy(paired)
        self.units = paired.units
        self.height = paired.height
        self.width = paired.width
        self.recurrent_eta = recurrent_eta
        self.decoder_eta = decoder_eta
        self.transitions = torch.zeros((self.units, self.units, 3, 3))
        for unit in range(self.units):
            self.transitions[unit, unit, 1, 1] = .5
        self.decoder = torch.zeros((self.units, paired.fast_units, 11, 11))

    @property
    def state(self):
        return self.paired.state

    def reset_state(self):
        self.paired.reset_state()

    @torch.no_grad()
    def step(self, fast):
        return self.paired.step(fast)

    @torch.no_grad()
    def predict_state(self, source=None):
        source = self.state if source is None else source
        return F.conv2d(source[None], self.transitions,
                        padding=1)[0].clamp(0., 1.)

    @torch.no_grad()
    def decode(self, activity):
        return F.conv_transpose2d(
            activity[None], self.decoder, stride=4,
            padding=5, output_padding=3)[0]

    @torch.no_grad()
    def forecast(self, fast_dictionary, baseline_fast):
        combined = (baseline_fast+self.decode(self.predict_state())).clamp(
            0., 1.)
        return F.conv_transpose2d(combined[None], fast_dictionary,
                                  padding=2)[0].clamp(0., 1.)

    @torch.no_grad()
    def credit_decoder(self, activity, target):
        error = F.pad(target-self.decode(activity), (5, 5, 5, 5))
        for unit, y, x in activity.nonzero(as_tuple=False).tolist():
            self.decoder[unit] += self.decoder_eta*error[
                :, 4*y:4*y+11, 4*x:4*x+11]
        self.decoder.clamp_(-1., 1.)

    @torch.no_grad()
    def credit_transition(self, previous, current):
        error = F.pad(current-self.predict_state(previous),
                      (1, 1, 1, 1))
        for unit, y, x in previous.nonzero(as_tuple=False).tolist():
            self.transitions[:, unit] += self.recurrent_eta*error[
                :, y:y+3, x:x+3].flip(-2, -1)
        self.transitions.clamp_(0., 1.)
