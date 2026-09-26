"""Opt-in locally plastic sparse recurrent visual field."""

import torch
import torch.nn.functional as F


class SparseRecurrentField:
    """Shared sensory motifs and local error-trained recurrent transitions."""

    def __init__(self, *, units=8, height=32, width=64,
                 max_winners=16, sensory_eta=.02, recurrent_eta=.05):
        self.units = units
        self.height = height
        self.width = width
        self.max_winners = max_winners
        self.sensory_eta = sensory_eta
        self.recurrent_eta = recurrent_eta
        generator = torch.Generator().manual_seed(17)
        self.dictionary = .1*torch.rand((units, 2, 5, 5),
                                         generator=generator)
        self.transitions = torch.zeros((units, units, 7, 7))
        self.reset_state()

    def reset_state(self):
        self.state = torch.zeros((self.units, self.height, self.width))

    @torch.no_grad()
    def predict_field(self, source=None):
        source = self.state if source is None else source
        return F.conv2d(source[None], self.transitions,
                        padding=3)[0].clamp(min=0., max=1.)

    @torch.no_grad()
    def forecast(self):
        return F.conv_transpose2d(self.predict_field()[None],
                                  self.dictionary, padding=2)[0].clamp(
                                      min=0., max=1.)

    @torch.no_grad()
    def step(self, event, *, learn_sensory=False):
        if not bool(event.any()):
            return self.state.clone()
        feed = F.conv2d(event[None], self.dictionary, padding=2)[0]
        scores = feed+.5*self.predict_field()
        support = F.max_pool2d(event.sum(0)[None, None], 5,
                               stride=1, padding=2)[0, 0] > 0
        amplitude, channel = scores.max(0)
        sites = [(float(amplitude[y, x]), y, x, int(channel[y, x]))
                 for y, x in support.nonzero(as_tuple=False).tolist()
                 if amplitude[y, x] > 0]
        sites.sort(reverse=True)
        result = torch.zeros_like(self.state)
        selected = []
        for _, y, x, unit in sites:
            if any(max(abs(y-oy), abs(x-ox)) <= 2
                   for oy, ox in selected):
                continue
            result[unit, y, x] = 1.
            selected.append((y, x))
            if len(selected) == self.max_winners:
                break
        self.state = result
        if learn_sensory:
            self.credit_sensory(event, result)
        return result.clone()

    @torch.no_grad()
    def credit_sensory(self, event, activity):
        reconstruction = F.conv_transpose2d(
            activity[None], self.dictionary, padding=2)[0]
        error = F.pad(event-reconstruction, (2, 2, 2, 2))
        for unit, y, x in activity.nonzero(as_tuple=False).tolist():
            self.dictionary[unit] += self.sensory_eta*error[
                :, y:y+5, x:x+5]
        self.dictionary.clamp_(min=0., max=1.)

    @torch.no_grad()
    def credit_transition(self, previous, current):
        error = F.pad(current-self.predict_field(previous),
                      (3, 3, 3, 3))
        for unit, y, x in previous.nonzero(as_tuple=False).tolist():
            self.transitions[:, unit] += self.recurrent_eta*error[
                :, y:y+7, x:x+7].flip(-2, -1)
        self.transitions.clamp_(min=0., max=1.)
