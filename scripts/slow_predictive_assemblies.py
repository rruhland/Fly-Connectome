"""Opt-in slow locally plastic assembly field."""

import torch
import torch.nn.functional as F


class SlowPredictiveAssemblies:
    """Sparse spatially shared motifs with persistent recurrent context."""

    def __init__(self, *, fast_units=8, units=16, height=32, width=64,
                 max_winners=8, decay=.85, sensory_eta=.02,
                 recurrent_eta=.05):
        self.fast_units = fast_units
        self.units = units
        self.height = height
        self.width = width
        self.max_winners = max_winners
        self.decay = decay
        self.sensory_eta = sensory_eta
        self.recurrent_eta = recurrent_eta
        generator = torch.Generator().manual_seed(29)
        self.dictionary = .05*torch.rand((units, 2*fast_units, 7, 7),
                                          generator=generator)
        self.transitions = torch.zeros((units, units, 3, 3))
        self.reset_state()

    def reset_state(self):
        self.previous_fast = torch.zeros((self.fast_units, self.height,
                                          self.width))
        self.state = torch.zeros((self.units, self.height//4,
                                  self.width//4))
        self.winners = torch.zeros_like(self.state)

    @torch.no_grad()
    def predict_field(self, source=None):
        source = self.state if source is None else source
        return F.conv2d(source[None], self.transitions,
                        padding=1)[0].clamp(min=0., max=1.)

    @torch.no_grad()
    def forecast(self, fast_dictionary):
        predicted = self.predict_field()
        future_fast = F.conv_transpose2d(
            predicted[None], self.dictionary[:, :self.fast_units],
            stride=4, padding=3, output_padding=3)[0]
        return F.conv_transpose2d(
            future_fast[None], fast_dictionary,
            padding=2)[0].clamp(min=0., max=1.)

    @torch.no_grad()
    def step(self, fast, *, learn_sensory=False):
        self.state.mul_(self.decay)
        self.winners.zero_()
        if not bool(fast.any()):
            return self.winners.clone()
        observed = torch.cat((fast, self.previous_fast))
        feed = F.conv2d(observed[None], self.dictionary,
                        stride=4, padding=3)[0]
        scores = feed+.5*self.predict_field()
        support = F.max_pool2d(fast.sum(0)[None, None], 7,
                               stride=4, padding=3)[0, 0] > 0
        amplitude, channel = scores.max(0)
        sites = [(float(amplitude[y, x]), y, x, int(channel[y, x]))
                 for y, x in support.nonzero(as_tuple=False).tolist()
                 if amplitude[y, x] > 0]
        sites.sort(reverse=True)
        selected = []
        for _, y, x, unit in sites:
            if any(max(abs(y-oy), abs(x-ox)) <= 1
                   for oy, ox in selected):
                continue
            self.winners[unit, y, x] = 1.
            selected.append((y, x))
            if len(selected) == self.max_winners:
                break
        self.state = torch.maximum(self.state, self.winners)
        if learn_sensory:
            self.credit_sensory(observed, self.winners)
        self.previous_fast = fast.clone()
        return self.winners.clone()

    @torch.no_grad()
    def credit_sensory(self, observed, activity):
        reconstruction = F.conv_transpose2d(
            activity[None], self.dictionary,
            stride=4, padding=3, output_padding=3)[0]
        error = F.pad(observed-reconstruction, (3, 3, 3, 3))
        for unit, y, x in activity.nonzero(as_tuple=False).tolist():
            self.dictionary[unit] += self.sensory_eta*error[
                :, 4*y:4*y+7, 4*x:4*x+7]
        self.dictionary.clamp_(min=0., max=1.)

    @torch.no_grad()
    def credit_transition(self, previous, current):
        error = F.pad(current-self.predict_field(previous),
                      (1, 1, 1, 1))
        for unit, y, x in previous.nonzero(as_tuple=False).tolist():
            self.transitions[:, unit] += self.recurrent_eta*error[
                :, y:y+3, x:x+3].flip(-2, -1)
        self.transitions.clamp_(min=0., max=1.)
