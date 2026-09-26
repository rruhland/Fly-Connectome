"""Opt-in slow visual assemblies selected by local present/future pairs."""

import torch
import torch.nn.functional as F


class PredictivePairAssemblies:
    """Shared sparse units with local delayed pairing and persistent state."""

    def __init__(self, *, fast_units=8, units=16, height=32, width=64,
                 max_winners=8, decay=.85, eta=.02):
        self.fast_units = fast_units
        self.units = units
        self.height = height
        self.width = width
        self.max_winners = max_winners
        self.decay = decay
        self.eta = eta
        generator = torch.Generator().manual_seed(37)
        self.input_templates = .05*torch.rand(
            (units, 2*fast_units, 7, 7), generator=generator)
        self.future_templates = .05*torch.rand(
            (units, fast_units, 11, 11), generator=generator)
        self.reset_state()

    def reset_state(self):
        self.previous_fast = torch.zeros((self.fast_units, self.height,
                                          self.width))
        self.state = torch.zeros((self.units, self.height//4,
                                  self.width//4))

    @torch.no_grad()
    def select(self, observed, future=None, *, use_future=False):
        weights = self.input_templates / self.input_templates.flatten(
            1).norm(dim=1).clamp(min=1e-6)[:, None, None, None]
        scores = F.conv2d(observed[None], weights, stride=4, padding=3)[0]
        if use_future:
            if future is None:
                raise ValueError('future evidence is required for pair training')
            weights = self.future_templates / self.future_templates.flatten(
                1).norm(dim=1).clamp(min=1e-6)[:, None, None, None]
            scores += .5*F.conv2d(future[None], weights, stride=4,
                                   padding=5)[0]
        support = F.max_pool2d(observed[:self.fast_units].sum(0)[None,
                               None], 7, stride=4, padding=3)[0, 0] > 0
        amplitude, channel = scores.max(0)
        sites = [(float(amplitude[y, x]), y, x, int(channel[y, x]))
                 for y, x in support.nonzero(as_tuple=False).tolist()
                 if amplitude[y, x] > 0]
        sites.sort(reverse=True)
        result = torch.zeros_like(self.state)
        selected = []
        for _, y, x, unit in sites:
            if any(max(abs(y-oy), abs(x-ox)) <= 1
                   for oy, ox in selected):
                continue
            result[unit, y, x] = 1.
            selected.append((y, x))
            if len(selected) == self.max_winners:
                break
        return result

    @torch.no_grad()
    def credit_pair(self, observed, future, activity):
        present = F.pad(observed, (3, 3, 3, 3))
        upcoming = F.pad(future, (5, 5, 5, 5))
        for unit, y, x in activity.nonzero(as_tuple=False).tolist():
            self.input_templates[unit].lerp_(
                present[:, 4*y:4*y+7, 4*x:4*x+7], self.eta)
            self.future_templates[unit].lerp_(
                upcoming[:, 4*y:4*y+11, 4*x:4*x+11], self.eta)

    @torch.no_grad()
    def step(self, fast):
        self.state.mul_(self.decay)
        if not bool(fast.any()):
            return torch.zeros_like(self.state)
        observed = torch.cat((fast, self.previous_fast))
        winners = self.select(observed)
        self.state = torch.maximum(self.state, winners)
        self.previous_fast = fast.clone()
        return winners

    @torch.no_grad()
    def predict_fast(self):
        return F.conv_transpose2d(
            self.state[None], self.future_templates,
            stride=4, padding=5, output_padding=3)[0].clamp(0., 1.)

    @torch.no_grad()
    def forecast(self, fast_dictionary, baseline_fast):
        combined = (baseline_fast+self.predict_fast()).clamp(0., 1.)
        return F.conv_transpose2d(combined[None], fast_dictionary,
                                  padding=2)[0].clamp(0., 1.)
