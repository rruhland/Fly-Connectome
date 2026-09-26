"""Opt-in local continuity and common-fate recurrent visual state."""

import copy

import torch
import torch.nn.functional as F


class ContinuityBindingField:
    """Learn local temporal transport and co-moving lateral associations."""

    def __init__(self, paired, *, decay=.85):
        self.paired = copy.deepcopy(paired)
        self.units = paired.units
        self.height = paired.height
        self.width = paired.width
        self.decay = decay
        self.temporal_counts = torch.zeros((self.units, self.units, 5, 5))
        for unit in range(self.units):
            self.temporal_counts[unit, unit, 2, 2] = 1.
        self.lateral_counts = torch.zeros_like(self.temporal_counts)
        self.normalize()
        self.reset_state()

    def reset_state(self):
        self.paired.reset_state()
        self.state = torch.zeros((self.units, self.height//4,
                                  self.width//4))

    @torch.no_grad()
    def credit(self, previous, current):
        past = previous.nonzero(as_tuple=False).tolist()
        future = current.nonzero(as_tuple=False).tolist()
        locations = {(y, x) for _, y, x in future}
        for source, y, x in past:
            for target, ny, nx in future:
                dy, dx = ny-y, nx-x
                if max(abs(dy), abs(dx)) <= 2:
                    self.temporal_counts[target, source,
                                         2-dy, 2-dx] += 1.
        for index, (source, y, x) in enumerate(past):
            for neighbor, ny, nx in past[index+1:]:
                dy, dx = ny-y, nx-x
                if max(abs(dy), abs(dx)) > 2:
                    continue
                if any((y+vy, x+vx) in locations and
                       (ny+vy, nx+vx) in locations
                       for vy in (-1, 0, 1) for vx in (-1, 0, 1)):
                    self.lateral_counts[neighbor, source,
                                        2-dy, 2-dx] += 1.
                    self.lateral_counts[source, neighbor,
                                        2+dy, 2+dx] += 1.

    @torch.no_grad()
    def normalize(self):
        self.temporal = self.temporal_counts / self.temporal_counts.sum(
            (0, 2, 3)).clamp(min=1.)[None, :, None, None]
        self.lateral = self.lateral_counts / self.lateral_counts.sum(
            (0, 2, 3)).clamp(min=1.)[None, :, None, None]

    @torch.no_grad()
    def step(self, fast):
        sensory = self.paired.step(fast)
        if not bool(fast.any()):
            self.state.mul_(self.decay)
            return sensory
        continuation = F.conv2d(self.state[None], self.temporal,
                                padding=2)[0]
        binding = F.conv2d(sensory[None], self.lateral, padding=2)[0]
        self.state = (sensory+continuation+binding).clamp(0., 1.)
        return sensory

    @torch.no_grad()
    def forecast(self, fast_dictionary, baseline_fast):
        predicted = F.conv_transpose2d(
            self.state[None], self.paired.future_templates,
            stride=4, padding=5, output_padding=3)[0]
        combined = (baseline_fast+predicted).clamp(0., 1.)
        return F.conv_transpose2d(combined[None], fast_dictionary,
                                  padding=2)[0].clamp(0., 1.)
