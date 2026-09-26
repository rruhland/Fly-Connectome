"""Opt-in recurrent visual hypotheses with soft local spatial evidence."""

import torch
import torch.nn.functional as F

from confident_event_surface import ConfidentEventSurface
from persistent_entity_files import surface_components


class GradedVisualState:
    """Assimilate signed events and occasional intensity without resetting identity."""

    def __init__(self, *, height=32, width=64, diagonal_links=None):
        self.height = height
        self.width = width
        self.diagonal_links = diagonal_links
        self.surface = ConfidentEventSurface(height=height, width=width)
        self.reset_state()

    def reset_state(self):
        self.surface.reset_state()
        self.contrast = torch.zeros((self.height, self.width))
        self.hypotheses = []
        self.next_id = 0
        self.frame = -1

    @property
    def live_slots(self):
        return [row for row in self.hypotheses if row['confidence'] >= .5]

    def _proposals(self, event, absolute):
        if absolute is not None:
            mask = absolute.abs() > .5
            changed = mask
            source = absolute
        else:
            nearby = F.max_pool2d(event.sum(0)[None, None], 3,
                                  stride=1, padding=1)[0, 0] > 0
            mask = self.surface.confident
            changed = nearby
            source = self.surface.contrast
        return [row for row in surface_components(
            mask, source, changed, self.diagonal_links) if row['changed']]

    def _support(self, proposal):
        support = torch.zeros((self.height, self.width))
        for y, x in proposal['pixels']:
            support[y, x] = 1.
        return support

    def _cost(self, hypothesis, proposal):
        if hypothesis['sign'] != proposal['sign']:
            return None
        gap = max(self.frame-hypothesis['last_seen'], 1)
        predicted = tuple(hypothesis['center'][axis]
                          + hypothesis['velocity'][axis]*gap
                          for axis in (0, 1))
        distance2 = sum((predicted[axis]-proposal['center'][axis])**2
                        for axis in (0, 1))
        if distance2 > (3*gap)**2:
            return None
        overlap = sum(float(hypothesis['support'][y, x])
                      for y, x in proposal['pixels'])/len(proposal['pixels'])
        return distance2+2*(1-overlap)

    def _update(self, hypothesis, proposal, *, absolute):
        elapsed = max(self.frame-hypothesis['last_seen'], 1)
        observed = tuple((proposal['center'][axis]-hypothesis['center'][axis])
                         / elapsed for axis in (0, 1))
        hypothesis['velocity'] = observed
        hypothesis['period'] = elapsed
        hypothesis['center'] = proposal['center']
        hypothesis['last_seen'] = self.frame
        hypothesis['support'].lerp_(self._support(proposal), .4)
        gain = .5 if absolute else .3
        hypothesis['confidence'] += gain*(1-hypothesis['confidence'])

    @torch.no_grad()
    def step(self, event, *, absolute=None):
        self.frame += 1
        self.surface.step(event)
        self.contrast.add_(event[1]-event[0]).clamp_(-1., 1.)
        if absolute is not None:
            self.contrast.lerp_(absolute, .35)
            self.surface.contrast.lerp_(absolute, .35)
        proposals = self._proposals(event, absolute)
        for row in self.hypotheses:
            row['confidence'] *= .98
        options = []
        for row in self.hypotheses:
            for index, proposal in enumerate(proposals):
                cost = self._cost(row, proposal)
                if cost is not None:
                    options.append((cost, row['id'], index))
        options.sort()
        matched_ids = set()
        matched_proposals = set()
        for _, identity, index in options:
            if identity in matched_ids or index in matched_proposals:
                continue
            row = next(row for row in self.hypotheses
                       if row['id'] == identity)
            self._update(row, proposals[index], absolute=absolute is not None)
            matched_ids.add(identity)
            matched_proposals.add(index)
        if absolute is not None:
            for row in self.hypotheses:
                if row['id'] in matched_ids:
                    continue
                aligned = (row['support']*(row['sign']*absolute > .5)).sum()
                total = row['support'].sum().clamp(min=1e-8)
                if float(aligned/total) < .25:
                    row['confidence'] *= .7
                    row['support'].mul_(.8)
        for index, proposal in enumerate(proposals):
            if index in matched_proposals:
                continue
            self.hypotheses.append(dict(
                id=self.next_id, center=proposal['center'],
                velocity=(0., 0.), period=1, last_seen=self.frame,
                sign=proposal['sign'], support=self._support(proposal),
                confidence=.8 if absolute is not None else .45))
            self.next_id += 1
        self.hypotheses = [row for row in self.hypotheses
                           if row['confidence'] >= .05]
        return self.live_slots

    @torch.no_grad()
    def forecast(self):
        delta = torch.zeros((self.height, self.width))
        for row in self.live_slots:
            support = row['support']
            dy = round(row['velocity'][0]*max(row['period'], 1))
            dx = round(row['velocity'][1]*max(row['period'], 1))
            shifted = torch.roll(support, shifts=(dy, dx), dims=(0, 1))
            if dy > 0:
                shifted[:dy] = 0
            elif dy < 0:
                shifted[dy:] = 0
            if dx > 0:
                shifted[:, :dx] = 0
            elif dx < 0:
                shifted[:, dx:] = 0
            delta.add_((shifted-support)*row['sign']*row['confidence'])
        return torch.stack(((-delta).clamp(0., 1.),
                            delta.clamp(0., 1.)))
