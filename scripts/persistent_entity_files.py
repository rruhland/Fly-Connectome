"""Opt-in online entity hypotheses over a generic signed visual surface."""

import random

import torch
import torch.nn.functional as F

from confident_event_surface import ConfidentEventSurface


def surface_components(mask, contrast, changed):
    """Connected sensory proposals; identity is assigned by the recurrent files."""
    remaining = set(map(tuple, mask.nonzero(as_tuple=False).tolist()))
    groups = []
    while remaining:
        stack = [remaining.pop()]
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for neighbor in ((y-1, x), (y+1, x), (y, x-1),
                             (y, x+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        cy = sum(y for y, _ in pixels)/len(pixels)
        cx = sum(x for _, x in pixels)/len(pixels)
        feature = torch.zeros((7, 7))
        for y, x in pixels:
            fy, fx = y-round(cy)+3, x-round(cx)+3
            if 0 <= fy < 7 and 0 <= fx < 7:
                feature[fy, fx] = 1.
        groups.append(dict(center=(cy, cx), pixels=pixels,
                           sign=1. if sum(float(contrast[y, x])
                           for y, x in pixels) >= 0 else -1.,
                           feature=feature,
                           changed=any(bool(changed[y, x])
                                       for y, x in pixels)))
    return groups


class PersistentEntityFiles:
    """Compete online for visual proposals and learn local continuation."""

    def __init__(self, *, height=32, width=64, credit='aligned', seed=0):
        if credit not in ('aligned', 'shuffled', 'frozen'):
            raise ValueError('unknown local credit mode')
        self.height = height
        self.width = width
        self.credit = credit
        self.seed = seed
        self.surface = ConfidentEventSurface(height=height, width=width)
        self.reset_state()

    def reset_state(self):
        self.surface.reset_state()
        self.frame = -1
        self.next_id = 0
        self.slots = []
        self.credit_history = []

    @property
    def live_slots(self):
        return [slot for slot in self.slots if slot['confidence'] >= .5]

    def predicted_center(self, slot, frame):
        elapsed = frame-slot['last_seen']
        return tuple(slot['center'][axis]+slot['velocity'][axis]*elapsed
                     for axis in (0, 1))

    def association_cost(self, slot, candidate):
        if slot['sign'] != candidate['sign']:
            return None
        predicted = self.predicted_center(slot, self.frame)
        distance2 = sum((predicted[axis]-candidate['center'][axis])**2
                        for axis in (0, 1))
        gap = self.frame-slot['last_seen']
        if distance2 > (3*max(gap, 1))**2:
            return None
        old = slot['feature'].flatten()
        new = candidate['feature'].flatten()
        similarity = float((old @ new)/(
            old.norm()*new.norm()).clamp(min=1e-8))
        return distance2+4*(1-similarity)

    def new_slot(self, candidate):
        slot = dict(id=self.next_id, center=candidate['center'],
                    last_seen=self.frame, velocity=(0., 0.), period=1,
                    sign=candidate['sign'],
                    feature=candidate['feature'].clone(),
                    pixels=candidate['pixels'], confidence=1.,
                    history=[(self.frame, *candidate['center'])])
        self.next_id += 1
        self.slots.append(slot)

    def update_slot(self, slot, candidate):
        elapsed = max(self.frame-slot['last_seen'], 1)
        observed = tuple((candidate['center'][axis]-slot['center'][axis])
                         / elapsed for axis in (0, 1))
        if self.credit == 'aligned':
            slot['velocity'] = observed
        elif self.credit == 'shuffled' and self.credit_history:
            slot['velocity'] = random.Random(
                self.seed+self.frame+slot['id']).choice(self.credit_history)
        self.credit_history.append(observed)
        slot['center'] = candidate['center']
        slot['last_seen'] = self.frame
        slot['period'] = elapsed
        slot['feature'].lerp_(candidate['feature'], .2)
        slot['pixels'] = candidate['pixels']
        slot['confidence'] = 1.
        slot['history'].append((self.frame, *candidate['center']))

    @torch.no_grad()
    def step(self, event):
        self.frame += 1
        self.surface.step(event)
        for slot in self.slots:
            slot['confidence'] *= .995
        if not bool(event.any()):
            return self.live_slots
        nearby_event = F.max_pool2d(event.sum(0)[None, None], 3,
                                    stride=1, padding=1)[0, 0] > 0
        proposals = [row for row in surface_components(
            self.surface.confident, self.surface.contrast, nearby_event)
            if row['changed']]
        live = self.live_slots
        merged = set()
        if len(proposals) < len(live):
            for index, proposal in enumerate(proposals):
                if sum(self.association_cost(slot, proposal) is not None
                       for slot in live) > 1:
                    merged.add(index)
        options = []
        for slot in live:
            for index, proposal in enumerate(proposals):
                if index in merged:
                    continue
                cost = self.association_cost(slot, proposal)
                if cost is not None:
                    options.append((cost, slot['id'], index))
        options.sort()
        claimed_slots = set()
        claimed_proposals = set(merged)
        for _, slot_id, index in options:
            if slot_id in claimed_slots or index in claimed_proposals:
                continue
            slot = next(row for row in live if row['id'] == slot_id)
            self.update_slot(slot, proposals[index])
            claimed_slots.add(slot_id)
            claimed_proposals.add(index)
        for index, proposal in enumerate(proposals):
            if index not in claimed_proposals:
                self.new_slot(proposal)
        return self.live_slots

    @torch.no_grad()
    def forecast(self):
        current = self.surface.contrast
        future = current.clone()
        slots = self.live_slots
        for slot in slots:
            for y, x in slot['pixels']:
                future[y, x] = 0.
        for slot in slots:
            predicted = self.predicted_center(
                slot, self.frame+max(slot['period'], 1))
            dy = round(predicted[0])-round(slot['center'][0])
            dx = round(predicted[1])-round(slot['center'][1])
            for y, x in slot['pixels']:
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    future[ny, nx] = slot['sign']
        delta = future-current
        return torch.stack(((-delta).clamp(0., 1.),
                            delta.clamp(0., 1.)))
