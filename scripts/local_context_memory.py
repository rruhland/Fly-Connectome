"""Opt-in local association of visible context with later motion change."""

import torch

from visual_history_controls import GenericMultiTracker


class LocalContextMemory:
    """Causal tracklets learn context-conditioned displacement at reappearance."""

    def __init__(self, *, height=32, width=64,
                 minimum_event_strength=.05):
        self.tracker = GenericMultiTracker(
            height=height, width=width,
            minimum_strength=minimum_event_strength)
        self.effects = torch.zeros(2)
        self.counts = torch.zeros(2)
        self.reset_state()

    def reset_state(self):
        self.tracker.reset_state()
        self.context = {}
        self.confidence = {}

    @staticmethod
    def nearby_context(sensory, position):
        contrast = sensory[6]+sensory[7]
        height, width = contrast.shape
        y, x = (round(float(position[0])), round(float(position[1])))
        left, right = max(0, x-5), min(width, x+6)
        above = contrast[max(0, y-10):max(0, y-5), left:right].sum()
        below = contrast[min(height, y+6):min(height, y+11),
                         left:right].sum()
        difference = float(above-below)
        if abs(difference) <= .3:
            return None
        return 0 if difference > 0 else 1

    @torch.no_grad()
    def step(self, sensory, *, learn=False):
        before = [(slot['position'].clone(), slot['last_seen'])
                  for slot in self.tracker.slots]
        self.tracker.step(sensory)
        for index, slot in enumerate(self.tracker.slots):
            if slot['last_seen'] != self.tracker.frame:
                continue
            if index < len(before):
                old_position, old_seen = before[index]
                if (learn and self.tracker.frame-old_seen > 1 and
                        index in self.context):
                    cue = self.context[index]
                    displacement = slot['position']-old_position
                    if abs(float(displacement[1])) > 1:
                        observed = float(displacement[0]/
                                         displacement[1].abs())
                        self.counts[cue] += 1
                        self.effects[cue] += ((observed-self.effects[cue])
                                              /self.counts[cue])
            if float(slot['velocity'].norm()) > .25:
                cue = self.nearby_context(sensory, slot['position'])
                if cue is not None:
                    self.context[index] = cue
        field = torch.zeros((self.tracker.height, self.tracker.width))
        for index, slot in enumerate(self.tracker.slots):
            velocity = slot['velocity']
            if float(velocity.norm()) <= .25:
                continue
            age = self.tracker.frame-slot['last_seen']
            if age > 64:
                continue
            center = slot['position']+age*velocity
            if index in self.context:
                center[0] += (age*abs(float(velocity[1]))*
                              self.effects[self.context[index]])
            if age == 0:
                self.confidence[index] = 1.
            else:
                cy, cx = (round(float(center[0])), round(float(center[1])))
                supported = (0 <= cy < self.tracker.height and
                             0 <= cx < self.tracker.width and
                             float(sensory[6:8, cy, cx].sum()) >= .2)
                if not supported:
                    self.confidence[index] = (.35*self.confidence.get(
                        index, 1.))
            radius = 1.5+.15*age
            field = torch.maximum(field, self.confidence.get(index, 1.)*
                                  torch.exp(-(
                (self.tracker.y-center[0]).square()+
                (self.tracker.x-center[1]).square())/(2*radius**2)))
        return field
