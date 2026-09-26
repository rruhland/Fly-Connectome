"""Opt-in local motion transition memory over unlabeled event tracklets."""

import torch

from visual_history_controls import GenericMultiTracker


class LocalCadenceMemory:
    """Associate observed speed transitions, then continue state through gaps."""

    def __init__(self, *, height=32, width=64):
        self.tracker = GenericMultiTracker(height=height, width=width)
        self.transitions = torch.zeros((9, 9))
        self.reset_state()

    def reset_state(self):
        self.tracker.reset_state()
        self.previous_bins = {}

    @staticmethod
    def speed_bin(velocity):
        return max(0, min(8, round(2*float(velocity.norm()))))

    @torch.no_grad()
    def step(self, sensory, *, learn=False):
        before = [slot['last_seen'] for slot in self.tracker.slots]
        self.tracker.step(sensory)
        for index, slot in enumerate(self.tracker.slots):
            if slot['last_seen'] != self.tracker.frame:
                continue
            current = self.speed_bin(slot['velocity'])
            if (learn and index < len(before) and
                    before[index] == self.tracker.frame-1 and
                    index in self.previous_bins):
                self.transitions[self.previous_bins[index], current] += 1
            self.previous_bins[index] = current
        field = torch.zeros((self.tracker.height, self.tracker.width))
        for slot in self.tracker.slots:
            age = self.tracker.frame-slot['last_seen']
            if age > 64:
                continue
            velocity = slot['velocity']
            magnitude = velocity.norm()
            direction = velocity/magnitude.clamp(min=1e-8)
            center = slot['position'].clone()
            speed = self.speed_bin(velocity)
            for _ in range(age):
                counts = self.transitions[speed]
                if bool(counts.sum()):
                    speed = int(counts.argmax())
                center += direction*(speed/2)
            radius = 1.5+.15*age
            field = torch.maximum(field, torch.exp(-(
                (self.tracker.y-center[0]).square()+
                (self.tracker.x-center[1]).square())/(2*radius**2)))
        return field
