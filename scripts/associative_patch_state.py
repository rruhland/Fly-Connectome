"""Opt-in generic local visual associations credited by observed continuity."""

import torch
import torch.nn.functional as F

from evidence_state import EvidenceTracker


class PatchAssociation:
    def __init__(self, *, dimensions=51):
        self.keys = torch.empty(0, dimensions)
        self.values = torch.empty(0, 2)
        self.metric = torch.zeros(dimensions)

    @torch.no_grad()
    def observe(self, key, value):
        self.keys = torch.cat((self.keys, key[None]))[-256:]
        self.values = torch.cat((self.values, value[None]))[-256:]
        pre = self.keys-self.keys.mean(0)
        post = self.values-self.values.mean(0)
        covariance = pre.T @ post/len(pre)
        variance = pre.square().mean(0)
        self.metric = covariance.square().sum(1)/(variance+1e-4).square()
        self.metric /= self.metric.sum().clamp(min=1e-8)

    @torch.no_grad()
    def predict(self, key):
        if not len(self.keys):
            return torch.zeros(2)
        distance = ((self.keys-key).square()*self.metric).sum(1)
        values, indices = distance.topk(min(4, len(distance)), largest=False)
        weights = torch.softmax(-values/.1, 0)
        return (self.values[indices]*weights[:, None]).sum(0)


class AssociativePatchState:
    def __init__(self, *, height=32, width=64, memory=None, include_static=False):
        self.tracker = EvidenceTracker(representation='surface', height=height,
                                       width=width)
        self.memory = memory if memory is not None else PatchAssociation()
        self.include_static = include_static
        self.reset_state()

    def reset_state(self):
        self.tracker.reset_state()
        self.keys = {}
        self.confidence = {}
        self.entities = []

    def key(self, sensory, slot):
        image = sensory[6]+sensory[7]
        y, x = (round(float(v)) for v in slot['position'])
        padded = F.pad(image, (10, 10, 10, 10))
        patch = padded[y:y+21, x:x+21]
        coarse = F.avg_pool2d(patch[None, None], 3, 3).flatten()
        velocity = slot['velocity']
        return torch.cat((coarse, velocity/velocity.norm().clamp(min=.25)))

    @torch.no_grad()
    def step(self, sensory, *, learn=False, observation_available=True):
        before = [(s['position'].clone(), s['velocity'].clone(), s['last_seen'])
                  for s in self.tracker.slots]
        self.tracker.step(sensory, observation_available=observation_available)
        for index, slot in enumerate(self.tracker.slots):
            if slot['last_seen'] != self.tracker.frame:
                continue
            if learn and index < len(before) and index in self.keys:
                old_position, old_velocity, old_seen = before[index]
                elapsed = self.tracker.frame-old_seen
                speed = float(old_velocity.norm())
                if elapsed > 1 and speed > .25:
                    correction = ((slot['position']-old_position)/elapsed-
                                  old_velocity)/speed
                    self.memory.observe(self.keys[index], correction)
            if float(slot['velocity'].norm()) > .25:
                self.keys[index] = self.key(sensory, slot)
        field = torch.zeros(self.tracker.height, self.tracker.width)
        self.entities = []
        for index, slot in enumerate(self.tracker.slots):
            speed = float(slot['velocity'].norm())
            if not slot['confirmed'] or (speed <= .25 and not self.include_static):
                continue
            age = self.tracker.frame-slot['last_seen']
            if age > 64:
                continue
            correction = (self.memory.predict(self.keys[index])*speed
                          if index in self.keys else torch.zeros(2))
            center = slot['position']+age*(slot['velocity']+correction)
            if age == 0:
                self.confidence[index] = 1.
            elif observation_available:
                y, x = (round(float(v)) for v in center)
                supported = (0 <= y < self.tracker.height and
                             0 <= x < self.tracker.width and
                             float(sensory[6:8, y, x].sum()) >= .2)
                if not supported:
                    self.confidence[index] = .35*self.confidence.get(index, 1.)
            radius = 1.5+.15*age
            self.entities.append(dict(id=index, position=center.clone(),
                                      strength=self.confidence.get(index, 1.)))
            field = torch.maximum(field, self.confidence.get(index, 1.)*
                                  torch.exp(-((self.tracker.y-center[0]).square()+
                                    (self.tracker.x-center[1]).square())/(2*radius**2)))
        return field
