"""Opt-in bounded association beams over weighted events or visual surfaces."""

import math

import torch
import torch.nn.functional as F

from local_context_memory import LocalContextMemory
from persistent_entity_files import surface_components
from visual_history_controls import GenericMultiTracker


class EvidenceTracker(GenericMultiTracker):
    def __init__(self, *, representation='events', height=32, width=64):
        self.representation = representation
        super().__init__(height=height, width=width)

    def motion_scale(self, history, age):
        return 1.5+.15*age

    def proposals(self, sensory):
        evidence = sensory[:2].sum(0)
        if self.representation == 'surface':
            contrast = sensory[6]+sensory[7]
            groups = surface_components(contrast > .5, contrast, evidence > .2)
            return [(torch.tensor(group['center']), float(len(group['pixels'])))
                    for group in groups
                    if group['changed'] and len(group['pixels']) <= 64]
        # Smooth amplitudes; isolated weak events do not bridge components.
        density = F.avg_pool2d(evidence[None, None], 5, 1, 2)[0, 0]*25
        peaks = (density == F.max_pool2d(density[None, None], 5, 1, 2)[0, 0])
        candidates = (peaks & (density >= .5) & (density < 20)).nonzero()
        ordered = sorted(candidates.tolist(),
                         key=lambda p: -float(density[p[0], p[1]]))
        proposals = []
        for y, x in ordered:
            if any(float((position-torch.tensor([y, x])).norm()) < 5
                   for position, _ in proposals):
                continue
            patch = evidence[max(0, y-2):y+3, max(0, x-2):x+3]
            mass = float(patch.sum())
            yy = self.y[max(0, y-2):y+3, max(0, x-2):x+3]
            xx = self.x[max(0, y-2):y+3, max(0, x-2):x+3]
            position = torch.stack(((patch*yy).sum(), (patch*xx).sum()))/mass
            proposals.append((position, mass))
        return proposals

    @torch.no_grad()
    def step(self, sensory, *, observation_available=True):
        self.frame += 1
        proposals = self.proposals(sensory) if observation_available else []
        claimed = set()
        # Confirmed histories compete before tentative births. Preserve slot IDs.
        order = sorted(range(len(self.slots)),
                       key=lambda i: not self.slots[i]['confirmed'])
        for index in order:
            slot = self.slots[index]
            branches = []
            for history in slot['hypotheses']:
                age = self.frame-history['last_seen']
                if age > (64 if slot['confirmed'] else 2):
                    continue
                branches.append({**history, 'score': history['score']-2.,
                                 'proposal': None})
                expected = history['position']+age*history['velocity']
                for p, (position, mass) in enumerate(proposals):
                    if p in claimed or mass < .5*history['mass']:
                        continue
                    distance = float((position-expected).norm())
                    if distance > 3+age:
                        continue
                    scale = self.motion_scale(history, age)
                    cost = (distance/scale)**2/2 + abs(math.log(
                        mass/history['mass']))
                    branches.append(dict(position=position.clone(),
                        velocity=(position-history['position'])/age,
                        last_seen=self.frame, mass=mass,
                        hits=history['hits']+1,
                        score=history['score']-cost, proposal=p))
            if not branches:
                slot['hypotheses'] = []
                continue
            branches.sort(key=lambda branch: -branch['score'])
            beam = branches[:4]
            normalizer = float(torch.logsumexp(torch.tensor(
                [branch['score'] for branch in beam]), 0))
            for branch in beam:
                branch['score'] -= normalizer
            best = beam[0]
            slot.update({key: best[key] for key in
                         ('position', 'velocity', 'last_seen', 'mass')})
            slot['confirmed'] = slot['confirmed'] or best['hits'] >= 3
            slot['hypotheses'] = beam
            if best['proposal'] is not None:
                claimed.add(best['proposal'])
        for p, (position, mass) in enumerate(proposals):
            if p not in claimed:
                history = dict(position=position, velocity=torch.zeros(2),
                               last_seen=self.frame, mass=mass, hits=1,
                               score=0., proposal=p)
                self.slots.append({**history, 'confirmed': False,
                                   'hypotheses': [history]})


class EvidenceContextMemory(LocalContextMemory):
    def __init__(self, *, representation='events'):
        super().__init__()
        self.tracker = EvidenceTracker(representation=representation)
        self.reset_state()

    @torch.no_grad()
    def step(self, sensory, *, learn=False):
        # Keep the previously tested observed-only context credit unchanged.
        super().step(sensory, learn=learn)
        field = torch.zeros((self.tracker.height, self.tracker.width))
        for index, slot in enumerate(self.tracker.slots):
            if not slot['confirmed']:
                continue
            belief = torch.zeros_like(field)
            for history in slot['hypotheses']:
                if float(history['velocity'].norm()) <= .25:
                    continue
                age = self.tracker.frame-history['last_seen']
                center = history['position']+age*history['velocity']
                if index in self.context:
                    center[0] += (age*abs(float(history['velocity'][1]))*
                                  self.effects[self.context[index]])
                radius = 1.5+.15*age
                belief += math.exp(history['score'])*torch.exp(-(
                    (self.tracker.y-center[0]).square()+
                    (self.tracker.x-center[1]).square())/(2*radius**2))
            field = torch.maximum(field, belief*self.confidence.get(index, 1.))
        return field
