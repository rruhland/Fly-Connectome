"""Statistical association beams; unchanged from the approved visual candidate."""
import math
import torch
import torch.nn.functional as F
from .observation import surface_components

class GenericMultiTracker:

    def __init__(self, *, height=32, width=64, minimum_strength=.05):
        self.height, self.width = height, width
        self.minimum_strength = minimum_strength
        self.y, self.x = torch.meshgrid(torch.arange(height),
                                        torch.arange(width), indexing='ij')
        self.reset_state()

    def reset_state(self):
        self.frame = -1
        self.slots = []


class EvidenceTracker(GenericMultiTracker):
    def __init__(self, *, representation='events', height=32, width=64):
        self.representation = representation
        super().__init__(height=height, width=width)

    def motion_scale(self, history, age):
        return 1.5+.15*age

    def accept_match(self, history, position, sensory):
        return True

    def association_radius(self, history, age):
        return 3+age

    def motion_metadata(self, history, velocity):
        return {}

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
                    if not self.accept_match(history, position, sensory):
                        continue
                    distance = float((position-expected).norm())
                    if distance > self.association_radius(history, age):
                        continue
                    scale = self.motion_scale(history, age)
                    cost = (distance/scale)**2/2 + abs(math.log(
                        mass/history['mass']))
                    velocity = (position-history['position'])/age
                    branches.append(dict(position=position.clone(),
                        velocity=velocity,
                        last_seen=self.frame, mass=mass,
                        hits=history['hits']+1,
                        score=history['score']-cost, proposal=p,
                        **self.motion_metadata(history, velocity)))
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


class FrameObservationTracker(EvidenceTracker):
    def __init__(self, *, height=32, width=64):
        super().__init__(representation='surface', height=height, width=width)
        self.measurement_available = True

    def motion_scale(self, history, age):
        return 3+age if history['hits'] == 1 else super().motion_scale(history, age)

    def proposals(self, sensory):
        if not self.measurement_available:
            return []
        contrast = sensory[6]+sensory[7]
        mask = contrast > .5
        groups = surface_components(mask, contrast, mask)
        expected = []
        for slot in self.slots:
            age = self.frame-slot['last_seen']
            if slot['confirmed'] and slot['hypotheses'] and age <= 64:
                position = slot['position']+age*slot['velocity']
                expected.append(tuple(round(float(v)) for v in position))
        proposals = []
        for group in groups:
            if len(group['pixels']) > 64:
                continue
            pixels = set(group['pixels'])
            if sum(position in pixels for position in expected) >= 2:
                continue
            proposals.append((torch.tensor(group['center']), float(len(pixels))))
        return proposals


class StatisticalObservationTracker(FrameObservationTracker):
    def association_radius(self, history, age):
        return float('inf')

    def motion_metadata(self, history, velocity):
        variance = history.get('motion_variance', 0.)
        if history['hits'] >= 2:
            variance = .75*variance+.25*float((velocity-history['velocity']).square().sum())
        return dict(motion_variance=variance)

    def motion_scale(self, history, age):
        scale = super().motion_scale(history, age)
        return math.sqrt(scale**2+age**2*history.get('motion_variance', 0.))
