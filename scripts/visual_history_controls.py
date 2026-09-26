"""Complete sensory, fixed temporal-memory, and generic motion controls."""

import torch
import torch.nn.functional as F

from generic_motion_probe import local_motion_map
from persistent_entity_files import surface_components


@torch.no_grad()
def sensory_sequence(observer, case, *, hybrid):
    """Expose the observer's entire causal state, not just its event output."""
    observer.reset_state()
    trace = torch.zeros_like(case['events'][0])
    states = []
    for frame, event in enumerate(case['events']):
        absolute = (case['intensity'][frame]
                    if hybrid and frame % 8 == 0 else None)
        filtered, _ = observer.step(event, absolute=absolute)
        motion = local_motion_map(trace, filtered).clamp(-1., 1.)
        trace.mul_(.875).add_(filtered)
        contrast = observer.contrast
        states.append(torch.cat((filtered, observer.fast.clone(),
                                 observer.slow.clone(),
                                 contrast.clamp(min=0.)[None],
                                 (-contrast).clamp(min=0.)[None],
                                 motion, trace.clone())))
    return states


class FixedLeakyMemory:
    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.reset_state()

    def reset_state(self):
        self.traces = [torch.zeros((12, self.height, self.width))
                       for _ in range(3)]

    @torch.no_grad()
    def step(self, sensory):
        for trace, decay in zip(self.traces, (.7, .95, .99)):
            trace.mul_(decay).add_(sensory, alpha=1-decay)
        return torch.cat((sensory, *self.traces)).clone()


class GenericTracker:
    """No-learning constant-velocity spatial control from observed events."""

    def __init__(self, *, height=32, width=64, acceleration=False):
        self.height, self.width = height, width
        self.use_acceleration = acceleration
        self.y, self.x = torch.meshgrid(torch.arange(height),
                                        torch.arange(width), indexing='ij')
        self.reset_state()

    def reset_state(self):
        self.frame = -1
        self.last_position = None
        self.last_seen = None
        self.last_mass = None
        self.velocity = torch.zeros(2)
        self.acceleration = torch.zeros(2)

    @torch.no_grad()
    def step(self, sensory):
        self.frame += 1
        evidence = sensory[:2].sum(0).clamp(min=0.)
        mass = float(evidence.sum())
        partial = self.last_mass is not None and mass < .75*self.last_mass
        if .1 < mass < 64 and not partial:
            position = torch.stack(((evidence*self.y).sum(),
                                    (evidence*self.x).sum()))/mass
            if self.last_position is not None:
                elapsed = self.frame-self.last_seen
                if elapsed <= 3:
                    new_velocity = (position-self.last_position)/elapsed
                    self.acceleration = new_velocity-self.velocity
                    self.velocity = new_velocity
            self.last_position = position
            self.last_seen = self.frame
            self.last_mass = mass
        if self.last_position is None:
            return torch.zeros((self.height, self.width))
        age = self.frame-self.last_seen
        center = self.last_position+age*self.velocity
        if self.use_acceleration:
            center += age*(age+1)/2*self.acceleration
        radius = 1.5+.15*age
        return torch.exp(-((self.y-center[0]).square()+
                           (self.x-center[1]).square())/(2*radius**2))


class GenericMultiTracker:
    """Fixed connected-event association and constant-velocity control."""

    def __init__(self, *, height=32, width=64, minimum_strength=.05):
        self.height, self.width = height, width
        self.minimum_strength = minimum_strength
        self.y, self.x = torch.meshgrid(torch.arange(height),
                                        torch.arange(width), indexing='ij')
        self.reset_state()

    def reset_state(self):
        self.frame = -1
        self.slots = []

    @torch.no_grad()
    def step(self, sensory):
        self.frame += 1
        active = sensory[:2].sum(0) > self.minimum_strength
        expanded = F.max_pool2d(active.float()[None, None], 5,
                                stride=1, padding=2)[0, 0] > 0
        proposals = [(torch.tensor(group['center']), len(group['pixels']))
                     for group in surface_components(
                         expanded, torch.zeros_like(expanded), expanded)
                     if len(group['pixels']) <= 160]
        options = []
        for slot_index, slot in enumerate(self.slots):
            age = self.frame-slot['last_seen']
            expected = slot['position']+age*slot['velocity']
            for proposal_index, (position, mass) in enumerate(proposals):
                distance = float((expected-position).norm())
                if distance <= 3+2*age:
                    options.append((distance, slot_index, proposal_index))
        assigned_slots, assigned_proposals = set(), set()
        for _, slot_index, proposal_index in sorted(options):
            if slot_index in assigned_slots or proposal_index in assigned_proposals:
                continue
            slot = self.slots[slot_index]
            position, mass = proposals[proposal_index]
            elapsed = self.frame-slot['last_seen']
            if mass >= .9*slot['mass']:
                slot['velocity'] = (position-slot['position'])/elapsed
                slot['position'] = position
                slot['last_seen'] = self.frame
                slot['mass'] = mass
            assigned_slots.add(slot_index)
            assigned_proposals.add(proposal_index)
        for index, (position, mass) in enumerate(proposals):
            if index not in assigned_proposals:
                self.slots.append(dict(position=position, velocity=torch.zeros(2),
                                       last_seen=self.frame, mass=mass))
        field = torch.zeros((self.height, self.width))
        for slot in self.slots:
            age = self.frame-slot['last_seen']
            if age > 64:
                continue
            center = slot['position']+age*slot['velocity']
            radius = 1.5+.15*age
            field = torch.maximum(field, torch.exp(-(
                (self.y-center[0]).square()+(self.x-center[1]).square())
                /(2*radius**2)))
        return field
