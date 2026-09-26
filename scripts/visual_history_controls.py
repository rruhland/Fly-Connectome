"""Complete sensory, fixed temporal-memory, and generic motion controls."""

import torch

from generic_motion_probe import local_motion_map


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

    def __init__(self, *, height=32, width=64):
        self.height, self.width = height, width
        self.y, self.x = torch.meshgrid(torch.arange(height),
                                        torch.arange(width), indexing='ij')
        self.reset_state()

    def reset_state(self):
        self.frame = -1
        self.last_position = None
        self.last_seen = None
        self.last_mass = None
        self.velocity = torch.zeros(2)

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
                    self.velocity = (position-self.last_position)/elapsed
            self.last_position = position
            self.last_seen = self.frame
            self.last_mass = mass
        if self.last_position is None:
            return torch.zeros((self.height, self.width))
        age = self.frame-self.last_seen
        center = self.last_position+age*self.velocity
        radius = 1.5+.15*age
        return torch.exp(-((self.y-center[0]).square()+
                           (self.x-center[1]).square())/(2*radius**2))
