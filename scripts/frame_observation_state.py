"""Opt-in state with a declared full-frame observation model."""

import torch

from associative_patch_state import AssociativePatchState
from evidence_state import EvidenceTracker
from persistent_entity_files import surface_components


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


class FrameObservationState(AssociativePatchState):
    def __init__(self, *, height=32, width=64, memory=None):
        super().__init__(height=height, width=width, memory=memory, include_static=True)
        self.tracker = FrameObservationTracker(height=height, width=width)
        self.reset_state()

    @torch.no_grad()
    def step(self, sensory, *, learn=False, observation_available=True):
        self.tracker.measurement_available = observation_available
        return super().step(sensory, learn=learn, observation_available=observation_available)
