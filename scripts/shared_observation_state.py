"""Opt-in separate entity histories under an ambiguous shared observation."""

import torch

from associative_patch_state import AssociativePatchState
from evidence_state import EvidenceTracker
from persistent_entity_files import surface_components


class SharedObservationTracker(EvidenceTracker):
    def __init__(self, *, height=32, width=64):
        super().__init__(representation='surface', height=height, width=width)

    def proposals(self, sensory):
        contrast = sensory[6]+sensory[7]
        groups = surface_components(contrast > .5, contrast, sensory[:2].sum(0) > .2)
        expected = []
        for slot in self.slots:
            age = self.frame-slot['last_seen']
            if (slot['confirmed'] and slot['hypotheses'] and age <= 64 and
                    float(slot['velocity'].norm()) > .25):
                position = slot['position']+age*slot['velocity']
                expected.append(tuple(round(float(v)) for v in position))
        proposals = []
        for group in groups:
            if not group['changed'] or len(group['pixels']) > 64:
                continue
            pixels = set(group['pixels'])
            if sum(position in pixels for position in expected) >= 2:
                continue
            proposals.append((torch.tensor(group['center']), float(len(pixels))))
        return proposals


class SharedObservationState(AssociativePatchState):
    def __init__(self, *, height=32, width=64, memory=None):
        super().__init__(height=height, width=width, memory=memory)
        self.tracker = SharedObservationTracker(height=height, width=width)
        self.reset_state()
