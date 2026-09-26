"""Opt-in visual files with an independent absolute-contrast observation."""

import torch

from persistent_entity_files import PersistentEntityFiles


class AbsoluteRefreshFiles(PersistentEntityFiles):
    """Correct local event memory when a new intensity frame is available."""

    @torch.no_grad()
    def refresh(self, contrast):
        self.surface.contrast.copy_(contrast)
        self.surface.confidence.zero_()
        self.surface.confidence[contrast.abs() > .5] = 1.
        supported = []
        for slot in self.slots:
            y, x = (round(value) for value in slot['center'])
            patch = contrast[max(0, y-2):min(self.height, y+3),
                             max(0, x-2):min(self.width, x+3)]
            if bool((slot['sign']*patch > .5).any()):
                supported.append(slot)
        self.slots = supported
        return self.live_slots

    @torch.no_grad()
    def step(self, event, *, absolute=None):
        super().step(event)
        if absolute is not None:
            self.refresh(absolute)
        return self.live_slots
