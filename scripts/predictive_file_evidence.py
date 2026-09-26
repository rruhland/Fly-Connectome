"""Opt-in entity files confirmed by local visual predictive evidence."""

import torch
import torch.nn.functional as F

from persistent_entity_files import PersistentEntityFiles


class PredictiveEvidenceFiles(PersistentEntityFiles):
    """Keep tentative regions until continuation or independent support."""

    def __init__(self, *, confirm_birth=True, condition_forecast=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.confirm_birth = confirm_birth
        self.condition_forecast = condition_forecast
        self.support = None

    def association_slots(self):
        return self.slots if self.confirm_birth else self.live_slots

    def new_slot(self, candidate):
        super().new_slot(candidate)
        if not self.confirm_birth:
            return
        confidence = .25
        if self.support is not None:
            flat = self.support.flatten()
            positive = int((flat > 0).sum())
            winners = set(flat.topk(min(32, positive)).indices.tolist())
            channel = 1 if candidate['sign'] > 0 else 0
            if any(channel*self.height*self.width+y*self.width+x in winners
                   for y, x in candidate['pixels']):
                confidence = 1.
        self.slots[-1]['confidence'] = confidence

    @torch.no_grad()
    def step(self, event, *, support=None):
        self.support = support
        super().step(event)
        if self.confirm_birth:
            self.slots = [slot for slot in self.slots
                          if slot['confidence'] >= .5 or
                          self.frame-slot['last_seen'] <= 20]
        return self.live_slots

    @torch.no_grad()
    def forecast(self, fast_forecast=None):
        rigid = super().forecast()
        if not self.condition_forecast or fast_forecast is None:
            return rigid
        support = torch.zeros((self.height, self.width))
        for slot in self.live_slots:
            for y, x in slot['pixels']:
                support[y, x] = 1.
        near = F.max_pool2d(support[None, None], 7, stride=1,
                            padding=3)[0, 0]
        return torch.maximum(rigid, fast_forecast*near[None])
