"""Opt-in local credit for two learned temporal transition pathways."""

import torch
import torch.nn.functional as F


class LocalTemporalMixture:
    """Select a future-event pathway from nearby state and local error."""

    def __init__(self, *, height=32, width=64):
        self.height = height
        self.width = width
        self.weights = torch.zeros((2, 16))
        self.bias = torch.zeros((2, 1, 1))

    @torch.no_grad()
    def features(self, slow_state, fast_state):
        joined = torch.cat((slow_state, fast_state))
        return F.avg_pool2d(joined[None], 5, stride=1,
                            padding=2)[0]*25

    @torch.no_grad()
    def forecast(self, features, slow, fast):
        score = torch.einsum('oc,chw->ohw',
                             self.weights, features)+self.bias
        gate = torch.sigmoid(score)
        return gate*slow+(1-gate)*fast

    @torch.no_grad()
    def credit(self, features, slow, fast, target):
        quality = (target-fast).square()-(target-slow).square()
        active = (target+slow+fast) > 0
        source = features.reshape(16, -1)
        eligibility = (quality*active).reshape(2, -1) @ source.T
        exposure = active.float().reshape(2, -1) @ source.T
        self.weights.add_(.05*eligibility/exposure.clamp(min=1.))
        total = active.sum((1, 2)).clamp(min=1.)
        self.bias.add_(.05*(quality*active).sum((1, 2))[:, None, None]
                       / total[:, None, None])
        self.weights.clamp_(-2., 2.)
        self.bias.clamp_(-2., 2.)
