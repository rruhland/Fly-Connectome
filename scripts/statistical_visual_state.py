"""Explicit sensor-noise and motion uncertainty for generic visual association."""

import math

import torch

from frame_observation_state import FrameObservationTracker, FrameObservationState
from streaming_visual_state import LearnedVisualCandidate


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


class StatisticalVisualCandidate(LearnedVisualCandidate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state.tracker = StatisticalObservationTracker(height=self.observer.height,
                                                           width=self.observer.width)
        self.reset_state()

    def frame_contrast(self, visible):
        contrast = visible-visible.median()
        sigma = contrast.abs().median()/.67448975
        bound = math.sqrt(2*math.log(2*visible.numel()/.01))
        scale = torch.maximum(torch.quantile(contrast.abs(), .999), 2*bound*sigma)
        if scale == 0:
            scale = contrast.abs().max()
        return (contrast/scale.clamp(min=torch.finfo(contrast.dtype).eps)).clamp(-1., 1.)


class StatisticalFrameState(FrameObservationState):
    """The same association stage for sensory-array regression audits."""

    def __init__(self, *, height=32, width=64, memory=None):
        super().__init__(height=height, width=width, memory=memory)
        self.tracker = StatisticalObservationTracker(height=height, width=width)
        self.reset_state()
