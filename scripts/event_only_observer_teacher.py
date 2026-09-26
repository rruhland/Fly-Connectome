"""Opt-in causal local event agreement as observer teaching signal."""

import torch
import torch.nn.functional as F

from local_observation_model import LocalObservationModel


@torch.no_grad()
def local_event_support(raw, past):
    """Same-sign neighboring or recent events support an arriving event."""
    temporal = F.max_pool2d(past[None], 5, stride=1, padding=2)[0]
    spatial = F.conv2d(raw[None], torch.ones((2, 1, 5, 5)),
                       padding=2, groups=2)[0]-raw
    return raw*((temporal > .2) | (spatial > .5)).float()


@torch.no_grad()
def fit_event_only_observer(episodes):
    model = LocalObservationModel()
    for episode in episodes:
        model.reset_state()
        past = torch.zeros_like(episode[0])
        for raw in episode:
            target = local_event_support(raw, past)
            _, features = model.step(raw)
            model.credit(features, raw, target)
            past.mul_(.85).add_(raw)
    model.reset_state()
    return model
