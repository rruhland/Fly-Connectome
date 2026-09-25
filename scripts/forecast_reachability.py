"""Frozen spatial support audit for the opt-in 5×5 local event readout."""

import torch.nn.functional as F


def reached_targets(source_state, future_event, *, radius=2):
    active = (source_state.max(0).values >= .5).float()[None, None]
    supported = F.max_pool2d(active, 2*radius+1, stride=1,
                             padding=radius)[0, 0].bool()
    targets = future_event >= .5
    return int((targets & supported[None]).sum()), int(targets.sum())
