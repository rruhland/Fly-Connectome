"""Opt-in visible-image anchors with explicitly declared sensor cadence."""

import torch

from generic_motion_probe import local_motion_map


@torch.no_grad()
def anchored_sequence(observer, case, *, period, hybrid=False):
    """Only current visible images anchor state; metadata is never read."""
    observer.reset_state()
    trace = torch.zeros_like(case['events'][0])
    states = []
    for frame, event in enumerate(case['events']):
        filtered, _ = observer.step(event)
        if frame % period == 0:
            visible = case['visible'][frame].float()
            observer.contrast.copy_(visible-visible.median())
        motion = local_motion_map(trace, filtered).clamp(-1., 1.)
        trace.mul_(.875).add_(filtered)
        contrast = observer.contrast
        states.append(torch.cat((filtered, observer.fast.clone(),
                                 observer.slow.clone(),
                                 contrast.clamp(min=0.)[None],
                                 (-contrast).clamp(min=0.)[None],
                                 motion, trace.clone())))
    return states
