"""Benchmark-only generic visual motifs associated with observed dynamics."""

import torch

from fly_connectome.sensor import EventCamera
from generic_motion_probe import events_map


@torch.no_grad()
def appearance_case(case):
    result = dict(case)
    cue = torch.zeros_like(case['cue_mask'])
    y = case['y']-8
    x = 20 if case['direction'] == 1 else 43
    if case['cue_sign'] < 0:
        cue[y, x:x+4] = True
    else:
        cue[y:y+4, x] = True
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(case['background'])
    visible, intensity, events = [], [], []
    for original in case['visible']:
        image = original.clone()
        image[case['cue_mask']] = case['background']
        image[cue] = not case['background']
        visible.append(image)
        intensity.append(image.float()-float(case['background']))
        events.append(events_map(camera.observe(image[None])))
    result.update(cue_mask=cue, visible=visible, intensity=intensity, events=events)
    return result
