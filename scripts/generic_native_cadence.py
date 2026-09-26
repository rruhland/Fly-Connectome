"""Unlabeled moving-pattern camera streams at fractional-pixel cadence."""

import math
import random

import torch

from fly_connectome.sensor import EventCamera
from generic_motion_probe import events_map


TRAIN_MOTIFS = (((0, 0),), ((0, 0), (0, 1)),
                ((0, 0), (1, 0)), ((0, 0), (0, 1), (1, 0), (1, 1)))
HELDOUT_MOTIFS = (((0, 0),), ((0, 0), (1, 1)),
                  ((0, 0), (1, 0), (1, 1)))


def reflect(position, low, high):
    width = high-low
    phase = (position-low) % (2*width)
    return low+phase if phase <= width else high-(phase-width)


@torch.no_grad()
def scene_events(seed, *, frames=80, heldout=False,
                 return_contrast=False, motion_stride=1):
    rng = random.Random(seed)
    background = bool(seed % 2)
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(background)
    shapes = HELDOUT_MOTIFS if heldout else TRAIN_MOTIFS
    objects = []
    for _ in range(1+seed % 3):
        objects.append((rng.uniform(7, 25), rng.uniform(12, 52),
                        rng.choice((-.24, -.16, -.08, 0., .08, .16, .24)),
                        rng.choice((-.36, -.28, -.20, .20, .28, .36)),
                        rng.choice(shapes)))
    static = (rng.randrange(6, 26), rng.randrange(8, 56)) if seed % 3 == 0 else None
    result = []
    intensity = []
    for t in range(frames):
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        if static is not None:
            y, x = static
            image[0, y, x:x+3] = not background
        for y0, x0, dy, dx, shape in objects:
            y = math.floor(reflect(y0+dy*t*motion_stride, 2, 29)+.5)
            x = math.floor(reflect(x0+dx*t*motion_stride, 2, 61)+.5)
            for oy, ox in shape:
                image[0, y+oy, x+ox] = not background
        result.append(events_map(camera.observe(image)))
        if return_contrast:
            intensity.append(image[0].float()-float(background))
    return (result, intensity) if return_contrast else result
