"""Generic event-camera histories with controlled hidden visual state."""

import torch

from fly_connectome.sensor import EventCamera
from generic_motion_probe import events_map


SHAPES = dict(dot=((0, 0),),
              square=tuple((dy, dx) for dy in (-1, 0, 1)
                           for dx in (-1, 0, 1)),
              plus=((0, 0), (-2, 0), (-1, 0), (1, 0), (2, 0),
                    (0, -2), (0, -1), (0, 1), (0, 2)))


def _object_mask(y, x, shape):
    mask = torch.zeros((32, 64), dtype=torch.bool)
    for dy, dx in SHAPES[shape]:
        py, px = y+dy, x+dx
        if 0 <= py < 32 and 0 <= px < 64:
            mask[py, px] = True
    return mask


@torch.no_grad()
def render_case(*, direction, speed, y, shape, disappear=False,
                background=False, frames=64):
    """One moving pattern; only visible pixels reach either sensor."""
    if direction not in (-1, 1) or speed not in (1, 2, 3):
        raise ValueError('direction and speed must be supported')
    if shape not in SHAPES or not 4 <= y <= 27:
        raise ValueError('shape or location outside the visual field')
    occluder = torch.zeros((32, 64), dtype=torch.bool)
    occluder[:, 24:40] = True
    start = 8 if direction == 1 else 55
    paths = [_object_mask(y, start+direction*speed*frame, shape)
             for frame in range(frames)]
    invisible = [frame for frame, path in enumerate(paths)
                 if bool(path.any()) and not bool((path & ~occluder).any())]
    if not invisible:
        raise ValueError('trajectory never enters occluder')
    # Score while the two approach histories imply distinct hidden sites.
    decision = invisible[min(3, len(invisible)//3)]
    first_hidden = invisible[0]
    reveal = next((frame for frame in range(invisible[-1]+1, frames)
                   if bool((paths[frame] & ~occluder).any())), None)
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(background)
    visible = []
    visible_objects = []
    hidden = []
    intensity = []
    events = []
    for frame, path in enumerate(paths):
        if disappear and frame >= first_hidden:
            path = torch.zeros_like(path)
        concealed = path & occluder
        visible_object = path & ~occluder
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        image[0, occluder | visible_object] = not background
        visible.append(image[0].clone())
        visible_objects.append(visible_object)
        hidden.append(concealed)
        intensity.append(image[0].float()-float(background))
        events.append(events_map(camera.observe(image)))
    return dict(direction=direction, speed=speed, y=y, shape=shape,
                disappear=disappear, decision=decision,
                reveal=None if disappear else reveal, visible=visible,
                visible_objects=visible_objects, occluder=occluder,
                hidden=hidden, intensity=intensity, events=events)
