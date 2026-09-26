"""Generic event-camera histories with controlled hidden visual state."""

import torch

from fly_connectome.sensor import EventCamera
from generic_motion_probe import events_map


SHAPES = dict(dot=((0, 0),),
              square=tuple((dy, dx) for dy in (-1, 0, 1)
                           for dx in (-1, 0, 1)),
              plus=((0, 0), (-2, 0), (-1, 0), (1, 0), (2, 0),
                    (0, -2), (0, -1), (0, 1), (0, 2)))


def periodic_steps(frames, *, cycle, phase=0):
    """Cumulative displacement from a repeated generic motion cadence."""
    position = 0
    result = []
    for frame in range(frames):
        result.append(position)
        position += cycle[(frame+phase) % len(cycle)]
    return result


def _object_mask(y, x, shape):
    mask = torch.zeros((32, 64), dtype=torch.bool)
    for dy, dx in SHAPES[shape]:
        py, px = y+dy, x+dx
        if 0 <= py < 32 and 0 <= px < 64:
            mask[py, px] = True
    return mask


@torch.no_grad()
def render_case(*, direction, speed, y, shape, disappear=False,
                background=False, frames=64, steps=None,
                decision_index=None, cue_sign=None, turn_sign=None):
    """One moving pattern; only visible pixels reach either sensor."""
    if direction not in (-1, 1) or speed not in (1, 2, 3):
        raise ValueError('direction and speed must be supported')
    if shape not in SHAPES or not 4 <= y <= 27:
        raise ValueError('shape or location outside the visual field')
    if cue_sign is not None and (cue_sign not in (-1, 1)
                                 or not 10 <= y <= 21):
        raise ValueError('context cue is outside the visual field')
    if turn_sign is not None and (cue_sign is None or turn_sign not in (-1, 1)):
        raise ValueError('turn requires a visible context scene')
    occluder = torch.zeros((32, 64), dtype=torch.bool)
    occluder[:, 24:40] = True
    cue_mask = torch.zeros_like(occluder)
    if cue_sign is not None:
        cue_x = 20 if direction == 1 else 43
        cue_mask[y+8*cue_sign, cue_x] = True
        cue_mask[y+8*cue_sign, cue_x+1] = True
    start = 8 if direction == 1 else 55
    if steps is not None and len(steps) != frames:
        raise ValueError('steps must cover every frame')
    paths = []
    for frame in range(frames):
        x = start+direction*(speed*frame if steps is None
                             else steps[frame])
        progress = ((x-24)/15 if direction == 1 else (39-x)/15)
        physical_turn = cue_sign if turn_sign is None else turn_sign
        shift = (0 if physical_turn is None else
                 round(8*physical_turn*max(0., min(1., progress))))
        paths.append(_object_mask(y+shift, x, shape))
    invisible = [frame for frame, path in enumerate(paths)
                 if bool(path.any()) and not bool((path & ~occluder).any())]
    if not invisible:
        raise ValueError('trajectory never enters occluder')
    # Score while the two approach histories imply distinct hidden sites.
    decision = invisible[min(3, len(invisible)//3)
                         if decision_index is None else decision_index]
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
        image[0, occluder | cue_mask | visible_object] = not background
        visible.append(image[0].clone())
        visible_objects.append(visible_object)
        hidden.append(concealed)
        intensity.append(image[0].float()-float(background))
        events.append(events_map(camera.observe(image)))
    return dict(direction=direction, speed=speed, y=y, shape=shape,
                background=background,
                disappear=disappear, decision=decision,
                reveal=None if disappear else reveal, visible=visible,
                visible_objects=visible_objects, occluder=occluder,
                cue_mask=cue_mask, cue_sign=cue_sign,
                turn_sign=physical_turn,
                hidden=hidden, intensity=intensity, events=events)


def render_context_case(*, direction, speed, y, shape, cue_sign,
                        disappear=False, background=False, frames=64,
                        turn_sign=None):
    return render_case(direction=direction, speed=speed, y=y, shape=shape,
                       cue_sign=cue_sign, disappear=disappear,
                       background=background, frames=frames,
                       turn_sign=turn_sign,
                       decision_index=-2)


@torch.no_grad()
def render_two_mover_case(*, speed, y, shapes, disappear=(False, False),
                          background=False, frames=64):
    """Opposite observed approaches; identity truth is evaluation-only."""
    if speed not in (1, 2, 3) or any(shape not in SHAPES for shape in shapes):
        raise ValueError('unsupported motion or shape')
    if len(shapes) != 2 or len(disappear) != 2 or not 4 <= y <= 27:
        raise ValueError('expected two movers inside the visual field')
    occluder = torch.zeros((32, 64), dtype=torch.bool)
    occluder[:, 24:40] = True
    paths = [[_object_mask(y, start+direction*speed*frame, shape)
              for frame in range(frames)]
             for start, direction, shape in zip((8, 55), (1, -1), shapes)]
    invisible = [[frame for frame, mask in enumerate(path)
                  if bool(mask.any()) and not bool((mask & ~occluder).any())]
                 for path in paths]
    if any(not interval for interval in invisible):
        raise ValueError('both trajectories must enter the occluder')
    overlap = sorted(set(invisible[0]) & set(invisible[1]))
    if not overlap:
        raise ValueError('trajectories are not hidden together')
    decision = overlap[min(3, len(overlap)//3)]
    first_hidden = [interval[0] for interval in invisible]
    reveal = [next((frame for frame in range(interval[-1]+1, frames)
                    if bool((path[frame] & ~occluder).any())), None)
              if not gone else None
              for path, interval, gone in zip(paths, invisible, disappear)]
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(background)
    visible, visible_objects, hidden, intensity, events = [], [], [], [], []
    hidden_by_entity = [[], []]
    for frame in range(frames):
        shown = torch.zeros((32, 64), dtype=torch.bool)
        concealed = torch.zeros_like(shown)
        for index, path in enumerate(paths):
            mask = (torch.zeros_like(path[frame])
                    if disappear[index] and frame >= first_hidden[index]
                    else path[frame])
            part = mask & occluder
            hidden_by_entity[index].append(part)
            concealed |= part
            shown |= mask & ~occluder
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        image[0, occluder | shown] = not background
        visible.append(image[0].clone())
        visible_objects.append(shown)
        hidden.append(concealed)
        intensity.append(image[0].float()-float(background))
        events.append(events_map(camera.observe(image)))
    return dict(speed=speed, y=y, shapes=shapes, disappear=disappear,
                decision=decision, reveal_by_entity=reveal,
                visible=visible, visible_objects=visible_objects,
                occluder=occluder, hidden=hidden,
                hidden_by_entity=hidden_by_entity,
                intensity=intensity, events=events)
