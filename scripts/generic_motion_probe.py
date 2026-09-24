"""Opt-in local motion-map transfer across shapes and independent objects."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional

from fly_connectome.sensor import EventCamera


OUT = Path('docs/experiments/2026-09-23-generic-motion-map-results.json')
DECAY = math.exp(-1/8)
SHAPES = {
    'square': [(y, x) for y in range(-1, 2) for x in range(-1, 2)],
    'plus': [(y, 0) for y in range(-2, 3)] + [(0, x) for x in (-2, -1, 1, 2)],
    'ell': [(y, -2) for y in range(-2, 3)] + [(2, x) for x in range(-1, 3)],
    'bar': [(y, x) for y in range(-3, 4) for x in (0, 1)],
}


def local_motion_map(trace, current):
    """Signed local x/y evidence at each current-event pixel."""
    x = torch.zeros_like(current[0])
    y = torch.zeros_like(current[0])
    for distance in (1, 2):
        x[:, distance:] += (trace[:, :, :-distance]
                            * current[:, :, distance:]).sum(0)
        x[:, :-distance] -= (trace[:, :, distance:]
                             * current[:, :, :-distance]).sum(0)
        y[distance:, :] += (trace[:, :-distance, :]
                            * current[:, distance:, :]).sum(0)
        y[:-distance, :] -= (trace[:, distance:, :]
                             * current[:, :-distance, :]).sum(0)
    return torch.stack((x, y))


def object_mask(shape, center):
    result = torch.zeros((32, 64), dtype=torch.bool)
    cy, cx = center
    for dy, dx in SHAPES[shape]:
        y, x = cy+dy, cx+dx
        if 0 <= y < 32 and 0 <= x < 64:
            result[y, x] = True
    return result


def events_map(events):
    result = torch.zeros((2, 32*64))
    result[events.on.long(), events.pixels] = 1
    return result.view(2, 32, 64)


def expanded(mask):
    return functional.max_pool2d(mask.float()[None, None], 5,
                                  stride=1, padding=2)[0, 0].bool()


def direction(vector):
    x, y = vector
    if abs(x) == abs(y):
        return None
    return ('right' if x > 0 else 'left') if abs(x) > abs(y) else (
        'down' if y > 0 else 'up')


def evaluate(objects, *, background=True, active=range(3, 15)):
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(background)
    trace = torch.zeros((2, 32, 64))
    vectors = torch.zeros((len(objects), 2))
    local_energy = torch.zeros(len(objects))
    total_energy = 0.
    for frame in range(18):
        combined = torch.zeros((32, 64), dtype=torch.bool)
        masks = []
        if 2 <= frame < 15:
            for item in objects:
                step = frame-8 if item['moving'] else 0
                center = (item['center'][0]+item['velocity'][0]*step,
                          item['center'][1]+item['velocity'][1]*step)
                mask = object_mask(item['shape'], center)
                combined |= mask
                masks.append(mask)
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        image[0, combined] = not background
        current = events_map(camera.observe(image))
        motion = local_motion_map(trace, current)
        if frame in active:
            energy = motion.abs().sum(0)
            total_energy += float(energy.sum())
            for index, mask in enumerate(masks):
                region = expanded(mask)
                vectors[index] += motion[:, region].sum(1)
                local_energy[index] += energy[region].sum()
        trace.mul_(DECAY).add_(current)
    return dict(vectors=vectors.tolist(),
        predictions=[direction(vector) for vector in vectors.tolist()],
        local_energy=local_energy.tolist(), total_energy=total_energy)


def item(shape, center, direction_name, speed):
    velocity = {'up': (-speed, 0), 'down': (speed, 0),
                'left': (0, -speed), 'right': (0, speed)}[direction_name]
    return dict(shape=shape, center=center, velocity=velocity, moving=True)


@torch.no_grad()
def main():
    trials = []
    static = []
    for shape in SHAPES:
        for direction_name in ('up', 'down', 'left', 'right'):
            centers = ((16, 20), (16, 44)) if direction_name in ('up', 'down') else (
                (10, 32), (22, 32))
            for center in centers:
                for speed in (1, 2):
                    for background in (True, False):
                        obj = item(shape, center, direction_name, speed)
                        result = evaluate([obj], background=background)
                        vector = result['vectors'][0]
                        primary = abs(vector[1 if direction_name in ('up', 'down') else 0])
                        wrong = abs(vector[0 if direction_name in ('up', 'down') else 1])
                        trials.append(dict(shape=shape, direction=direction_name,
                            center=center, speed=speed,
                            polarity='dark' if background else 'bright',
                            predicted=result['predictions'][0], vector=vector,
                            wrong_axis_ratio=wrong/max(primary, 1e-9),
                            localization=result['local_energy'][0]
                                /max(result['total_energy'], 1e-9)))
        for background in (True, False):
            still = dict(shape=shape, center=(16, 32),
                         velocity=(0, 0), moving=False)
            result = evaluate([still], background=background)
            static.append(dict(shape=shape,
                polarity='dark' if background else 'bright',
                vector=result['vectors'][0], energy=result['total_energy']))

    separated = []
    for axis in ('vertical', 'horizontal'):
        if axis == 'vertical':
            objects = [item('square', (16, 20), 'up', 1),
                       item('plus', (16, 44), 'down', 1),
                       dict(shape='ell', center=(4, 32), velocity=(0, 0), moving=False)]
            expected = ('up', 'down')
        else:
            objects = [item('square', (10, 32), 'left', 1),
                       item('plus', (22, 32), 'right', 1),
                       dict(shape='ell', center=(16, 4), velocity=(0, 0), moving=False)]
            expected = ('left', 'right')
        result = evaluate(objects)
        separated.append(dict(axis=axis, expected=expected,
            predicted=result['predictions'], vectors=result['vectors'],
            local_energy=result['local_energy'], total_energy=result['total_energy']))

    crossing = [item('square', (16, 32), 'right', 1),
                item('plus', (16, 32), 'left', 1)]
    overlap = {name: evaluate(crossing, active=frames)
               for name, frames in (('pre', range(3, 5)), ('post', range(12, 15)))}
    report = dict(trace_decay_frames=8, offsets_pixels=[1, 2], trials=trials,
        single_accuracy=dict(correct=sum(row['predicted'] == row['direction']
                                         for row in trials), total=len(trials)),
        static=static, separated=separated, crossing=overlap)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(single_accuracy=report['single_accuracy'],
        static_max_energy=max(row['energy'] for row in static),
        separated=[row['predicted'] for row in separated],
        crossing={name: row['predictions'] for name, row in overlap.items()})),
        flush=True)


if __name__ == '__main__':
    main()
