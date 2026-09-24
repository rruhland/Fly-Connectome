"""Opt-in shared local event coincidence control for M1A motion."""

import json
import math
import time
from pathlib import Path

import torch

from fly_connectome.pong import Pong
from fly_connectome.sensor import EventCamera
from m1a_motion_front_end import moving_square, static_square


OUT = Path('docs/experiments/2026-09-23-m1a-event-correlation-results.json')
DECAY = math.exp(-1/8)


def directed_correlation(trace, current):
    """Antisymmetric same-polarity coincidences at one- and two-pixel offsets."""
    x = y = 0
    for distance in (1, 2):
        x = x + (trace[..., :, :-distance]*current[..., :, distance:]).sum(
            dim=(-3, -2, -1))
        x = x - (trace[..., :, distance:]*current[..., :, :-distance]).sum(
            dim=(-3, -2, -1))
        y = y + (trace[..., :-distance, :]*current[..., distance:, :]).sum(
            dim=(-3, -2, -1))
        y = y - (trace[..., distance:, :]*current[..., :-distance, :]).sum(
            dim=(-3, -2, -1))
    return x, y


def map_events(events, batch):
    result = torch.zeros((batch, 2, 32*64))
    result[events.environments, events.on.long(), events.pixels] = 1
    return result.view(batch, 2, 32, 64)


def run_probe(frames, *, on=False):
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(not on)
    trace = torch.zeros((2, 32, 64))
    scores = torch.zeros(2)
    for t, frame in enumerate(frames):
        current = map_events(camera.observe(frame), 1)[0]
        x, y = directed_correlation(trace, current)
        if 3 <= t < 15:
            scores += torch.stack((x, y))
        trace.mul_(DECAY).add_(current)
    return [float(value) for value in scores]


def direction(scores):
    x, y = scores
    if abs(x) == abs(y):
        return None
    return ('right' if x > 0 else 'left') if abs(x) > abs(y) else (
        'down' if y > 0 else 'up')


@torch.no_grad()
def main():
    started = time.perf_counter()
    cases = ([('vertical', center, speed, False)
              for center in (10, 22, 14, 18) for speed in ((1, 2) if center in (14, 18) else (1,))]
             + [('horizontal', center, speed, False)
                for center in (18, 46, 32) for speed in ((1, 2) if center == 32 else (1,))]
             + [('vertical', 16, 1, True), ('horizontal', 32, 1, True)])
    rows = []
    for axis, center, speed, on in cases:
        static = static_square(axis, center)
        if on:
            static = [~frame for frame in static]
        static_scores = run_probe(static, on=on)
        for movement in (-1, 1):
            frames = moving_square(axis, center, movement, speed)
            if on:
                frames = [~frame for frame in frames]
            scores = run_probe(frames, on=on)
            expected = (('down' if movement > 0 else 'up') if axis == 'vertical'
                        else ('right' if movement > 0 else 'left'))
            rows.append(dict(axis=axis, center=center, speed=speed,
                polarity='on' if on else 'off', expected=expected,
                predicted=direction(scores), x_score=scores[0], y_score=scores[1],
                static_x_score=static_scores[0], static_y_score=static_scores[1]))

    seeds = tuple(range(16))
    pong = Pong(seeds)
    camera = EventCamera(len(seeds), 32, 64)
    trace = torch.zeros((len(seeds), 2, 32, 64))
    counts = {axis: dict(nonzero=0, correct=0, total=0)
              for axis in ('x', 'y')}
    for _ in range(800):
        current = map_events(camera.observe(pong.render()), len(seeds))
        current[..., :6] = 0
        current[..., 58:] = 0
        x_score, y_score = directed_correlation(trace, current)
        for axis, score, truth in (('x', x_score, pong.ball_velocity[:, 0]),
                                   ('y', y_score, pong.ball_velocity[:, 1])):
            valid = truth != 0
            active = valid & (score != 0)
            counts[axis]['total'] += int(valid.sum())
            counts[axis]['nonzero'] += int(active.sum())
            counts[axis]['correct'] += int(((score*truth) > 0)[active].sum())
        trace.mul_(DECAY).add_(current)
        pong.step(torch.zeros(len(seeds)))
    for row in counts.values():
        row['coverage'] = row['nonzero']/row['total'] if row['total'] else None
        row['accuracy_when_active'] = row['correct']/row['nonzero'] if row['nonzero'] else None
    off = [row for row in rows if row['polarity'] == 'off']
    blanks = {name: run_probe([torch.full((1, 32, 64), not on,
                          dtype=torch.bool)]*18, on=on)
              for name, on in (('off', False), ('on', True))}
    report = dict(trace_decay_frames=8, offsets_pixels=[1, 2],
        square_trials=rows, square_off_accuracy=dict(
            correct=sum(row['predicted'] == row['expected'] for row in off),
            total=len(off), abstentions=sum(row['predicted'] is None for row in off)),
        static_false_alarms=sum(bool(row['static_x_score'] or row['static_y_score'])
                                for row in rows),
        blank_scores=blanks,
        pong_axes=counts, pong_frames_per_seed=800, pong_seeds=list(seeds),
        elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in
        ('square_off_accuracy', 'static_false_alarms', 'pong_axes', 'elapsed_seconds')}),
        flush=True)


if __name__ == '__main__':
    main()
