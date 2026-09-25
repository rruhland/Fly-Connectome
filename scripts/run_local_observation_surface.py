"""Frozen audit of persistent local camera evidence on generic and Pong scenes."""

import json
import math
import time
from pathlib import Path

import torch

from diverse_visual_experience import TEST_SHAPES
from fly_connectome.pong import Pong
from fly_connectome.sensor import EventCamera
from generic_motion_probe import events_map
from history_gated_code import DIRECTIONS
from local_observation_surface import LocalObservationSurface
from run_pong_camera_transfer import SEEDS


OUT = Path('docs/experiments/2026-09-25-local-observation-surface-results.json')
SHAPES = {**TEST_SHAPES, 'dot': ((0, 0),)}


def generic_frames(shape, direction, speed, background):
    dy, dx = direction
    frames = []
    for t in range(18):
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        if 2 <= t < 15:
            cy = math.floor(16+dy*speed*(t-8)+.5)
            cx = math.floor(32+dx*speed*(t-8)+.5)
            for oy, ox in SHAPES[shape]:
                y, x = cy+oy, cx+ox
                if 0 <= y < 32 and 0 <= x < 64:
                    image[0, y, x] = not background
        frames.append(image)
    return frames


def empty():
    return dict(frames=0, camera_active=0, quiet_frames=0,
                quiet_with_known=0, quiet_with_known_on=0,
                known_pixels=0, correct_known_pixels=0,
                changed_pixels=0, known_changed_pixels=0)


@torch.no_grad()
def audit_episode(frames, *, initial_background, row):
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(initial_background)
    surface = LocalObservationSurface()
    for t, image in enumerate(frames):
        events = events_map(camera.observe(image))
        surface.step(events)
        if t < 4:
            continue
        row['frames'] += 1
        row['camera_active'] += int(events.sum() > 0)
        row['known_pixels'] += int(surface.known.sum())
        row['correct_known_pixels'] += int((
            (surface.value.bool() == image[0]) & surface.known).sum())
        changed = image[0] != initial_background
        row['changed_pixels'] += int(changed.sum())
        row['known_changed_pixels'] += int((changed & surface.known).sum())
        if events.sum() == 0:
            row['quiet_frames'] += 1
            row['quiet_with_known'] += int(surface.known.any())
            row['quiet_with_known_on'] += int(surface.channels[1].any())


def finish(row):
    return {**row,
            'known_pixel_accuracy': row['correct_known_pixels']/max(
                row['known_pixels'], 1),
            'known_fraction': row['known_pixels']/max(
                row['frames']*32*64, 1),
            'changed_pixel_coverage': row['known_changed_pixels']/max(
                row['changed_pixels'], 1)}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    generic = empty()
    for shape in SHAPES:
        for direction in DIRECTIONS.values():
            for speed in (.25, .5, 1, 2):
                for background in (False, True):
                    audit_episode(generic_frames(shape, direction, speed,
                                                 background),
                                  initial_background=background,
                                  row=generic)
    pong_row = empty()
    for seed in SEEDS:
        pong = Pong([seed])
        frames = []
        for _ in range(120):
            frames.append(pong.render())
            pong.step(torch.zeros(1))
        audit_episode(frames, initial_background=False, row=pong_row)
    result = dict(generic_episodes=len(SHAPES)*4*4*2,
                  pong_seeds=list(SEEDS),
                  generic=finish(generic), pong=finish(pong_row),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
