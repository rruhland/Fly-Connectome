"""Replay a saved mini-simulator point with the exact RGB images used for play."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .tt3_arena import HZ
from .tt3_replay import font, load_tape
from .tt4_camera import TT3PerspectiveArena


def panel(arena, notice):
    left, right = arena.frame()
    width = 410
    height = round(width * left.shape[0] / left.shape[1])
    margin, gap, top = 16, 12, 64
    image = Image.new('RGB', (2 * (width + margin) + gap, height + top + 63), '#14212b')
    draw = ImageDraw.Draw(image)
    draw.text((margin, 9), 'Table tennis from the arm’s end of the table', font=font(22), fill='#f5f8f7')
    for index, (label, frame) in enumerate(zip(('Left camera', 'Right camera'), (left, right), strict=True)):
        x = margin + index * (width + gap)
        draw.text((x, top - 22), label, font=font(14), fill='#b4c6d1')
        image.paste(Image.fromarray(frame).resize((width, height), Image.Resampling.NEAREST), (x, top))
    draw.text((margin, top + height + 9), notice, font=font(17), fill='#ffdb86')
    stats = f'{arena.frames / HZ:.2f} s   ·   Legal returns   Arm {arena.legal_returns}   Opponent {arena.rival_returns}'
    draw.text((margin, top + height + 34), stats, font=font(14), fill='#dbe6ec')
    return image


def replay(tape, output=None):
    if tape.get('view') != 'first_person':
        raise ValueError('expected a first-person mini-simulator tape')
    arena = TT3PerspectiveArena(tape['seed'], view_seed=tape['view_seed'], variant=tape['variant'],
                                distractor=tape['distractor'], max_frames=tape['max_frames'],
                                width=tape['width'], height=tape['height'], baseline=tape['baseline'])
    for _ in range(tape['extra_resets']):
        arena.reset(render=False)

    frames = []
    actions = sha256()
    notice = 'Incoming serve'
    for action in tape['actions']:
        if output is not None and arena.frames % 2 == 0:
            frames.append(panel(arena, notice))
        actions.update(np.asarray(action, dtype='<f8').tobytes())
        _, _, _, info = arena.step(action, render=False)
        for event in info['events']:
            if event['kind'] == 'contact':
                notice = 'Arm strikes' if event['side'] == 1 else 'Opponent strikes'
            elif event['kind'] == 'legal_return':
                notice = 'Arm lands a legal return' if event['side'] == 1 else 'Opponent lands a legal return'
            elif event['kind'] == 'point':
                notice = {1: 'Arm wins the point', -1: 'Opponent wins the point', 0: 'Time limit reached'}[event['winner']]

    result = arena.point_result()
    if actions.hexdigest() != tape['action_sha256'] or result != tape['expected']:
        raise ValueError('replay diverged from the captured actions or scored outcome')
    if output is not None:
        frames.append(panel(arena, notice))
        output.parent.mkdir(parents=True, exist_ok=True)
        durations = [*([round(2000 / HZ)] * (len(frames) - 1)), 1100]
        frames[0].save(output, save_all=True, append_images=frames[1:], lossless=True, method=2,
                       duration=durations, loop=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tape', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(load_tape(args.tape), args.output)))


if __name__ == '__main__':
    main()
