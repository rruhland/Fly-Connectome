"""Render the paired physical witness from recorded SO-101 actions."""

import argparse
import gzip
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from .tt24_bounce import BounceArena
from .tt24_bounce_trial import verify_replay


ROOT = Path('artifacts/tt24')
PANEL = (384, 288)
TITLES = {'live': 'Live wrist', 'blind': 'Vision ablated', 'hold': 'Stationary paddle'}
BACKGROUND = '#101720'
INK = '#f0f4fa'
MUTED = '#a8b7c8'
ACCENT = '#a3e7d0'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def render(arena, episode, stride):
    verify_replay(arena, episode)
    reset = {key: episode[key] for key in ('seed', 'split', 'regime') if key in episode}
    arena.reset(**reset, render=False)
    frames = []
    for tick in range(len(episode['actions']) + 1):
        if tick % stride == 0 or tick == len(episode['actions']):
            state = arena.judge_metrics()
            frames.append((arena.render_external(), arena.render_wrist(), state['rebounds'], state['done']))
        if tick < len(episode['actions']):
            arena.step(episode['actions'][tick], render=False)
    return frames


def movie(source, output, *, seed):
    with gzip.open(source, 'rt') as stream:
        evidence = json.load(stream)
    cases = {name: next(row for row in evidence['held'][name] if row['seed'] == seed) for name in TITLES}
    width, height = PANEL
    canvas = (3 * width + 64, 2 * height + 150)
    frame_stride = 2
    arena = BounceArena(view='none', width=width, height=height,
                        apex_reward_weight=evidence.get('apex_reward_weight') or 0.)
    try:
        panels = {name: render(arena, row, frame_stride) for name, row in cases.items()}
        fps = round(1. / (arena.dt * frame_stride))
    finally:
        arena.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', f'{canvas[0]}x{canvas[1]}', '-framerate', str(fps), '-i', '-',
                '-vf', 'tpad=stop_mode=clone:stop_duration=1', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    header = ImageFont.truetype(BOLD, 29)
    caption = ImageFont.truetype(FONT, 16)
    label = ImageFont.truetype(BOLD, 19)
    with subprocess.Popen(pipeline, stdin=subprocess.PIPE) as encoder:
        for frame in range(max(len(p) for p in panels.values())):
            image = Image.new('RGB', canvas, BACKGROUND)
            draw = ImageDraw.Draw(image)
            draw.text((16, 13), 'SO-101 paddle bouncing', font=header, fill=INK)
            draw.text((16, 52), f'Paired held drop {seed}', font=caption, fill=MUTED)
            draw.text((canvas[0] - 17, 24), f'{frame / fps:0.2f} s', anchor='ra', font=label, fill=MUTED)
            for col, (name, frames) in enumerate(panels.items()):
                x = 16 + col * (width + 16)
                external, wrist, rebounds, done = frames[min(frame, len(frames) - 1)]
                draw.text((x, 88), TITLES[name], font=label, fill=INK)
                word = 'rebound' if rebounds == 1 else 'rebounds'
                draw.text((x + width, 91), f'{rebounds} {word}', anchor='ra', font=caption,
                          fill=ACCENT if name == 'live' else MUTED)
                image.paste(Image.fromarray(external), (x, 116))
                draw.text((x, 413), 'Wrist camera', font=caption, fill=MUTED)
                if done:
                    draw.text((x + width, 413), 'Drop ended', anchor='ra', font=caption, fill=MUTED)
                image.paste(Image.fromarray(wrist), (x, 437))
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('ffmpeg failed to encode the verified replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'motor-position-witness.json.gz')
    parser.add_argument('--output', type=Path, default=ROOT / 'motor-position-witness.mp4')
    parser.add_argument('--seed', type=int, default=1687620118)
    args = parser.parse_args()
    movie(args.source, args.output, seed=args.seed)


if __name__ == '__main__':
    main()
