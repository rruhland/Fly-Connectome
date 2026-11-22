"""Render the verified wrist-only point with its actual RGB input."""

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt21_recovery_movie import BG, GREEN, INK, MUTED, font
from .tt23_single_eye import OUTPUT, arena_for, point_path, verify


MOVIE = OUTPUT/'single-eye-wrist-rally.mp4'
SIZE = (1120, 464)
SPECTATOR = (592, 360)
FPS = 30
HOLD = 18


def decorate(arena, renderer, wrist, tick, row):
    near, far = arena.legal_returns, arena.rival_returns
    frame = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(frame)
    draw.text((16, 8), 'Wrist-only continuation', fill=INK, font=font(27, True))
    draw.text((1104, 11), f'Legal returns  {near+far}', anchor='ra', fill=GREEN, font=font(23, True))
    draw.text((16, 45), 'Spectator replay', fill=MUTED, font=font(15))
    draw.text((624, 45), 'Actual wrist RGB', fill=MUTED, font=font(15))
    draw.text((1104, 47), f'Near arm  {near}     Far coach  {far}', anchor='ra', fill=INK, font=font(15))
    frame.paste(Image.fromarray(arena.render_broadcast(renderer)), (16, 70))
    frame.paste(Image.fromarray(wrist), (624, 70))
    status = 'Point ended' if tick == len(row['actions']) else 'Point in play'
    draw.text((16, 441), f"Point {row['seed']}  ·  {status}", fill=MUTED, font=font(14))
    draw.text((1104, 441), f'{tick/FPS:.2f} s', anchor='ra', fill=INK, font=font(14))
    return frame.tobytes()


def render(source, output):
    data = _read(source)
    if data['metadata']['eyes'] != ['wrist'] or data.get('verified') != len(data['rows']):
        raise ValueError('movie requires completed and verified wrist-only points')
    row = max(data['rows'], key=lambda trial: trial['longest'])
    arena = arena_for('wrist_twin')
    try:
        with TemporaryDirectory(dir=output.parent) as directory:
            target = Path(directory)/'replay.mp4'
            pipeline = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                        '-preset', 'fast', '-threads', '1', '-crf', '22', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(target)]
            with mujoco.Renderer(arena.model, width=SPECTATOR[0], height=SPECTATOR[1]) as spectator:
                with subprocess.Popen(pipeline, stdin=subprocess.PIPE) as encoder:
                    def paint(tick, cameras):
                        if tuple(cameras) != ('wrist',):
                            raise ValueError('the replay received extra policy cameras')
                        image = decorate(arena, spectator, cameras['wrist'], tick, row)
                        for _ in range(HOLD if tick == len(row['actions']) else 1):
                            encoder.stdin.write(image)

                    verify(row, arena, paint)
                    encoder.stdin.close()
                    if encoder.wait():
                        raise RuntimeError('ffmpeg failed to encode the paired view')
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(target),
                            '-f', 'null', '-'], check=True)
            target.replace(output)
    finally:
        arena.close()
    _write(output.with_suffix('.json'), dict(source=source.name, source_sha256=_hash(source), seed=row['seed'],
           video_sha256=_hash(output), verified=True, physical_frames=len(row['actions']), fps=FPS,
           wrist_rgb_sha256=row['rgb'], referee_sha256=digest_json(row['events']), alternating_returns=row['longest'],
           near_returns=row['outcome']['legal_returns'], far_returns=row['outcome']['rival_returns']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=point_path('wrist_twin'))
    parser.add_argument('--output', type=Path, default=MOVIE)
    args = parser.parse_args()
    render(args.source, args.output)
