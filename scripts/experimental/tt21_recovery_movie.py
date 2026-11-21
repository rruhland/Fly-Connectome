"""Render a synchronized, pixel-verified TT21 held recovery comparison."""

import argparse
from functools import lru_cache
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt4_arena import mujoco
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt20_fixture import TT20Arena
from .tt20_opponent import install_opponent


OUTPUT = ROOT / 'artifacts/tt21'
SOURCE = OUTPUT / 'recovery-held.json.gz'
PRACTICE = OUTPUT / 'recovery-practice.json.gz'
POINT = 638131348
BRANCHES = ('frozen', 'learned')
STREAMS = ('left', 'right')
SIZE = (1056, 776)
PANEL_WIDTH = 500
SPECTATOR_HEIGHT = 330
SENSOR_SIZE = (246, 184)
BG, INK, MUTED = '#101820', '#f4f6f2', '#b5c5cf'
CHIP, LINE, GREEN, AMBER = '#1d2a35', '#334653', '#9aefbc', '#f8cf8b'


@lru_cache(maxsize=None)
def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(name, size)


def native_events(row, kind, side=None):
    return [event['step'] for event in row['events']
            if event['kind'] == kind and (side is None or event.get('side') == side)]


def _frame(arena, spectator, digest=None):
    images = arena.render_cameras()
    if tuple(images) != STREAMS:
        raise ValueError('unexpected replay policy streams')
    if digest is not None:
        for stream, image in images.items():
            digest.update(stream.encode())
            digest.update(image)
    watch = arena.render_broadcast(spectator)
    return Image.fromarray(watch), *(Image.fromarray(images[key]).resize(SENSOR_SIZE, Image.Resampling.LANCZOS)
                                    for key in STREAMS)


def replay(arena, spectator, row):
    arena.reset(seed=row['seed'], server=row['role'], render=False)
    rgb = sha256()
    frames = []
    for tick, action in enumerate(row['actions']):
        frames.append(_frame(arena, spectator, rgb))
        _, reward, ended, capped, _ = arena.step(action, render=False)
        if bool(ended or capped) != (tick == len(row['actions']) - 1):
            raise ValueError(f'point ended on the wrong control step at {tick}')
    if (rgb.hexdigest() != row['rgb'] or reward != row['reward'] or arena.point_result() != row['outcome']
            or json.dumps(arena.events) != json.dumps(row['events'])):
        raise ValueError('pixel or strict physics replay differs from archive')
    # The final post-command frame is displayed after the point ends. The
    # verified camera digest contains only the policy-observed pre-action frames.
    frames.append(_frame(arena, spectator))
    return frames


def _line(draw, xy, value, *, size=17, color=MUTED, bold=False):
    draw.text(xy, value, fill=color, font=font(size, bold))


def _status(row, tick):
    returns = [step for step in native_events(row, 'legal_return', 1) if step <= tick]
    contacts = [step for step in native_events(row, 'contact', 1) if step <= tick]
    replies = [step for step in native_events(row, 'legal_return', -1) if step <= tick]
    if tick >= len(row['actions']):
        if len(returns) == 1 and row['launch'] is None:
            return 'Point lost · no second stroke launched', len(returns)
        return 'Point lost · rival returned both shots', len(returns)
    if len(returns) >= 2:
        return f'Second legal return lands · frame {returns[1]}', len(returns)
    if len(contacts) >= 2:
        return f'Second paddle contact · frame {contacts[1]}', len(returns)
    if row['launch'] is not None and tick >= row['launch']:
        return f'Second stroke starts · frame {row["launch"]}', len(returns)
    if replies:
        return f'Rival reply lands · frame {replies[0]}', len(returns)
    if returns:
        return f'First legal return lands · frame {returns[0]}', len(returns)
    return 'Opponent serves', 0


def decorate(rows, frames, tick, *, training_count, rate):
    canvas = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    _line(draw, (20, 12), 'SO-101 learns when to start its next stroke', size=31, color=INK, bold=True)
    _line(draw, (22, 55), f'Held point {POINT}  ·  Unmarked court  ·  Strict scoring  ·  Half-speed replay', size=16)
    draw.line((20, 82, 1036, 82), fill=LINE)
    for mode, x, accent in (('frozen', 20, AMBER), ('learned', 536, GREEN)):
        row = rows[mode]
        status, count = _status(row, tick)
        label = 'BASELINE' if mode == 'frozen' else 'LEARNED'
        detail = 'Factory continuation timing' if mode == 'frozen' else 'Earlier launch for fast replies'
        _line(draw, (x, 94), label, size=23, color=accent, bold=True)
        _line(draw, (x, 125), detail, size=16)
        draw.rounded_rectangle((x+344, 94, x+499, 129), radius=7, fill=CHIP)
        _line(draw, (x+356, 102), f'{count} legal return{"s" if count != 1 else ""}', size=16, color=accent, bold=True)
        watch, left, right = frames[mode][min(tick, len(frames[mode]) - 1)]
        canvas.paste(watch, (x, 151))
        draw.rounded_rectangle((x+8, 159, x+207, 184), radius=4, fill=BG)
        _line(draw, (x+17, 162), 'Spectator · replay only', size=14, color=INK, bold=True)
        _line(draw, (x, 489), 'Policy RGB · left camera', size=14, color=INK)
        _line(draw, (x+254, 489), 'Policy RGB · right camera', size=14, color=INK)
        canvas.paste(left, (x, 512))
        canvas.paste(right, (x+254, 512))
        draw.rounded_rectangle((x, 704, x+499, 737), radius=5, fill=CHIP)
        _line(draw, (x+10, 711), status, size=16, color=accent, bold=True)
    _line(draw, (20, 749), f'Timing learned from {training_count} practice points using RGB and point reward.', size=15)
    _line(draw, (883, 749), f'Game time  {tick/rate:0.2f} s', size=15, color=INK)
    return canvas


def encode(path, rows, frames, *, training_count, rate):
    total = max(len(branch) for branch in frames.values())
    opening, closing = rate * 2, rate * 2
    with subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(rate), '-i', '-', '-an', '-c:v', 'libx264',
                           '-preset', 'fast', '-threads', '1', '-crf', '20', '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE) as encoder:
        for tick in range(total):
            pixels = decorate(rows, frames, tick, training_count=training_count, rate=rate).tobytes()
            repeats = opening if tick == 0 else closing if tick == total - 1 else 2
            for _ in range(repeats):
                encoder.stdin.write(pixels)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('video encoding failed')
    expected = opening + closing + 2 * (total - 2)
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                   'stream=codec_name,width,height,nb_frames,r_frame_rate', '-of', 'json', str(path)])
    stream, = json.loads(raw)['streams']
    if (stream['codec_name'] != 'h264' or (stream['width'], stream['height']) != SIZE
            or int(stream['nb_frames']) != expected):
        raise ValueError('video probe detected an unexpected format or frame count')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(path),
                    '-f', 'null', '-'], check=True)
    return dict(stream, playable_seconds=expected/rate, sha256=_hash(path), bytes=path.stat().st_size)


def render(source, practice, destination, receipt):
    snapshot = source.read_bytes()
    source_hash = sha256(snapshot).hexdigest()
    data = json.loads(gzip.decompress(snapshot))
    if _hash(practice) != data['practice_sha256']:
        raise ValueError('practice receipt does not match the evaluated policy')
    lessons = _read(practice)
    rows = {mode: next(row for row in data['conditions'][mode]['held'] if row['seed'] == POINT)
            for mode in BRANCHES}
    frozen, learned = (rows[mode] for mode in BRANCHES)
    difference = next(tick for tick, (a, b) in enumerate(zip(frozen['actions'], learned['actions'])) if a != b)
    if (data['arena']['marked'] or not data['arena']['strict_endline'] or frozen['role'] != learned['role']
            or difference != learned['launch'] or frozen['outcome']['points_for'] or learned['outcome']['points_for']
            or [len(native_events(rows[mode], 'legal_return', 1)) for mode in BRANCHES] != [1, 2]):
        raise ValueError('archive does not match the paired held comparison')
    destination.parent.mkdir(exist_ok=True)
    arena = TT20Arena(**data['arena'])
    install_opponent(arena)
    view = mujoco.Renderer(arena.model, width=PANEL_WIDTH, height=SPECTATOR_HEIGHT)
    rate = round(1/arena.control_dt)
    try:
        frames = {mode: replay(arena, view, rows[mode]) for mode in BRANCHES}
    finally:
        view.close()
        arena.close()
    with TemporaryDirectory(dir=destination.parent) as folder:
        path = Path(folder) / destination.name
        movie = encode(path, rows, frames, training_count=len(lessons['training']), rate=rate)
        path.replace(destination)
    verified = {mode: dict(pixels_sha256=row['rgb'],
                           point_sha256=sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                           commands_sha256=sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest(),
                           steps=len(row['actions']), legal_returns=native_events(row, 'legal_return', 1),
                           rival_returns=native_events(row, 'legal_return', -1), contacts=native_events(row, 'contact', 1),
                           continuation_launch=row['launch'], outcome=row['outcome'], rgb_verified=True, physics_verified=True)
                for mode, row in rows.items()}
    _write(receipt, dict(schema='tt21-recovery-video-1', seed=POINT, source=source.name, snapshot_sha256=source_hash,
                         practice=practice.name, practice_sha256=data['practice_sha256'],
                         training_points=len(lessons['training']), training_control_frames=data['training_frames_per_condition'],
                         arena=data['arena'], opponent='search', policy_streams=STREAMS, spectator_excluded_from_policy=True,
                         frame_seconds=1/rate, first_action_difference=difference, movie=movie, branches=verified,
                         renderer_sha256=_hash(Path(__file__))))
    print(destination, movie, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--practice', type=Path, default=PRACTICE)
    parser.add_argument('--destination', type=Path, default=OUTPUT / 'recovery-paired-638131348.mp4')
    parser.add_argument('--receipt', type=Path, default=OUTPUT / 'recovery-paired-638131348.json')
    args = parser.parse_args()
    render(args.source, args.practice, args.destination, args.receipt)
