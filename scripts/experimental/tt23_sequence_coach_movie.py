"""Render the complete 46-return receiving point from its verified two-arm command tapes."""

from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt21_recovery_movie import BG, GREEN, INK, MUTED, font
from .tt22_rally_trial import CONFIG, alternating
from .tt23_sequence import SequenceArena
from .tt23_sequence_coach import OUTPUT as SOURCE


MOVIE = SOURCE.with_name('tt23_sequence_coach_46.mp4')
SEED = 777728055
SIZE = (692, 464)
VIEW = (616, 352)
FPS = 30
HOLD = 18


def frame(arena, spectator, tick, subtitle, footer='Both arms replaying recorded control commands'):
    canvas = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    near, far = arena.legal_returns, arena.rival_returns
    draw.text((16, 10), 'SO-101 receiving rally', fill=INK, font=font(25, True))
    draw.text((17, 46), subtitle, fill=MUTED, font=font(15))
    draw.text((676, 13), f'{near + far} legal returns', anchor='ra', fill=GREEN, font=font(22, True))
    draw.text((676, 47), f'Near arm {near}    Far arm {far}', anchor='ra', fill=INK, font=font(15))
    image = Image.fromarray(arena.render_broadcast(spectator))
    image = image.crop((74, 44, 550, 316)).resize((660, 377), Image.Resampling.BILINEAR)
    canvas.paste(image, (16, 72))
    draw.rectangle((16, 424, 676, 449), fill=BG)
    draw.text((24, 430), footer, fill=MUTED, font=font(13))
    draw.text((669, 430), f'{tick / FPS:.2f} s', anchor='ra', fill=INK, font=font(13))
    return canvas.tobytes()


def render_tape(row, source, movie, *, branch, subtitle, terminal_caption, maxrate='480k', arena_factory=None):
    length = len(row['actions'])
    total = length + HOLD
    digest = sha256()
    with TemporaryDirectory(dir=movie.parent) as directory, ExitStack() as resources:
        target = Path(directory) / 'rally.mp4'
        arena = SequenceArena(**CONFIG) if arena_factory is None else arena_factory()
        resources.callback(arena.close)
        arena.lane = row['lane']
        arena.reset(seed=row['seed'], server=-1, render=False)
        spectator = resources.enter_context(mujoco.Renderer(arena.model, width=VIEW[0], height=VIEW[1]))
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                   '-threads', '1', '-preset', 'fast', '-crf', '22', '-maxrate', maxrate, '-bufsize', maxrate,
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
                for eye, pixels in arena.render_cameras().items():
                    digest.update(eye.encode())
                    digest.update(pixels)
                encoder.stdin.write(frame(arena, spectator, tick, subtitle))
                rates = np.asarray(far)
                arena._aim = arena._command_to = lambda *args, rates=rates: rates
                _, reward, done, cap, _ = arena.step(near, render=False)
                if bool(done or cap) != (tick + 1 == length):
                    raise ValueError('the playback changed the exact terminal frame')
            terminal = frame(arena, spectator, length, subtitle, terminal_caption)
            for _ in range(HOLD):
                encoder.stdin.write(terminal)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg failed to encode the receiving point')
        events = digest_json(arena.events)
        if (digest.hexdigest() != row['rgb'] or events != digest_json(row['events']) or reward != row['reward']
                or arena.point_result() != row['outcome'] or alternating(arena.events) != row['longest']):
            raise ValueError('the rendered replay changed the raw RGB, referee tape or point outcome')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(target), '-f', 'null', '-'],
                       check=True)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                         'stream=nb_frames', '-of', 'json', str(target)])
        if int(json.loads(probe)['streams'][0]['nb_frames']) != total or target.stat().st_size >= 2_000_000:
            raise ValueError('the finished movie must contain every frame and fit below 2 MB')
        target.replace(movie)
    _write(movie.with_suffix('.json'), dict(source=source.name, source_sha256=_hash(source),
           video_sha256=_hash(movie), seed=row['seed'], lane=row['lane'], branch=branch, fps=FPS,
           frames=length, total_frames=total, terminal_hold_frames=HOLD, stereo_rgb_sha256=digest.hexdigest(),
           referee_sha256=events, alternating_returns=row['longest'], near_returns=row['outcome']['legal_returns'],
           far_returns=row['outcome']['rival_returns'], verified=True, outcome=row['outcome']))


def render():
    data = _read(SOURCE)
    if data['verified'] != 50:
        raise ValueError('expected the complete verified comparison')
    case = next(case for case in data['cases'] if case['seed'] == SEED)
    row = case['results']['neutral']
    if (row['longest'], row['outcome']['legal_returns'], row['outcome']['rival_returns'], row['outcome']['reason']) != (
            46, 23, 23, 'own_half'):
        raise ValueError('the selected point no longer matches the 46-return record')
    render_tape(row, SOURCE, MOVIE, branch='neutral', subtitle='Neutral opener · Learned far coach',
                terminal_caption='Point ended · Near arm landed on its own half')


if __name__ == '__main__':
    render()
