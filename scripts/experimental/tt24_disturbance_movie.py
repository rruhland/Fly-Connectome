"""Render synchronized native RGB from two verified disturbed-flight replays."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt21_recovery_movie import BG, GREEN, INK, MUTED, font
from .tt23_sequence import CONFIG, SequenceArena
from .tt23_sequence_coach_movie import FPS, VIEW
from .tt24_disturbance import DESTINATION, _verify
from .tt24_rally_policy import EYES


MOVIE = DESTINATION.with_name('tt24_disturbance.mp4')
RECEIPT = MOVIE.with_name('tt24_disturbance-video.json')
SEED = 1690569978
PHASE = 'reply'
PANEL = (476, 272)
CANVAS = (984, 362)
HOLD = 18
LABELS = {'live': 'Live RGB joint feedback', 'fixed': 'Frozen nominal joint commands'}


def _render_branch(row, dimensions):
    arena = SequenceArena(**dict(CONFIG, width=dimensions[0], height=dimensions[1]))
    frames = []
    stereo = sha256()
    kick = {key: row['kick'][key] for key in ('tick', 'velocity_delta_x')}
    length = len(row['actions'])
    try:
        with mujoco.Renderer(arena.model, width=VIEW[0], height=VIEW[1]) as spectator:
            def capture(arena, tick):
                if tick < length:
                    raw = arena.render_cameras()
                    for eye in EYES:
                        stereo.update(eye.encode())
                        stereo.update(raw[eye])
                rgb = Image.fromarray(arena.render_broadcast(spectator))
                frames.append(rgb.crop((74, 44, 550, 316)))
            _verify(arena, row, kick, capture=capture)
    finally:
        arena.close()
    if 'rgb' in row and row['rgb'] != stereo.hexdigest():
        raise ValueError('the live replay changed the pupil camera images')
    check = dict(stereo_rgb_sha256=stereo.hexdigest(), source_stereo_matched='rgb' in row,
                 referee_sha256=digest_json(row['events']), physics_sha256=row['state_sha256'],
                 terminal_tick=length, outcome=row['outcome'], verified=True)
    return frames, check


def _compose(tick, pictures, pair):
    canvas = Image.new('RGB', CANVAS, BG)
    draw = ImageDraw.Draw(canvas)
    kick = pair['live']['kick']
    for side, left in [('live', 10), ('fixed', 498)]:
        frames = pictures[side]
        actual = min(tick, len(frames) - 1)
        caption = LABELS[side]
        count = sum(event <= actual for event in pair['metrics']['near_return_ticks'][side])
        draw.text((left, 8), caption, fill=INK, font=font(15, True))
        draw.text((left + PANEL[0], 8), f'Post-kick near returns  {count}', anchor='ra',
                  fill=GREEN, font=font(13, True))
        canvas.paste(frames[actual], (left, 31))
        if tick >= len(frames) - 1:
            draw.text((left, 312), 'Point ended', fill=MUTED, font=font(12))
    direction = '−' if kick['velocity_delta_x'] < 0 else '+'
    pulse = f'Identical lateral kick {direction}{abs(kick["velocity_delta_x"]):.2f} m/s at {kick["tick"]/FPS:.2f} s'
    draw.text((10, 337), pulse, fill=INK if tick >= kick['tick'] else MUTED, font=font(13))
    draw.text((974, 337), f'{tick/FPS:.2f} s', anchor='ra', fill=INK, font=font(13))
    return canvas.tobytes()


def render():
    receipt = _read(DESTINATION)
    case = next(row for row in receipt['cases'] if row['seed'] == SEED)
    pair = case['disturbances'][PHASE]
    if (len(receipt['cases']) != 24 or receipt['verified_tapes'] != 96 or
            pair['live']['kick'] != pair['fixed']['kick'] or
            len(pair['metrics']['near_return_ticks']['live']) != 4 or pair['metrics']['near_return_ticks']['fixed']):
        raise ValueError('the witness must match the verified later-flight 4-to-0 result')
    pictures, verified = {}, {}
    for side in LABELS:
        pictures[side], verified[side] = _render_branch(pair[side], receipt['metadata']['dimensions'])
    kick_tick = pair['live']['kick']['tick']
    if any(left.tobytes() != right.tobytes() for left, right in
           zip(pictures['live'][:kick_tick+1], pictures['fixed'][:kick_tick+1], strict=True)):
        raise ValueError('the raw broadcast RGB changed before the kick could affect position')
    last = max(len(frames) for frames in pictures.values()) - 1
    frames = last + 1 + HOLD
    with TemporaryDirectory(dir=MOVIE.parent) as directory:
        temporary = Path(directory) / 'disturbance.mp4'
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{CANVAS[0]}x{CANVAS[1]}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                   '-threads', '1', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(temporary)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for tick in range(frames):
                encoder.stdin.write(_compose(min(tick, last), pictures, pair))
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg failed to encode the disturbance replay')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(temporary),
                        '-f', 'null', '-'], check=True)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                         'stream=nb_frames', '-of', 'json', str(temporary)])
        if int(json.loads(probe)['streams'][0]['nb_frames']) != frames:
            raise ValueError('the encoded witness is missing frames')
        temporary.replace(MOVIE)
    _write(RECEIPT, dict(schema='tt24-flight-disturbance-witness-1', source=DESTINATION.name,
                         source_sha256=_hash(DESTINATION), mp4_sha256=_hash(MOVIE), seed=SEED, phase=PHASE,
                         kick=pair['live']['kick'], post_kick_near_legal_ticks=pair['metrics']['near_return_ticks'],
                         view='native MuJoCo broadcast RGB, labels outside the images',
                         pre_kick_broadcast_equal_through=kick_tick, fps=FPS, frames=frames, branches=verified))


if __name__ == '__main__':
    render()
