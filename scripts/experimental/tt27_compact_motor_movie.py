"""Replay the matched stale and learned motor-decoder points at normal speed."""

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
from .tt21_recovery_movie import AMBER, BG, GREEN, INK, LINE, MUTED, font
from .tt22_rally_trial import alternating
from .tt27_compact import DIMS, EYES, OUTPUT, CompactStation
from .tt27_compact_motor import PhysicalRoute, interface


SOURCE = OUTPUT / 'compact_motor.json.gz'
MOVIE = OUTPUT / 'compact_motor-live.mp4'
SEED = 1321462031
FPS = 30
HOLD = 18
SIZE = (1280, 526)
VIEW = (616, 352)
TITLES = {'stale': 'Stale motor map', 'estimated': 'Learned motor map'}


class Playback:
    def __init__(self, stack, row, actual):
        self.row = row
        self.route = PhysicalRoute(actual)
        self.arena = CompactStation.make_arena('rebound', width=DIMS[0], height=DIMS[1])
        stack.callback(self.arena.close)
        self.arena.lane = row['lane']
        self.arena.reset(seed=row['seed'], server=-1, render=False)
        cameras = self.arena.render_cameras()
        rgb = sha256()
        for eye in EYES:
            rgb.update(cameras[eye])
        if rgb.hexdigest() != row['first_rgb']:
            raise ValueError('replay did not start with the recorded stereo frame')
        self.spectator = stack.enter_context(mujoco.Renderer(self.arena.model, width=VIEW[0], height=VIEW[1]))
        self.length = len(row['actions'])
        self.frames = zip(row['wire_requests'], row['actions'], row['rival_actions'], strict=True)
        self.terminal_image = None

    def image(self, tick):
        if self.terminal_image is not None:
            return self.terminal_image
        view = Image.fromarray(self.arena.render_broadcast(self.spectator))
        view = view.crop((74, 44, 550, 316)).resize((608, 348), Image.Resampling.BILINEAR)
        if tick >= self.length:
            self.terminal_image = view
        return view

    def step(self, tick):
        if tick >= self.length:
            return
        request, recorded, far = next(self.frames)
        applied = self.route.execute(np.asarray(request))
        if not np.array_equal(applied, recorded):
            raise ValueError('wire replay did not reproduce the recorded motor action')
        far = np.asarray(far)
        self.arena._aim = self.arena._command_to = lambda *args: far
        _, reward, done, capped, _ = self.arena.step(applied, render=False)
        if bool(done or capped) != (tick + 1 == self.length):
            raise ValueError('replay changed the recorded point length')
        if tick + 1 == self.length:
            if (reward != self.row['reward'] or self.arena.point_result() != self.row['outcome']
                    or digest_json(self.arena.events) != digest_json(self.row['events'])
                    or alternating(self.arena.events) != self.row['longest']):
                raise ValueError('rendered point changed its recorded referee outcome')


def picture(plays, tick):
    canvas = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 11), 'Same serve, changed motor wiring', fill=INK, font=font(29, True))
    draw.text((21, 52), 'Solid compact court · Fresh serve · Normal speed', fill=MUTED, font=font(17))
    draw.text((1260, 20), f'{min(tick, plays["estimated"].length) / FPS:.2f} s', anchor='ra', fill=INK,
              font=font(23, True))
    draw.line((20, 81, 1260, 81), fill=LINE, width=1)
    for (mode, play), x in zip(plays.items(), (20, 652), strict=True):
        color = AMBER if mode == 'stale' else GREEN
        near, far = play.arena.legal_returns, play.arena.rival_returns
        draw.text((x, 94), TITLES[mode], fill=color, font=font(23, True))
        draw.text((x+608, 101), f'Near {near}  ·  Far {far}  ·  Total {near+far}', anchor='ra', fill=INK,
                  font=font(18, True))
        canvas.paste(play.image(tick), (x, 131))
        if tick >= play.length:
            caption = f'Point ended at {play.length / FPS:.2f} s'
        else:
            caption = 'Previous setting’s map' if mode == 'stale' else 'Map estimated on the regulation court'
        draw.text((x, 491), caption, fill=MUTED, font=font(16))
    return canvas.tobytes()


def render():
    source = _read(SOURCE)
    block, case = next((block, row) for block in source['blocks'] for row in block['cases']
                       if row['seed'] == SEED and row['role'] == 'fresh')
    rows = {mode: case['results'][mode] for mode in TITLES}
    if [row['longest'] for row in rows.values()] != [0, 12]:
        raise ValueError('expected a fresh zero-versus-twelve-return pair')
    actual = interface(block['fixture'])
    with TemporaryDirectory(dir=MOVIE.parent) as directory, ExitStack() as stack:
        temporary = Path(directory) / MOVIE.name
        plays = {mode: Playback(stack, row, actual) for mode, row in rows.items()}
        length = max(play.length for play in plays.values())
        count = length + HOLD
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                   '-threads', '1', '-preset', 'fast', '-crf', '21', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(temporary)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for tick in range(count):
                encoder.stdin.write(picture(plays, tick))
                for play in plays.values():
                    play.step(tick)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg failed to encode the comparison')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(temporary), '-f', 'null', '-'],
                       check=True)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                          'stream=nb_frames,r_frame_rate', '-of', 'json', str(temporary)])
        video = json.loads(probe)['streams'][0]
        if int(video['nb_frames']) != count or video['r_frame_rate'] != f'{FPS}/1':
            raise ValueError('encoder changed the comparison duration or rate')
        temporary.replace(MOVIE)
    records = {name: dict(frames=len(row['actions']), near=row['outcome']['legal_returns'],
                          far=row['outcome']['rival_returns'], alternating=row['longest'],
                          wire_sha256=digest_json(row['wire_requests']), motor_sha256=digest_json(row['actions']),
                          opponent_sha256=digest_json(row['rival_actions']), referee_sha256=digest_json(row['events']))
               for name, row in rows.items()}
    _write(MOVIE.with_suffix('.json'), dict(source=SOURCE.name, source_sha256=_hash(SOURCE),
           video_sha256=_hash(MOVIE), seed=SEED, fps=FPS, frames=count, freeze_frames=HOLD,
           verified_wire_and_referee=True, branches=records))


if __name__ == '__main__':
    render()
