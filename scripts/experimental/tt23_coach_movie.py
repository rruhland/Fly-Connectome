"""Render the two verified physical coach tapes on a shared playback clock."""

import argparse
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
from .tt21_service_drill import OPTICAL_TICKS
from .tt22_opponent import RallyArena
from .tt22_rally_trial import CONFIG
from .tt23_coach_trial import CONDITIONS, OUTPUT, RECEIPT, verify_service


SEED = 1823220138
MOVIE = OUTPUT/'coach-comparison.mp4'
SIZE = (1280, 520)
VIEW = (616, 352)
FPS = 30
HOLD = 18


class Replay:
    def __init__(self, resources, row):
        self.row = row
        self.arena = RallyArena(**CONFIG)
        resources.callback(self.arena.close)
        self.arena.reset(seed=row['seed'], server=row['role'], render=False)
        self.spectator = resources.enter_context(mujoco.Renderer(self.arena.model, width=VIEW[0], height=VIEW[1]))
        self.commands = iter(zip(row['actions'], row['rival_actions'], strict=True))
        self.rgb, self.toss = sha256(), sha256()
        self.ending = None
        self.tick = 0
        self.length = len(row['actions'])

    def picture(self):
        if self.ending is not None:
            return self.ending
        frames = self.arena.render_cameras()
        if self.tick < self.length:
            for name, pixels in frames.items():
                self.rgb.update(name.encode())
                self.rgb.update(pixels)
            if self.tick in OPTICAL_TICKS:
                self.toss.update(b'left')
                self.toss.update(frames['left'])
        view = Image.fromarray(self.arena.render_broadcast(self.spectator))
        view = view.crop((74, 44, 550, 316)).resize(VIEW, Image.Resampling.BILINEAR)
        if self.tick == self.length:
            self.ending = view
        return view

    def advance(self):
        if self.tick == self.length:
            return
        near, far = next(self.commands)
        rates = np.asarray(far)
        self.arena._aim = self.arena._command_to = lambda *args: rates
        _, self.reward, done, capped, _ = self.arena.step(near, render=False)
        self.tick += 1
        if bool(done or capped) != (self.tick == self.length):
            raise ValueError(f"termination changed at {self.row['seed']}:{self.tick}")

    def verify(self):
        row, arena = self.row, self.arena
        events = digest_json(arena.events)
        if (self.rgb.hexdigest() != row['rgb'] or self.toss.hexdigest() != row['service_rgb']
                or events != digest_json(row['events']) or self.reward != row['reward']
                or arena.point_result() != row['outcome']):
            raise ValueError('rendered replay changed the recorded RGB, physical events or result')
        return dict(frames=self.length, alternating_returns=row['longest'], near_returns=arena.legal_returns,
                    far_returns=arena.rival_returns, rgb_sha256=self.rgb.hexdigest(),
                    toss_rgb_sha256=self.toss.hexdigest(), referee_sha256=events, outcome=arena.point_result())


def frame(replays, seed):
    canvas = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), 'Same serve, two far-arm coaches', fill=INK, font=font(29, True))
    draw.text((17, 47), f'Matched point {seed}  ·  Same frozen RGB SO-101 receiver', fill=MUTED, font=font(15))
    draw.text((1264, 47), 'Serve excluded from return counts', anchor='ra', fill=MUTED, font=font(15))
    draw.line((16, 72, 1264, 72), fill=LINE)
    panes = ((16, 'Stock geometry', AMBER), (648, 'Learned continuation', GREEN))
    for replay, (x, label, accent) in zip(replays, panes, strict=True):
        arena = replay.arena
        near, far = arena.legal_returns, arena.rival_returns
        draw.text((x, 83), label, fill=accent, font=font(22, True))
        draw.text((x, 114), f'Near arm  {near}     Far coach  {far}', fill=INK, font=font(16))
        draw.text((x + VIEW[0], 112), f'Legal returns  {near + far}', anchor='ra', fill=accent, font=font(18, True))
        canvas.paste(replay.picture(), (x, 140))
        status = 'Point in play'
        if replay.ending is not None:
            outcome = replay.row['outcome']
            if outcome['reason'] == 'own_half':
                side = 'Near arm' if outcome['points_against'] else 'Far coach'
                detail = f'{side} landed on its own half'
            else:
                detail = outcome['reason'].replace('_', ' ').capitalize()
            status = f'Point ended · {detail}'
        draw.text((x, 500), status, fill=MUTED, font=font(14))
        draw.text((x + VIEW[0], 500), f'{replay.tick/FPS:.2f} s', anchor='ra', fill=INK, font=font(14))
    return canvas.tobytes()


def render(source, output, seed):
    source_hash = _hash(source)
    data = _read(source)
    if data.get('verified') != sum(len(rows) for rows in data['rows'].values()):
        raise ValueError('movie requires completed and verified native points')
    rows = [next(row for row in data['rows'][name] if row['seed'] == seed) for name in CONDITIONS]
    service = verify_service(*rows)
    total_frames = max(len(row['actions']) for row in rows) + HOLD
    with TemporaryDirectory(dir=output.parent) as directory, ExitStack() as resources:
        target = Path(directory)/'comparison.mp4'
        pipeline = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                    '-preset', 'fast', '-threads', '1', '-crf', '22', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(target)]
        replays = [Replay(resources, row) for row in rows]
        with subprocess.Popen(pipeline, stdin=subprocess.PIPE) as encoder:
            for _ in range(total_frames):
                encoder.stdin.write(frame(replays, seed))
                for replay in replays:
                    replay.advance()
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg failed to encode the paired replay')
        branches = dict(zip(CONDITIONS, (replay.verify() for replay in replays), strict=True))
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(target), '-f', 'null', '-'],
                       check=True)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                         'stream=nb_frames', '-of', 'json', str(target)])
        if int(json.loads(probe)['streams'][0]['nb_frames']) != total_frames:
            raise ValueError('the encoded video has an unexpected frame count')
        target.replace(output)
    _write(output.with_suffix('.json'), dict(source=source.name, source_sha256=source_hash,
           video_sha256=_hash(output), seed=seed, verified=True, fps=FPS, total_frames=total_frames,
           terminal_hold_frames=HOLD, service=service, branches=branches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=RECEIPT)
    parser.add_argument('--output', type=Path, default=MOVIE)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    render(args.source, args.output, args.seed)


if __name__ == '__main__':
    main()
