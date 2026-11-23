"""Show ordinary native joystick play followed by the first paired wrist release on drive B."""
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from fly_connectome.pong import Physics, Pong
from .tt4_transfer_trial import _read
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import HEIGHT, WIDTH
from .tt24_motor_movie import ACCENT, BACKGROUND, BOLD, FONT, INK, MUTED
from .tt24_pong_arm import PARKED
from .tt24_pong_arm_tip import GAIN, Joystick, PhysicalTipPaddle, plan
from .tt24_ready import prepare_step


def page(title, subtitle):
    im = Image.new('RGB', (900, 440), BACKGROUND)
    draw = ImageDraw.Draw(im)
    draw.text((14, 10), title, font=ImageFont.truetype(BOLD, 24), fill=INK)
    draw.text((15, 48), subtitle, font=ImageFont.truetype(FONT, 16), fill=MUTED)
    return im, draw


def text(draw, x, y, msg):
    draw.text((x, y), msg, font=ImageFont.truetype(BOLD, 16), fill=ACCENT)


def source_frames(arena, archive):
    clips = archive['clips'][:12]
    index = next(i for i, c in enumerate(clips) if c['hits'] >= 1)
    clip = clips[index]
    joystick = Joystick()
    starts, _ = plan(joystick, count=18, seed=80412719)
    seed, pose = starts[index]
    spec = Physics(opponent_speed=.35)
    game = Pong([seed], spec)
    wiring = np.array(archive['wiring'])
    paddle = PhysicalTipPaddle(arena, joystick, pose, wiring, spec)
    class Replay:
        position = paddle.position
        def step(self, drive):
            if paddle.tick % 2 == 0:
                n = paddle.tick // 2
                request = np.array(clip['commands'][n])
                prepare_step(arena, np.clip(wiring @ request, -1, 1), PARKED)
                q = arena._observation(False)['joints'].astype(float)
                if not np.array_equal(q, clip['encoders'][n + 1]):
                    raise ValueError('native source joint motion changed')
                y = .5 + (joystick.coordinate(q[None])[0] - paddle.home) * GAIN
                self.position[0] = np.clip(y, spec.paddle_half, 1 - spec.paddle_half)
            paddle.tick += 1
    game.body = Replay()
    hits = 0
    for tick in range(clip['ticks']):
        if tick % 4 == 0:
            im, draw = page('Native Pong driven by the simulated arm', 'The arm tip moves the Pong paddle. The learner records how its joints respond.')
            game_pixels = (game.render(HEIGHT, WIDTH).numpy()[0] * 255).clip(0, 255).astype(np.uint8)
            im.paste(Image.fromarray(game_pixels).convert('RGB').resize((192, 288), Image.Resampling.NEAREST), (78, 89))
            im.paste(Image.fromarray(arena.render_external()).resize((448, 336)), (416, 84))
            text(draw, 76, 391, f'Native screen   ·   {hits} returns')
            yield im
        step = game.step(torch.zeros(1), auto_reset=False)
        hits += int(step.hits.item())
    if hits != clip['hits'] or int(step.scores.item()) != clip['score']:
        raise ValueError('native Pong score or returns changed')


def target_frames(arena, seed, record):
    arena.reset(seed=seed, split='held', render=False)
    frames = []
    for tick, move in enumerate(record['actions']):
        _, _, done = arena.step(move, render=False)
        if done != (tick + 1 == len(record['actions'])):
            raise ValueError('wrist replay duration changed')
        if tick % 2 == 0 or done:
            frames.append((arena.render_external(), arena.render_wrist(), len(arena._rebounds), done))
    if digest_json(arena.judge_metrics()) != digest_json(record['metrics']):
        raise ValueError('wrist replay lost a contact')
    return frames


def main():
    torch.set_num_threads(1)
    archive = _read(ROOT/'learned-tip-transfer-1.json.gz')
    row = archive['rows'][0]
    arena = BounceArena(view='wrist', width=192, height=144, max_steps=300)
    try:
        with TemporaryDirectory(dir=ROOT) as folder:
            movie = Path(folder) / 'pong-to-wrist.mp4'
            cmd = ['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','900x440',
                   '-r','30','-i','-','-vf','tpad=stop_mode=clone:stop_duration=1','-c:v','libx264',
                   '-pix_fmt','yuv420p','-threads','1','-crf','20','-movflags','+faststart',str(movie)]
            with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
                for frame in source_frames(arena, archive):
                    encoder.stdin.write(frame.tobytes())
                groups = [target_frames(arena, row['seed'], row['conditions'][name]) for name in ('source','stale')]
                for i in range(max(map(len, groups))):
                    frame, draw = page('First wrist-only bouncing test on this drive',
                                       'Same release. Both arms already learned bouncing on three donor arms.')
                    for side, samples in enumerate(groups):
                        outside, wrist, count, done = samples[min(i,len(samples)-1)]
                        x = 14 + side * 444
                        text(draw, x, 79, ('Motor map from 68 s of Pong','Original motor map')[side])
                        frame.paste(Image.fromarray(outside).resize((424, 318)), (x, 106))
                        frame.paste(Image.fromarray(wrist).resize((132, 99)), (x+287, 320))
                        draw.rectangle((x, 394, x+236, 423), fill=BACKGROUND)
                        text(draw, x+3, 398, f'{count} rebounds' + ('   ·   End' if done else ''))
                    encoder.stdin.write(frame.tobytes())
                encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError('failed to encode native-to-wrist replay')
            movie.replace(ROOT/movie.name)
    finally:
        arena.close()


if __name__ == '__main__':
    main()
