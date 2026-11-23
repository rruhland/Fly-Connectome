"""Check frozen planar policies and render a paired powered-rebound witness."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt24_planar import PlanarBounce
from .tt24_planar_effects import OpticalEffects
from .tt24_planar_learning import LocalCredit
from .tt24_planar_trial import play
from .tt24_source import VisualConsequence


ROOT = Path('artifacts/tt24')
PIL_FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
PIL_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def restore(axes, record, gain):
    actor = LocalCredit(axes, np.random.default_rng(0), weights=np.asarray(record['actor']))
    planner = OpticalEffects(axes, source_gain=gain)
    dynamics = record['optical_effects']
    planner.motor.weights[:] = dynamics['actor']
    planner.mover.weights[:] = dynamics['mover']
    planner.motor.samples = dynamics['actor_samples']
    planner.mover.samples = dynamics['mover_samples']
    return actor, planner


def scene(setting, stage):
    return PlanarBounce(stage, max_frames=setting['max_frames'], presentation=setting['presentation'])


def evaluate(data, stage):
    setting = data['settings']
    arena = scene(setting, stage)
    trained, cold = VisualConsequence.load('pong'), VisualConsequence.load('cold')
    options = dict(stride=setting['stride'])
    report = []
    for cohort in data['cohorts']:
        block = cohort['stages'][stage]
        record = block['modes']['pong']
        actor, planner = restore(arena.action_size, record, setting['source_gain'])
        launches = block['held_seeds']
        replay = [play(arena, seed, actor, trained, planner=planner, **options)['rebounds'] for seed in launches]
        reference = record['held'][str(setting['practice'])]['scores']
        if replay != reference:
            raise ValueError('reloaded controller changed held rebound counts')
        removed = [play(arena, seed, actor, cold, planner=planner, **options)['rebounds'] for seed in launches]
        report.append(dict(seed=cohort['seed'], with_source=replay, source_removed=removed))
    return report


def witness(data):
    setting = data['settings']
    arena = scene(setting, 'damped')
    training = str(setting['practice'])
    joined = [(score, block, seed) for cohort in data['cohorts'] for block in [cohort['stages']['damped']]
              for score, seed in zip(block['modes']['pong']['held'][training]['scores'], block['held_seeds'], strict=True)]
    score, block, seed = max(joined, key=lambda row: row[0])
    actor, planner = restore(arena.action_size, block['modes']['pong'], setting['source_gain'])
    trained = VisualConsequence.load('pong')
    cases = {name: play(arena, seed, actor, trained, planner=planner, stride=setting['stride'], witness=True,
                        fixed_vertical=name == 'vertical_pinned') for name in ('live', 'vertical_pinned')}
    if cases['live']['rebounds'] != score or cases['vertical_pinned']['rebounds'] != 0:
        raise ValueError('witness is incompatible with the saved controller and energy gate')
    return arena, seed, cases


def frames(arena, case, stride):
    started = arena.reset(case['seed'])
    video = [(started, 0, False)]
    seen = sha256(started.tobytes())
    reward = 0.
    for tick, action in enumerate(case['actions'], start=1):
        image, amount, done = arena.step(action)
        reward += amount
        seen.update(image)
        if tick % stride == 0 or done:
            video.append((image, int(reward), done))
    if arena.judge()['events'] != case['events'] or reward != case['score']:
        raise ValueError('action replay changed the scored powered rebound events')
    return video, seen.hexdigest()


def movie(arena, seed, cases, output):
    rendered = {name: frames(arena, row, 2) for name, row in cases.items()}
    shape = (928, 552)
    f = ImageFont.truetype(PIL_FONT, 16)
    bold = ImageFont.truetype(PIL_BOLD, 21)
    title = ImageFont.truetype(PIL_BOLD, 30)
    subtitles = {'live': 'Live planar control', 'vertical_pinned': 'Vertical rate pinned'}
    pipe = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
            '-video_size', f'{shape[0]}x{shape[1]}', '-framerate', '30', '-i', '-',
            '-vf', 'tpad=stop_mode=clone:stop_duration=1', '-c:v', 'libx264', '-threads', '1', '-crf', '19',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    with subprocess.Popen(pipe, stdin=subprocess.PIPE) as encoder:
        for tick in range(max(len(rendered[name][0]) for name in cases)):
            image = Image.new('RGB', shape, '#14212a')
            pen = ImageDraw.Draw(image)
            pen.text((16, 12), 'Powered rebounds in planar gravity', font=title, fill='#edf4f8')
            pen.text((16, 51), f'Paired held drop {seed}', font=f, fill='#a8bac9')
            pen.text((911, 53), f'{tick / 30:.2f} s', anchor='ra', font=f, fill='#a8bac9')
            for index, key in enumerate(cases):
                x = 16 + 448 * index
                shots = rendered[key][0]
                pixels, score, done = shots[min(tick, len(shots) - 1)]
                pen.text((x, 87), subtitles[key], font=bold, fill='#edf4f8')
                count = 'rebound' if score == 1 else 'rebounds'
                pen.text((x + 432, 90), f'{score} {count}', anchor='ra', font=f, fill='#aee7db')
                if done:
                    pen.text((x + 432, 518), 'Drop ended', anchor='ra', font=f, fill='#a8bac9')
                image.paste(Image.fromarray(pixels).resize((432, 386), Image.Resampling.NEAREST), (x, 120))
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('ffmpeg failed to encode the planar witness')
    return {name: digest for name, (_, digest) in rendered.items()}


def load(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, default=ROOT / 'planar-planner.json.gz')
    parser.add_argument('--powered', type=Path, default=ROOT / 'planar-powered.json.gz')
    parser.add_argument('--output', type=Path, default=ROOT / 'planar-replay.json')
    parser.add_argument('--movie', type=Path, default=ROOT / 'planar-powered-witness.mp4')
    args = parser.parse_args()
    powered = load(args.powered)
    arena, seed, cases = witness(powered)
    digests = movie(arena, seed, cases, args.movie)
    main = load(args.main)
    retention = {stage: evaluate(main, stage) for stage in ('lateral', 'energy')}
    retention['damped'] = evaluate(powered, 'damped')
    record = dict(witness_seed=seed, cases=cases, rgb_sha256=digests, retention=retention)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print('witness', seed, {name: row['rebounds'] for name, row in cases.items()})
    print('mean rebound counts', {stage: [float(np.mean([score for row in rows for score in row[key]]))
                                        for key in ('with_source', 'source_removed')] for stage, rows in retention.items()})


if __name__ == '__main__':
    main()
