"""Render the changing physical court and camera feeds from a real scored game tape."""
import argparse
from hashlib import sha256
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .tt4_replay import recording
from .tt4_transfer_trial import _read
from .tt6_continuous_trial import prepare_point
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt17_motor_protocol import wiring_for
from .tt18_combined import fresh_arena
from .tt18_viewplay_trial import layout, specification


@lru_cache(maxsize=2)
def font_at(size):
    return ImageFont.truetype('DejaVuSans.ttf', size=size)


def draw(rgb, score, changed, count, contacts):
    w, h = next(iter(rgb.values())).shape[1::-1]
    # Keep missing feeds blank so a removed camera visibly disappears.
    canvas = Image.new('RGB', (3*w, h+88), '#151b20')
    pen = ImageDraw.Draw(canvas)
    font, header = font_at(17), font_at(23)
    pen.text((12,7), f'Arm  {score[0]}   –   {score[1]}  Opponent', font=header, fill='white')
    caption = 'Compact table, heavier ball and wrist load' if changed else 'Original table and arm'
    pen.text((3*w-12,10), caption, font=font, fill='#dfebed',anchor='rt')
    for i,(key,frame) in enumerate(rgb.items()):
        canvas.paste(Image.fromarray(frame), (i*w,39))
        pen.text((i*w+10, h+44), f'Camera {key}', font=font, fill='#e6e6e6')
    pen.text((3*w-12,h+67), f'{count} active view{"s" if count != 1 else ""}    ·    Arm returns this point  {contacts}',
             font=font, fill='white',anchor='rm')
    return canvas


def movie(receipt, output):
    from contextlib import ExitStack
    game = _read(receipt)
    spec = specification(game['seed'])
    score = ServiceScore(first_server=-1)
    target = game.get('target_body','gantry')
    with ExitStack() as lifetime:
        arena = append = None
        for i,row in enumerate(game['points']):
            if i in (0,2):
                if arena is not None:
                    arena.close()
                arena = fresh_arena(spec,i==2,target,render=True)
                lifetime.callback(arena.close)
                wiring = wiring_for(arena,spec)
            _, streams = layout('adaptive',i)
            arena.capture(streams)
            interface = PointInterface(wiring,prepare_point(arena,row['phase']),seed=game['seed'],server=score.server)
            frames = interface.reset(seed=row['seed'])
            digest = sha256()
            for tick in range(len(row['actions'])+1):
                for k,rgb in frames.items():
                    digest.update(k.encode())
                    digest.update(rgb)
                if tick == len(row['actions']):
                    score.record(arena.point_result())
                frame = draw(frames, score.score, i>=2, len(streams), arena.legal_returns)
                if append is None:
                    append = lifetime.enter_context(recording(output,frame,arena.control_dt))
                else:
                    append(frame)
                if tick < len(row['actions']):
                    frames,*_ = interface.step(row['actions'][tick])
            if digest.hexdigest()!=row['rgb'] or row['evaluator']!=arena.point_result():
                raise ValueError(f"replayed inputs or score changed at point {row['seed']}")
        if score.score != tuple(game['score']):
            raise ValueError('final replay score differs')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    movie(args.receipt,args.output)
