"""Render the measured 3D trajectory and actual policy cameras from saved joint commands."""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tt3_arena import HZ, NET_HEIGHT, PADDLE_RADIUS, TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, TT3Arena, arm_pose


def load_tape(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


class CourtView:
    def __init__(self):
        self.eye = np.array([3.1, -2.1, 2.1])
        self.forward = np.array([0., 0., .05]) - self.eye
        self.forward /= np.linalg.norm(self.forward)
        self.right = np.cross(self.forward, [0., 0., 1.])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.small, self.title = font(13), font(21)

    def xy(self, point):
        delta = np.asarray(point) - self.eye
        scale = 468 / (self.forward @ delta)
        return (round(297 + scale * (self.right @ delta)), round(160 - scale * (self.up @ delta)))

    def draw(self, arena, trail, notice):
        image = Image.new('RGB', (816, 382), '#14212b')
        d = ImageDraw.Draw(image)
        def line(a, b, color, width):
            d.line((self.xy(a), self.xy(b)), fill=color, width=width)
        def disk(center, radius, color):
            u, v = self.xy(center)
            d.ellipse((u-radius, v-radius, u+radius, v+radius), fill=color)
        def blade(center, normal, color):
            normal = np.asarray(normal)
            tangent = np.cross(normal, [0, 0, 1])
            tangent /= np.linalg.norm(tangent)
            vertical = np.cross(normal, tangent)
            ring = [self.xy(np.asarray(center) + PADDLE_RADIUS*(math.cos(t)*tangent+math.sin(t)*vertical))
                    for t in np.linspace(0, 2*math.pi, 24, endpoint=False)]
            d.polygon(ring, fill=color, outline='#d9e4e3')
        w, l = TABLE_HALF_WIDTH, TABLE_HALF_LENGTH
        for x in (-w+.06, w-.06):
            for y in (-l+.06, l-.06):
                line((x,y,-.63), (x,y,0), '#526577', 5)
        corners = [(-w,-l,0),(w,-l,0),(w,l,0),(-w,l,0)]
        d.polygon([self.xy(c) for c in corners], fill='#25625f',outline='#a6d1c4')
        line((0,-l,0),(0,l,0),'#98c8b8',1)
        rival = arena.rival
        line((0.,1.85,.10),rival,'#8596a1',4)
        blade(rival,arena.rival_normal,'#a293f9')
        for x in np.linspace(-w,w,25):
            line((x,0,0),(x,0,NET_HEIGHT),'#b8c7c5',1)
        line((-w,0,NET_HEIGHT),(w,0,NET_HEIGHT),'#e5efeb',2)
        shoulder, elbow, paddle, normal = arm_pose(arena.joints)
        line(shoulder,elbow,'#91adbf',8)
        line(elbow,paddle,'#9ebccc',6)
        disk(shoulder,6,'#d5e1e5')
        disk(elbow,5,'#d5e1e5')
        blade(paddle,normal,'#f09e69')
        for p,q in zip(trail,trail[1:]):
            line(p,q,'#a7a06b',1)
        disk(arena.ball,4,'#ffda64')
        d.text((13,8),'Table tennis through a three-joint arm',font=self.title,fill='#f5f8f7')
        d.text((14,40),'Spectator view  ·  orange is the learner',font=self.small,fill='#adc5ce')
        for j, (label, frame) in enumerate(zip(('Overhead camera','Side camera'),arena.frame(),strict=True)):
            raw = Image.fromarray(frame)
            y = 24 + j * 151
            d.text((566,y-19),label,font=self.small,fill='#c4d4d9')
            image.paste(raw.resize((228,133),Image.Resampling.NEAREST),(565,y))
        d.rectangle((0, 308, image.width - 1, image.height - 1), fill='#14212b')
        d.text((14,316),notice,font=self.title,fill='#ffdb86')
        d.text((14,352),f'{arena.frames / HZ:.2f} game seconds     Legal returns   arm {arena.legal_returns}   opponent {arena.rival_returns}',
               font=self.small,fill='#e4edeb')
        d.text((565,328),'Both cameras feed the learner.',font=self.small,fill='#b5c9d2')
        return image


def replay(tape, output=None):
    prepared = 'preparation_frames' in tape
    options = dict(view_seed=tape['view_seed'], variant=tape.get('variant', 'standard'),
                   distractor=tape.get('distractor', False))
    if prepared:
        from tt3_motor_beginner import PreparedTT3Arena

        arena = PreparedTT3Arena(tape['seed'], preparation_frames=tape['preparation_frames'],
                                 max_frames=tape.get('max_frames'), **options)
    else:
        arena = TT3Arena(tape['seed'], max_frames=tape.get('max_frames', 300), **options)
    for _ in range(tape.get('extra_resets',1)):
        arena.reset(render=False)
    view = CourtView() if output else None
    images, trail = [], []
    notice = 'Opponent prepares to serve' if prepared else 'Incoming serve'
    awaiting_receiver_bounce = prepared
    for action in tape['actions']:
        if arena.done:
            raise ValueError('extra actions after terminal score')
        if view and arena.frames % 2 == 0:
            images.append(view.draw(arena,trail,notice))
        _, _, _, info = arena.step(action,render=False)
        trail = [*trail[-9:], arena.ball]
        for event in info['events']:
            k = event['kind']
            if k in ('contact','legal_return'):
                name = 'Arm' if event['side'] == 1 else 'Opponent'
                notice = f'{name} strikes' if k=='contact' else f'{name} lands a legal return'
            elif k == 'serve':
                notice = 'Opponent serves'
            elif k == 'serve_bounce':
                notice = 'Serve bounces on the opponent’s half'
            elif k == 'bounce' and awaiting_receiver_bounce:
                notice = 'Serve bounces on the learner’s half'
                awaiting_receiver_bounce = False
            elif k == 'point':
                notice = {1:'Arm wins the point',-1:'Opponent wins the point',0:'Time limit reached'}[event['winner']]
    result = arena.point_result()
    for key, expected in tape.get('expected', {}).items():
        if key in result and result[key] != expected:
            raise ValueError(f'physical replay mismatch for {key}: {result[key]} vs {expected}')
    if output:
        images.append(view.draw(arena,trail,notice))
        output.parent.mkdir(parents=True,exist_ok=True)
        images[0].save(output, save_all=True, append_images=images[1:],lossless=True,method=2,
                       duration=[*([65]*(len(images)-1)),1200],loop=0)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tape',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(load_tape(args.tape),args.output)))
