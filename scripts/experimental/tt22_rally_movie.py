"""Render a scored optical SO-101 rally and verify the policy RGB and referee tape."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _read
from .tt21_recovery_movie import BG, INK, MUTED, GREEN, AMBER, font
from .tt22_rally_trial import RallyArena, opponent


def render(path, output, seed):
    data = _read(path)
    if data['privileged']:
        raise ValueError('expected a camera-only player')
    row = next(r for r in data['rows'] if r['seed'] == seed)
    arena = RallyArena(data['spread'], data['pace'])
    opponent(arena, data['rival'])
    arena.reset(seed=seed,server=-1,render=False)
    spectator = mujoco.Renderer(arena.model,width=600,height=360)
    counts = {1:0,-1:0}
    frames = sha256()
    command = ['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
               '-s','924x438','-r','30','-i','-','-an','-c:v','libx264','-preset','fast',
               '-threads','1','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    try:
        with subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
            for tick in range(len(row['actions'])+1):
                rgb = arena.render_cameras()
                if tick < len(row['actions']):
                    for name, pixels in rgb.items():
                        frames.update(name.encode())
                        frames.update(pixels)
                picture = Image.new('RGB',(924,438),BG)
                draw = ImageDraw.Draw(picture)
                draw.text((12,8),'SO-101   /   COOPERATIVE RALLY',font=font(21,True),fill=INK)
                draw.text((12,40),'Court camera',font=font(15),fill=MUTED)
                draw.text((624,40),'Fixed stereo policy inputs',font=font(15),fill=MUTED)
                picture.paste(Image.fromarray(arena.render_broadcast(spectator)),(12,68))
                for i,key in enumerate(('left','right')):
                    picture.paste(Image.fromarray(rgb[key]).resize((234,176)),(624,68+i*181))
                draw.text((865,70),'L',font=font(17),fill=MUTED)
                draw.text((865,251),'R',font=font(17),fill=MUTED)
                draw.text((472,9),f"Legal returns   {sum(counts.values())}",font=font(20,True),fill=GREEN)
                draw.text((716,12),f"Arm {counts[1]}   Rival {counts[-1]}",font=font(16),fill=AMBER)
                proc.stdin.write(picture.tobytes())
                if tick == len(row['actions']):
                    for _ in range(15):
                        proc.stdin.write(picture.tobytes())
                    break
                _,_,_,_,info=arena.step(row['actions'][tick],render=False)
                for event in info['events']:
                    if event['kind']=='legal_return':
                        counts[event['side']] += 1
        if proc.returncode:
            raise RuntimeError(f'ffmpeg exited with code {proc.returncode}')
        if (frames.hexdigest()!=row['rgb'] or arena.point_result()!=row['outcome']
                or json.dumps(arena.events)!=json.dumps(row['events'])):
            raise ValueError('replay diverged from the live policy record')
        print(seed,counts,frames.hexdigest(),flush=True)
    finally:
        spectator.close()
        arena.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args()
    render(args.source,args.output,args.seed)
