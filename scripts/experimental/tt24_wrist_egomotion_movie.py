"""Replay a held wrist-only match from the recorded, verified motor commands."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .tt24_bounce import BounceArena
from .tt24_wrist_egomotion import HEIGHT, OUT, TRIAL, WIDTH
from .tt4_transfer_trial import _read

WITNESS=368112


def frames(row):
    arena=BounceArena(view='wrist',width=WIDTH,height=HEIGHT,arm='alternate',max_steps=600)
    try:
        obs=arena.reset(seed=row['seed'],split='held',regime='changed')
        rendered=[obs['rgb']]
        for command in row['actions']:
            obs,_,_=arena.step(command)
            rendered.append(obs['rgb'])
        if json.loads(json.dumps(arena.judge_metrics()))!=row['judge']:
            raise ValueError('physical judging differs from the saved qualification')
        return rendered
    finally:
        arena.close()


def replay_movie(receipt,out):
    data=_read(receipt)
    conditions=('unmodified','mount')
    rows=[next(row for row in data['conditions'][name] if row['seed']==WITNESS)
          for name in conditions]
    streams=[frames(row) for row in rows]
    captions=('Untouched stock camera map','Wrist-only camera recalibration')
    rises=[np.asarray([event['apex_time'] for event in row['judge']['rebound_events']]) for row in rows]
    width,height=WIDTH*4,HEIGHT*2+62
    args=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
          '-r','30','-i','-','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(out)]
    with subprocess.Popen(args,stdin=subprocess.PIPE) as writer:
        for t in range(0,max(map(len,streams))+60,2):
            canvas=Image.new('RGB',(width,height),(14,19,22))
            draw=ImageDraw.Draw(canvas)
            draw.text((12,7),f'Held changed-arm drop {WITNESS}  |  wrist-camera action replay',fill=(236,242,244))
            for i,(stream,caption,events,row) in enumerate(zip(streams,captions,rises,rows,strict=True)):
                x=i*WIDTH*2
                image=Image.fromarray(stream[min(t,len(stream)-1)]).resize((WIDTH*2,HEIGHT*2),Image.Resampling.NEAREST)
                canvas.paste(image,(x,26))
                count=int((events<=t/60.).sum())
                stats=f'{count} scored'
                if t>=len(stream)-1:
                    stats+=f'   |   {row["judge"]["longest_consecutive_rebounds"]} consecutive'
                draw.text((x+12,HEIGHT*2+35),caption,fill=(236,242,244))
                draw.text((x+12,HEIGHT*2+49),stats,fill=(169,218,224))
            writer.stdin.write(canvas.tobytes())
        writer.stdin.close()
        if writer.wait()!=0:
            raise RuntimeError('video encoder failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,default=TRIAL)
    parser.add_argument('--out',type=Path,default=OUT/'wrist-egomotion-held.mp4')
    args=parser.parse_args()
    replay_movie(args.receipt,args.out)
