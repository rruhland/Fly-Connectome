"""Replay the wrist images delivered under late and corrected control."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .tt24_bounce import BounceArena
from .tt24_latency_trial import HEIGHT, OUT, QUALIFICATION, WIDTH
from .tt4_transfer_trial import _read


def frames(row,horizon):
    arena=BounceArena(view='wrist',width=WIDTH,height=HEIGHT,max_steps=horizon)
    try:
        obs=arena.reset(seed=row['seed'],split='held',regime='standard')
        source=[]
        delivered=[]
        for action,(age,_) in zip(row['actions'],row['ages'],strict=True):
            source.append(obs['rgb'])
            delivered.append(source[-age-1])
            obs,_,_=arena.step(action)
        if json.loads(json.dumps(arena.judge_metrics()))!=row['judge']:
            raise ValueError('physical judging differs from saved qualification')
        return delivered
    finally:
        arena.close()


def replay_movie(receipt,out,seed):
    data=_read(receipt)
    modes=('naive','adaptive')
    rows=[next(row for row in data['results'][mode] if row['seed']==seed) for mode in modes]
    streams=[frames(row,data['horizon']) for row in rows]
    captions=('Naive late delivery','Estimated capture time and forecast')
    rises=[np.asarray([event['apex_time'] for event in row['judge']['rebound_events']]) for row in rows]
    width,height=WIDTH*4,HEIGHT*2+70
    args=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
          '-r','30','-i','-','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(out)]
    with subprocess.Popen(args,stdin=subprocess.PIPE) as writer:
        for tick in range(0,max(map(len,streams))+60,2):
            canvas=Image.new('RGB',(width,height),(14,19,22))
            draw=ImageDraw.Draw(canvas)
            draw.text((12,7),f'Held drop {seed}   |   Delivered wrist RGB, replayed joint commands',fill=(236,242,244))
            for i,(stream,caption,events,row) in enumerate(zip(streams,captions,rises,rows,strict=True)):
                x=i*WIDTH*2
                index=min(tick,len(stream)-1)
                image=Image.fromarray(stream[index]).resize((WIDTH*2,HEIGHT*2),Image.Resampling.NEAREST)
                canvas.paste(image,(x,26))
                count=int((events<=tick/60.).sum())
                actual,inferred=row['ages'][index]
                if tick>=len(stream):
                    summary=f'{row["judge"]["rebounds"]} scored  |  {row["judge"]["longest_consecutive_rebounds"]} consecutive'
                else:
                    summary=f'Actual image age {actual} ticks   |   {count} scored'
                    if i==1:
                        summary+=f'   |   Estimated {inferred}'
                draw.text((x+12,HEIGHT*2+36),caption,fill=(236,242,244))
                draw.text((x+12,HEIGHT*2+52),summary,fill=(169,218,224))
            writer.stdin.write(canvas.tobytes())
        writer.stdin.close()
        if writer.wait()!=0:
            raise RuntimeError('video encoder failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,default=QUALIFICATION)
    parser.add_argument('--out',type=Path,default=OUT/'latency-held.mp4')
    parser.add_argument('--seed',type=int,default=369105)
    args=parser.parse_args()
    replay_movie(args.receipt,args.out,args.seed)
