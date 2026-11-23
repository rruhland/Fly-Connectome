"""Replay held wrist command tapes for paired flight metrology and the witness video."""

import argparse
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .tt19_view_full_game import digest_json
from .tt24_event_align import OUT, NewWide, line_fit
from .tt24_event_align_live import MAX_STEPS, OUTCOME
from .tt24_startup_audit import ContactAudit
from .tt4_transfer_trial import _read, _write

WITNESS=371401
AUDIT=OUT/'event-align-live-audit.json.gz'
MOVIE=OUT/'event-align-live.mp4'


class AuditNew(ContactAudit,NewWide):
    pass


def inspect(arena,row,render=False):
    arena.max_steps=MAX_STEPS
    obs=arena.reset(seed=row['seed'],split='held',regime='changed',render=render)
    arena.begin_audit()
    points,frames=[],[]
    for u in row['actions']:
        if render:
            frames.append(obs['rgb'])
        points.append(arena._ball_position.copy())
        obs,_,_=arena.step(u,render=render)
    judge=arena.judge_metrics()
    if digest_json(judge)!=digest_json(row['judge']):
        raise ValueError('held wrist replay diverged')
    episode=dict(seed=row['seed'],first_horizontal_speed=None,second_edge_mm=None)
    if judge['rebound_events']:
        first=judge['rebound_events'][0]
        at=int(np.ceil(first['contact_time']*60))
        stop=min(len(points),at+7)
        if stop-at>=4:
            velocity=line_fit(points[at+1:stop],np.arange(at+1,stop))
            episode['first_horizontal_speed']=float(np.linalg.norm(velocity[0][:2]))
        going_down=[x for x in arena.audit_crossings if x['time']>first['apex_time']]
        if going_down:
            episode['second_edge_mm']=going_down[0]['edge_mm']
    return episode,frames


def audit(arg):
    receipt=_read(arg.receipt)
    arena=AuditNew(view='none')
    try:
        groups={mode:[inspect(arena,row)[0] for row in rows] for mode,rows in receipt['held'].items()}
    finally:
        arena.close()
    _write(arg.out,dict(schema='tt24-live-event-geometry-audit-1',groups=groups,best_seed=WITNESS))
    print({mode:next(row for row in episodes if row['seed']==WITNESS)
           for mode,episodes in groups.items()},flush=True)


def movie(arg):
    selected=('frozen','level_only','learned')
    data=_read(arg.receipt)
    episodes=[next(x for x in data['held'][mode] if x['seed']==WITNESS) for mode in selected]
    arena=AuditNew()
    try:
        replays=[inspect(arena,row,render=True)[1] for row in episodes]
    finally:
        arena.close()
    captions=('Frozen donor motor','Leveling comparison','Learned event response')
    apex=[np.asarray([p['apex_time'] for p in r['judge']['rebound_events']]) for r in episodes]
    width,height=1152,350
    args=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
          '-r','30','-i','-','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(arg.out)]
    with subprocess.Popen(args,stdin=subprocess.PIPE) as writer:
        for tick in range(0,max(map(len,replays))+60,2):
            canvas=Image.new('RGB',(width,height),(14,19,22))
            draw=ImageDraw.Draw(canvas)
            draw.text((12,7),f'New wide body, held drop {WITNESS}  |  Wrist-camera action replay',fill=(236,242,244))
            for k,(sequence,caption,rises) in enumerate(zip(replays,captions,apex,strict=True)):
                x=k*384
                frame=Image.fromarray(sequence[min(tick,len(sequence)-1)]).resize((384,288),Image.Resampling.NEAREST)
                canvas.paste(frame,(x,24))
                draw.text((x+12,323),caption,fill=(236,242,244))
                draw.text((x+12,336),f'{int((rises<=tick/60.).sum())} scored rebounds',fill=(169,218,224))
            writer.stdin.write(canvas.tobytes())
        writer.stdin.close()
        if writer.wait()!=0:
            raise RuntimeError('video encoder failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('audit','movie'))
    parser.add_argument('--receipt',type=Path,default=OUTCOME)
    parser.add_argument('--out',type=Path)
    arg=parser.parse_args()
    if arg.out is None:
        arg.out=AUDIT if arg.task=='audit' else MOVIE
    (audit if arg.task=='audit' else movie)(arg)
