"""Render the longest scored point from the online full-speed match with its actual two motor tapes."""
from hashlib import sha256
import json
import subprocess

import numpy as np
from PIL import Image,ImageDraw

from .tt4_arena import mujoco
from .tt4_transfer_trial import _read,_write,_hash,ROOT
from .tt21_recovery_movie import BG,INK,GREEN,MUTED,AMBER,font
from .tt22_opponent import RallyArena
from .tt22_rally_trial import CONFIG

DIR=ROOT/'artifacts/tt22'


def render(branch='frozen', *, after_service=False):
    source=DIR/('after-service.json.gz' if after_service else 'ordinary-game.json.gz')
    doc=_read(source)
    if 'verified' not in doc:
        raise ValueError('expected a complete verified game')
    by_condition=doc['rows'] if after_service else doc['branches']
    row=max(by_condition[branch],key=lambda r:r['longest'])
    paired=('pong' if branch=='neutral' else 'neutral') if after_service else ('online' if branch=='frozen' else 'frozen')
    other=next((r for r in by_condition[paired] if r['seed']==row['seed'] and r['role']==row['role']),None)
    arena=RallyArena(**CONFIG)
    arena.reset(seed=row['seed'],server=row['role'],render=False)
    watch=mujoco.Renderer(arena.model,width=600,height=360)
    digest=sha256()
    prefix='after-service' if after_service else 'ordinary-rally'
    output=DIR/f'{prefix}-{branch}.mp4'
    caption='Matched point  ·  Ordinary SO-101 serve  ·  Five-axis practice partner' if after_service else 'Full match  ·  Ordinary serve physics  ·  Five-axis practice partner'
    score_label=f'Lateral policy after the serve  ·  {branch}' if after_service else f"Game score {row['score_before'][0]}–{row['score_before'][1]} before serve"

    pipeline=['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','940x470',
              '-r','30','-i','-','-an','-c:v','libx264','-threads','1','-crf','19','-pix_fmt','yuv420p',
              '-movflags','+faststart',str(output)]
    near=far=0
    try:
        with subprocess.Popen(pipeline,stdin=subprocess.PIPE) as proc:
            for i in range(len(row['actions'])+1):
                imgs=arena.render_cameras()
                frame=Image.new('RGB',(940,470),BG)
                draw=ImageDraw.Draw(frame)
                draw.text((12,6),'SO-101 table tennis',font=font(23,True),fill=INK)
                draw.text((365,9),'Legal returns',font=font(18),fill=MUTED)
                draw.text((493,6),str(near+far),font=font(23,True),fill=GREEN)
                draw.text((640,10),f'Arm {near}    Partner {far}',font=font(16),fill=AMBER)
                draw.text((12,40),caption,font=font(15),fill=MUTED)
                frame.paste(Image.fromarray(arena.render_broadcast(watch)),(12,70))
                for j,k in enumerate(('left','right')):
                    frame.paste(Image.fromarray(imgs[k]).resize((234,176)),(626,70+181*j))
                draw.text((866,74),'L',font=font(16),fill=MUTED)
                draw.text((866,255),'R',font=font(16),fill=MUTED)
                draw.text((12,438),score_label,font=font(15),fill=INK)
                draw.text((624,438),'Actual fixed stereo input',font=font(15),fill=MUTED)
                proc.stdin.write(frame.tobytes())
                if i==len(row['actions']):
                    for _ in range(14):
                        proc.stdin.write(frame.tobytes())
                    break
                for k,pixels in imgs.items():
                    digest.update(k.encode());digest.update(pixels)
                arena._command_to=arena._aim=lambda *a:np.asarray(row['rival_actions'][i])
                _,_,_,_,info=arena.step(row['actions'][i],render=False)
                for e in info['events']:
                    if e['kind']=='legal_return':
                        near+=e['side']==1
                        far+=e['side']==-1
        if proc.returncode or digest.hexdigest()!=row['rgb'] or json.dumps(arena.events)!=json.dumps(row['events']) or arena.point_result()!=row['outcome']:
            raise ValueError('movie replay failed')
    finally:
        watch.close();arena.close()
    _write(DIR/f'{prefix}-{branch}-video.json',dict(branch=branch,source_sha256=_hash(source),video_sha256=_hash(output),
           seed=row['seed'],frames=len(row['actions']),returns=[near,far],other_condition_same_start=None if other is None else other['longest'],
           rgb_verified=digest.hexdigest()==row['rgb'],score_before=row.get('score_before')))
    print(row['seed'],near,far,flush=True)


if __name__=='__main__':
    from argparse import ArgumentParser
    p=ArgumentParser(description=__doc__)
    p.add_argument('--branch',choices=('online','frozen','neutral','pong','opposed'),default='frozen')
    a=p.parse_args()
    render(a.branch, after_service=a.branch in ('neutral','pong','opposed'))
