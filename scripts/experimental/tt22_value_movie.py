"""Render a verified native rally tape scored by the trajectory learner."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import mujoco
from PIL import Image, ImageDraw, ImageFont

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_imaging_trial import ArchivedOpponent
from .tt22_rally_trial import OUTPUT, Station


SIZE = (960, 540)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
TITLE = 'SO-101 · learned trajectory value'


def decorate(renderer,arena,camera,title):
    renderer.update_scene(arena.data,camera=camera)
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    size = ImageFont.truetype(FONT,20)
    small = ImageFont.truetype(FONT,15)
    legal = [e for e in arena.events if e['kind']=='legal_return']
    near = sum(e['side']==1 for e in legal)
    far = len(legal)-near
    draw.rectangle((0,0,SIZE[0],66),fill=(16,24,30))
    draw.text((20,9),title,font=size,fill='white')
    draw.text((20,39),f'Native replay — near {near}   far {far}   legal returns',font=small,fill='white')
    draw.text((SIZE[0]-135,18),f'{arena.steps/30:.2f} s',font=size,fill='white')
    return image.tobytes()


def make_movie(source,output,mode,minimum,*,seed=None,title=TITLE):
    data = _read(source)
    points = [pair[mode] for pair in data['pairs'] if mode in pair and (seed is None or pair[mode]['seed']==seed)]
    if not points:
        raise ValueError('no point matches the requested ranker and opening')
    selected = max(points,key=lambda row:row['longest'])
    if selected['longest']<minimum:
        raise ValueError('the requested ranker did not reach the requested run length')
    station = Station(data['spread'],data['pace'],data['rival'])
    arena = station.arena
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.,0.,.91)
    camera.distance = 2.55
    camera.azimuth = 123.
    camera.elevation = -24.
    arena.model.vis.global_.offwidth,arena.model.vis.global_.offheight = SIZE
    try:
        arena.reset(seed=selected['seed'],server=-1,render=False)
        arena._aim = ArchivedOpponent(station,selected['rival'])
        with mujoco.Renderer(arena.model,height=SIZE[1],width=SIZE[0]) as renderer:
            with subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                                   '-s',f'{SIZE[0]}x{SIZE[1]}','-r','20','-i','-','-an','-c:v','libx264',
                                   '-preset','fast','-threads','1','-crf','20','-pix_fmt','yuv420p',
                                   '-movflags','+faststart',str(output)],stdin=subprocess.PIPE) as encoder:
                for action in selected['actions']:
                    encoder.stdin.write(decorate(renderer,arena,camera,title))
                    _,reward,_,_,_ = arena.step(action,render=False)
                ending = decorate(renderer,arena,camera,title)
                for _ in range(24):
                    encoder.stdin.write(ending)
                encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError('ffmpeg failed to encode native live play replay')
        if (arena.point_result()!=selected['outcome'] or reward!=selected['reward']
                or json.dumps(arena.events)!=json.dumps(selected['events'])):
            raise ValueError('physical movie replay differs from the scored points')
    finally:
        station.close()
    subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(output),'-f','null','-'],check=True)
    _write(output.with_suffix('.json'),dict(source=source.name,source_sha256=_hash(source),ranker=mode,
                                           seed=selected['seed'],verified=True,longest=selected['longest'],
                                           outcome=selected['outcome'],actions=len(selected['actions']),
                                           video_sha256=_hash(output),output_fps=20,physical_fps=30))


def make_comparison(source,output,mode,comparison,seed,minimum):
    labels = {'pong':'Pong preparation','stationary':'Stationary preparation','mirrored':'Mirrored preparation'}
    names = (mode,comparison)
    with TemporaryDirectory() as directory:
        clips = [Path(directory)/f'{name}.mp4' for name in names]
        for name,clip,required in zip(names,clips,(minimum,0),strict=True):
            make_movie(source,clip,name,required,seed=seed,title=labels.get(name,name))
        points = [_read(clip.with_suffix('.json')) for clip in clips]
        frames = max(point['actions'] for point in points)
        pad = [frames-point['actions'] for point in points]
        graph = (f'[0:v]tpad=stop_mode=clone:stop={pad[0]}[left];'
                 f'[1:v]tpad=stop_mode=clone:stop={pad[1]}[right];[left][right]hstack=inputs=2:shortest=1')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(clips[0]),'-i',str(clips[1]),
                        '-filter_complex_threads','1','-filter_complex',graph,'-an','-c:v','libx264',
                        '-threads','1','-crf','20','-movflags','+faststart',str(output)],check=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(output),'-f','null','-'],check=True)
    branches = {name:{key:point[key] for key in ('longest','outcome','actions')}
                for name,point in zip(names,points,strict=True)}
    _write(output.with_suffix('.json'),dict(source=source.name,source_sha256=_hash(source),seed=seed,verified=True,
                                           branches=branches,video_sha256=_hash(output),output_fps=20,physical_fps=30))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--mode',default='online')
    parser.add_argument('--compare')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--minimum',type=int,default=10)
    parser.add_argument('--output',type=Path,default=OUTPUT/'value-native-rally.mp4')
    args = parser.parse_args()
    if args.compare:
        if args.seed is None or args.compare==args.mode:
            parser.error('a paired replay requires one opening seed and two distinct modes')
        make_comparison(args.source,args.output,args.mode,args.compare,args.seed,args.minimum)
    else:
        make_movie(args.source,args.output,args.mode,args.minimum,seed=args.seed)
