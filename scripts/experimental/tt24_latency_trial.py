"""Measure unseen wrist frame delays and compare their effects on real bounces."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _features, _INC
from .tt24_latency import MAX_AGE, MODEL, OUT, PUPIL, LateOptics, SceneClock
from .tt24_public_body import load_public
from .tt24_wrist_egomotion import CALIBRATION, HEIGHT, MAP, WIDTH
from .tt4_transfer_trial import _hash, _read, _write

QUALIFICATION=OUT/'latency-held.json.gz'
SEEDS=tuple(range(369101,369117))
CONDITIONS=('intact','naive','adaptive','oracle','adaptive_intact')


def latency_at(seed,tick):
    """Simulator generates delays independently of the controller."""
    lead=11+seed%7
    restore=84+seed%19
    change=restore+40+seed%11
    if lead<=tick<restore:
        return 2+seed%4
    if change<=tick<change+75:
        return 2+(seed+1)%4
    return 0


def make_optics(pupil, geometry, scene, model):
    clock=SceneClock(geometry,scene['source_theta'],scene['points'],
                     model['center'],model['offset'],model['scale'])
    forecasts={age:model[f'h{age}'] for age in range(1,MAX_AGE+1)}
    return LateOptics(pupil.sensor,clock,forecasts)


def qualify(arena, seed, mode, geometry, scene, model, prior):
    agent=WristPupil.load(prior)
    obs=arena.reset(seed=seed,split='held',regime='standard')
    intact=mode in ('intact','adaptive_intact')
    local=make_optics(agent,geometry,scene,model) if mode in ('adaptive','oracle','adaptive_intact') else None
    delivered=deque(maxlen=MAX_AGE+1)
    actions=[]
    ages=[]
    while True:
        tick=len(actions)
        delivered.appendleft(obs['rgb'])
        age=0 if intact else min(latency_at(seed,tick),len(delivered)-1)
        public=dict(joints=obs['joints'],rgb=delivered[age])
        if local is None:
            action=agent.command(public)
            estimate=0
        else:
            state=local.observe(public,age if mode=='oracle' else None)
            if agent.home is None:
                agent.home=obs['joints'].copy()
            action=agent.actor.command(_features(state,obs['joints'],agent.home),agent.issued)
            agent.issued+=_INC*action
            estimate=local.age
        ages.append((age,estimate))
        actions.append(action.tolist())
        obs,_,done=arena.step(action)
        if done:
            break
    judge=arena.judge_metrics()
    arena.reset(seed=seed,split='held',regime='standard',render=False)
    for action in actions:
        arena.step(action,render=False)
    if arena.judge_metrics()!=judge:
        raise ValueError('motor tape differs from online bounce result')
    return dict(seed=seed,judge=judge,actions=actions,ages=ages)


def timing(episodes):
    ages=np.vstack([r['ages'] for r in episodes])
    result={}
    for label,mask in [('all',np.ones(len(ages),dtype=bool)),('delayed',ages[:,0]>0),('zero',ages[:,0]==0)]:
        chosen=ages[mask]
        if len(chosen):
            diff=np.abs(chosen[:,0]-chosen[:,1])
            result[label]=dict(frames=len(chosen),exact=float(np.mean(diff==0)),
                               within_one=float(np.mean(diff<=1)),mean_absolute_ticks=float(diff.mean()))
    switches={}
    for episode in episodes:
        measurement=np.asarray(episode['ages'])
        for s in np.flatnonzero(np.diff(measurement[:,0])!=0)+1:
            # Record brief, sustained agreement after the change if it occurs before the next change.
            label='restoration' if measurement[s,0]==0 else 'delay_arrival'
            consecutive=np.flatnonzero(np.convolve((measurement[s:s+31,0]==measurement[s:s+31,1]).astype(int),
                                                     np.ones(3,dtype=int),mode='valid')==3)
            next_ed=np.flatnonzero(np.diff(measurement[s:,0])!=0)
            maximum=min(30,int(next_ed[0]) if len(next_ed) else len(measurement)-s-1)
            duration=int(consecutive[0]) if len(consecutive) and consecutive[0]+2<=maximum else None
            switches.setdefault(label,[]).append(duration)
    return dict(frames=result,switches=switches)


def evaluate(args):
    geometry=load_public(args.sensor).decoder
    with np.load(args.scene,allow_pickle=False) as archive:
        scene={k:archive[k] for k in ('source_theta','points')}
    with np.load(args.model,allow_pickle=False) as archive:
        model={k:v.copy() for k,v in archive.items()}
    out=dict(schema='tt24-wrist-latency-held-1',seeds=list(SEEDS),horizon=args.horizon,
             clock_inputs='delivered wrist RGB and current/previous joint encoders',
             schedule='unchanged then 2–5 frame delay, restored, second delay and final restoration if alive',
             model_sha256=_hash(args.model),scene_sha256=_hash(args.scene),pupil_sha256=_hash(args.pupil),
             exact_physical_replay=True,results={},timing={})
    if args.resume and args.out.exists():
        original=_read(args.out)
        if any(original[key]!=value for key,value in out.items() if key not in ('results','timing')):
            raise ValueError('existing held receipt uses a different configuration')
        out['results']=original['results']
        out['timing']=original['timing']
    arena=BounceArena(view='wrist',width=WIDTH,height=HEIGHT,max_steps=args.horizon)
    try:
        for mode in args.conditions:
            rows=out['results'].get(mode,[])
            if list(SEEDS[:len(rows)])!=[r['seed'] for r in rows]:
                raise ValueError('held seeds differ from existing tape order')
            for seed in SEEDS[len(rows):]:
                row=qualify(arena,seed,mode,geometry,scene,model,args.pupil)
                rows.append(row)
                print(mode,seed,row['judge']['rebounds'],row['judge']['longest_consecutive_rebounds'],flush=True)
                out['results'][mode]=rows
                if mode in ('adaptive','adaptive_intact'):
                    out['timing'][mode]=timing(rows)
                _write(args.out,out)
    finally:
        arena.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=QUALIFICATION)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--scene',type=Path,default=MAP)
    parser.add_argument('--sensor',type=Path,default=CALIBRATION)
    parser.add_argument('--pupil',type=Path,default=PUPIL)
    parser.add_argument('--horizon',type=int,default=330)
    parser.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=CONDITIONS)
    parser.add_argument('--resume',action='store_true')
    evaluate(parser.parse_args())
