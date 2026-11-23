"""Change actuation during actual wrist-only bouncing and learn through ordinary commands."""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena, _CHANGED_DRIVE, _CHANGED_FORCE
from .tt24_bounce_ident import WristPupil
from .tt24_effects import transmission
from .tt24_ongoing_game import DIR, History, Survey, residual
from .tt24_voluntary_id import OnlineDrive


def setup(arena, checkpoint, seed, learn_nominal=False):
    rng = np.random.default_rng(seed)
    survey = Survey(rng)
    costs = []
    complete = False
    while not complete:
        obs=arena.reset(seed=int(rng.integers(2**31-1)),split='practice',render=False)
        history=History(obs)
        survey.restart()
        while True:
            cmd=survey.act(obs)
            obs,_,done=arena.step(cmd,render=False)
            complete=survey.receive(history.transition(cmd,obs),obs,history)
            if complete or done:
                break
        costs.append(history.ticks)
    _,reference=survey.result()
    measured=[]
    performance=[]
    nominal_x, nominal_y = list(survey.states), list(survey.responses)
    for index in range(8 if learn_nominal else 4):
        if learn_nominal and index == 5:
            x,y = np.array(nominal_x),np.array(nominal_y)
            reference = np.linalg.solve(x.T@x+.025*np.eye(x.shape[1]),x.T@y)
        pupil=WristPupil.load(checkpoint)
        obs=arena.reset(seed=int(rng.integers(2**31-1)),split='practice')
        history=History(obs)
        recent=deque(maxlen=8)
        while True:
            u=pupil.command(obs)
            obs,_,done=arena.step(u)
            pair=history.transition(u,obs)
            if history.ticks>=5:
                recent.append(pair)
            if learn_nominal and index<5 and history.ticks>=5:
                nominal_x.append(pair[0])
                nominal_y.append(pair[1])
            elif len(recent)==8:
                measured.append(residual(reference,recent))
            if done:
                break
        performance.append(history.ticks)
    return reference,max(.055,2.2*np.percentile(measured,95)),dict(probes=costs,ordinary=performance)


def set_load(arena, changed, base_drag):
    gains=_CHANGED_DRIVE if changed else np.ones(6)
    force=_CHANGED_FORCE if changed else np.ones(6)
    m=arena._model
    m.actuator_gainprm[:,0]=arena._standard_gain*gains
    m.actuator_biasprm[:,1:3]=arena._standard_bias*gains[:,None]
    m.actuator_forcerange[:]=arena._standard_force*force[:,None]
    m.geom_solref[arena._blade]=(-100000.,-(1.25 if changed else 1.)*arena._drop.damping)
    arena._drag_force_scale=(1.45 if changed else 1.)*base_drag


def evaluate(arena,checkpoint,reference,threshold,starts,wiring,mode):
    active=False
    learner=None
    rows=[]
    for index,seed in enumerate(starts):
        pupil=WristPupil.load(checkpoint)
        obs=arena.reset(seed=seed,split='held')
        history=History(obs)
        recent=deque(maxlen=8)
        events=[]
        actuals=[]
        rewards=[]
        base_drag=arena._drag_force_scale
        previous=None
        while True:
            tick=history.ticks
            changed=(index==2 and tick>=24) or 3<=index<=10 or (index==11 and tick<24)
            if changed!=previous:
                set_load(arena,changed,base_drag)
                previous=changed
            desired=pupil.command(obs)
            req=learner.request(desired,False) if active else desired
            actual=wiring@req if changed and mode!='familiar' else req
            actuals.append(actual.tolist())
            obs,r,done=arena.step(actual)
            if r>0:
                rewards.append(tick)
            pair=history.transition(req,obs)
            if mode=='online' and history.ticks>=5:
                recent.append(pair)
                just_started=False
                if len(recent)==8 and history.ticks%4==0:
                    old_error=residual(reference,recent)
                    if not active and old_error>threshold:
                        learner=OnlineDrive(reference)
                        for x,y in recent:
                            learner.update(x,y)
                        active=True
                        just_started=True
                        events.append((history.ticks,'new'))
                    elif (active and learner.samples>30 and old_error<threshold*.65
                          and residual(learner.predictor(reference),recent)>max(threshold,2*old_error)):
                        active=False
                        events.append((history.ticks,'recall'))
                        recent.clear()
                if active and not just_started:
                    learner.update(*pair)
            if done:
                break
        judge=arena.judge_metrics()
        arena.reset(seed=seed,split='held',render=False)
        base_drag=arena._drag_force_scale
        previous=None
        for t,u in enumerate(actuals):
            changed=(index==2 and t>=24) or 3<=index<=10 or (index==11 and t<24)
            if changed!=previous:
                set_load(arena,changed,base_drag)
                previous=changed
            arena.step(u,render=False)
        if arena.judge_metrics()!=judge:
            raise ValueError('physical tape mismatch')
        rows.append(dict(seed=seed,run=judge['longest_consecutive_rebounds'],score=judge['rebounds'],
                         ticks=history.ticks,adaptation=events,reward_frames=rewards,actions=actuals))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--seed',type=int,default=34192071)
    p.add_argument('--streams',type=int,default=3)
    p.add_argument('--learn-nominal',action='store_true')
    args=p.parse_args()
    arena=BounceArena(view='wrist',width=192,height=144,max_steps=360)
    cp=DIR/'bounce-ident-pupil.npz'
    rng=np.random.default_rng(args.seed)
    try:
        reference,threshold,cost=setup(arena,cp,args.seed+1,args.learn_nominal)
        rows=[]
        for i in range(args.streams):
            starts=rng.integers(2**31-1,size=15).tolist()
            wiring=transmission(rng)
            modes={}
            for mode in ('online','frozen','familiar'):
                data=evaluate(arena,cp,reference,threshold,starts,wiring,mode)
                modes[mode]=data
                print(i,mode,[q['run'] for q in data],flush=True)
            rows.append(dict(wiring=wiring.tolist(),modes=modes))
            with gzip.open(args.out,'wt') as f:
                json.dump(dict(threshold=threshold,cost=cost,learn_nominal=args.learn_nominal,streams=rows,verified=True,
                               hidden_switch_frames={'drop_3':24,'drop_12':24}),f,separators=(',',':'))
    finally:
        arena.close()


if __name__=='__main__':
    main()
