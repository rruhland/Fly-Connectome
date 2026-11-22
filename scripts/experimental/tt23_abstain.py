"""Estimate whether either local motor step beats holding from native Pong time alone."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt3_pong import _native_rallies
from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_pong import CAP, OPPONENT_SPEED, ROLE_MODEL, cases
from .tt23_anonymous_action import CHECK, WIDTH, FixedMotor
from .tt23_anonymous_replication import TRAIN
from .tt23_direction import CHOICES, MIRROR, NATIVE as EARLY_HELD, durations
from .tt23_optical_contact import OUT


SOURCE = OUT/'abstain-source.json.gz'
DELAY_OFFSETS = (1327000000,1360000000)
HELD_OFFSET = 1398000000
DELAYS = (5,12,20,28,36)
DELAY_SEED = 585331921
SHUFFLE_SEED = 142828901
SETTINGS = tuple((width,penalty) for width in (1.,1.8) for penalty in (0.,.5,1.))
PRIOR = .7
LOCAL = np.array(CHOICES)
MODES = ('learned','shuffled','center')


class DelayedMotor(FixedMotor):
    def __init__(self,role,seed,choice,delay):
        super().__init__(role,seed,choice)
        self.delay = delay
        self.clock = 0
        self.begin_wait = None
        self.decision = None

    def choose(self):
        frame = self.clock
        self.clock += 1
        eligible = self.motor.axis is not None and self.actor is not None and len(self.mover_history)>=2
        if self.context is None and eligible:
            if self.begin_wait is None:
                self.begin_wait = frame
            if frame-self.begin_wait>=self.delay:
                self.decision = frame
                return super().choose()
            body = np.asarray(self.actor.position)
            velocity = (self.actor_history[-1]-self.actor_history[0])/max(1,len(self.actor_history)-1)
            return self.motor.action(float((self.motor.ready-body-6.*velocity)@self.motor.axis))
        return super().choose()


def collect(offset,count,role):
    rng = np.random.default_rng(np.random.SeedSequence([DELAY_SEED,offset]))
    starts = cases(count,offset)
    waits = rng.choice(DELAYS,size=count).tolist()
    queue = [(case,delay,j) for case,delay in zip(starts,waits,strict=True) for j in CHOICES]
    by_seed = {case.serve:[] for case in starts}
    for begin in range(0,len(queue),42):
        batch = queue[begin:begin+42]
        agents = [DelayedMotor(role,c.exploration,j,delay) for c,delay,j in batch]
        samples = _native_rallies([c for c,_,_ in batch],agents,max_ticks=CAP,opponent_speed=OPPONENT_SPEED)
        for (case,delay,j),agent,output in zip(batch,agents,samples,strict=True):
            context = None if agent.context is None else agent.context.tolist()
            by_seed[case.serve].append(dict(output,delay=delay,decision=agent.decision,context=context,option=j))
    rows = []
    for seed,trials in by_seed.items():
        first = trials[0]
        if any(row['case']!=first['case'] or row['context']!=first['context'] or row['decision']!=first['decision']
               or row['option']!=choice for choice,row in zip(CHOICES,trials,strict=True)):
            raise ValueError('native alternatives changed their shared observation window')
        rows.append(dict(seed=seed,context=first['context'],options=trials))
    observed = [r['context'] for r in rows if r['context'] is not None]
    print('native delayed',offset,len(rows),len(observed),'opposite motion',
          sum(x[0]*x[1]<0 for x in observed),flush=True)
    return rows


def arrays(rows):
    seen = np.array([row['context'] is not None for row in rows])
    x = np.array([row['context'] if row['context'] is not None else np.zeros(4) for row in rows])
    ticks = np.array([[trial['ticks'] for trial in row['options']] for row in rows])
    return x,ticks,seen


def early_groups():
    banks = [(1559000000,_read(CHECK)['training']),*((r['offset'],r['rows']) for r in _read(TRAIN)['sets'])]
    groups = []
    for offset,bank in banks:
        x,ticks,seeds = durations(bank)
        if len(x)!=len(seeds):
            raise ValueError('the early archive lost a dormant launch')
        groups.append(dict(offset=offset,family='early',x=x,ticks=ticks[:,LOCAL],seen=np.ones(len(x),dtype=bool),seeds=seeds))
    return groups


def delayed_group(offset,rows):
    x,ticks,seen = arrays(rows)
    return dict(offset=offset,family='delayed',x=x,ticks=ticks,seen=seen,seeds=[row['seed'] for row in rows])


def kernel(groups,setting,*,shuffle=False):
    width,penalty = setting
    positions,advantages = [],[]
    for group in groups:
        seen = group['seen']
        x,ticks = group['x'][seen],group['ticks'][seen]
        if shuffle:
            rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED,group['offset']]))
            ticks = np.take_along_axis(ticks,rng.random(ticks.shape).argsort(axis=1),axis=1)
        value = np.log1p(ticks/120.)
        delta = value[:,[0,2]]-value[:,[1]]
        positions.extend((x,x*MIRROR))
        advantages.extend((delta,delta[:,::-1]))
    x,y = np.concatenate(positions),np.concatenate(advantages)
    return dict(x=x.tolist(),advantage=y.tolist(),width=width,penalty=penalty,
                second_prior=np.mean(y*y,axis=0).tolist())


def preferences(model,query):
    x,y = np.asarray(model['x']),np.asarray(model['advantage'])
    distance = ((np.atleast_2d(query)[:,None]-x)/(model['width']*WIDTH))**2
    weights = np.exp(-.5*np.minimum(distance.sum(axis=-1),100.))
    mass = weights.sum(axis=1)[:,None]+PRIOR
    mean = (weights@y)/mass
    second = (weights@(y*y)+PRIOR*np.asarray(model['second_prior']))/mass
    uncertainty = np.sqrt(np.maximum(0.,second-mean*mean)/mass)
    return mean-model['penalty']*uncertainty


def choose(model,query,seen):
    negative,positive = preferences(model,query).T
    chosen = np.argmax(np.column_stack((negative,np.full(len(query),1.e-12),positive)),axis=1)
    return np.where(seen,LOCAL[chosen],3)


def parameter_search(groups):
    results = []
    for setting in SETTINGS:
        folds = []
        for i,held in enumerate(groups):
            model = kernel([group for j,group in enumerate(groups) if i!=j],setting)
            action = choose(model,held['x'],held['seen'])-2
            ticks = held['ticks']
            gain = np.mean(ticks[np.arange(len(action)),action]-ticks[:,1])/120.
            folds.append(dict(offset=held['offset'],family=held['family'],gain=float(gain)))
        family = {kind:float(np.mean([r['gain'] for r in folds if r['family']==kind]))
                  for kind in ('early','delayed')}
        results.append(dict(setting=list(setting),folds=folds,mean_family_gain=np.mean(list(family.values())),family=family))
    selected = max(results,key=lambda r:(r['mean_family_gain'],r['setting'][1],-r['setting'][0]))
    return selected['setting'],results


def assess(rows,models):
    x,ticks,seen = arrays(rows)
    available = dict(learned=choose(models['learned'],x,seen),shuffled=choose(models['shuffled'],x,seen),
                     center=np.full(len(rows),3))
    result = {}
    for mode,choices in available.items():
        selected = [row['options'][j-2] for row,j in zip(rows,choices,strict=True)]
        result[mode] = dict(seconds=float(ticks[np.arange(len(rows)),choices-2].mean()/120.),
                            first_returns=sum(bool(r['contacts'] and r['contacts'][0]['returned']) for r in selected),
                            choices=choices.tolist(),holds=int(sum(choices==3)))
    opposite = int(np.sum((x[:,0]*x[:,1]<0)&seen))
    return dict(points=len(rows),seen=int(sum(seen)),opposite_motion=opposite,scores=result)


def run(output):
    import torch

    started = perf_counter()
    torch.set_num_threads(1)
    role = RoleModel(**_read(ROLE_MODEL))
    early = early_groups()
    practice = {offset:collect(offset,24,role) for offset in DELAY_OFFSETS}
    delayed = [delayed_group(offset,rows) for offset,rows in practice.items()]
    groups = early+delayed
    seed_list = [seed for group in groups for seed in group['seeds']]
    early_held = [dict(row,options=[row['options'][i] for i in CHOICES]) for row in _read(EARLY_HELD)['rows']]
    delayed_held = collect(HELD_OFFSET,32,role)
    held = [row['seed'] for row in early_held+delayed_held]
    if len(set(seed_list+held))!=len(seed_list)+len(held):
        raise ValueError('native pretraining and held launches overlap')
    chosen,validation = parameter_search(groups)
    models = {mode:kernel(groups,chosen,shuffle=mode=='shuffled') for mode in ('learned','shuffled')}
    diagnostics = {kind:assess(rows,models) for kind,rows in (('early',early_held),('delayed',delayed_held))}
    coverage = [dict(offset=g['offset'],family=g['family'],starts=g['seeds'],seen=int(g['seen'].sum()),
                     opposing=int(np.sum((g['x'][:,0]*g['x'][:,1]<0)&g['seen']))) for g in groups]
    result = dict(schema='tt23-abstain-native-1',sources={p.name:_hash(p) for p in (CHECK,TRAIN,EARLY_HELD,ROLE_MODEL)},
                  chosen=chosen,validation=validation,models=models,coverage=coverage,training=practice,
                  delayed_held=delayed_held,held=diagnostics,wall_seconds=perf_counter()-started)
    _write(output,result)
    print('native frozen',chosen,coverage[-2:],diagnostics,flush=True)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=SOURCE)
    run(parser.parse_args().output)
