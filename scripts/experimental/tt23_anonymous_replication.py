"""Repeat reward-only, fixed-displacement learning across independent native source experience."""

import argparse
import numpy as np

from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_pong import ROLE_MODEL, cases
from .tt22_duration_transfer import MACRO
from .tt22_motor_transfer import MotorPrimitive
from .tt23_anonymous_action import WIDTH, collect, train
from .tt23_anonymous_target import target_context
from .tt23_optical_contact import OUT

TRAIN=OUT/'anonymous-replication-pong.json.gz'
ARM=OUT/'anonymous-replication-arm.json.gz'
RESULT=OUT/'anonymous-replication.json.gz'
OFFSETS=(1050000000,1100000000,1150000000,1200000000)


def run_source():
    import torch
    torch.set_num_threads(1)
    models=[]
    if TRAIN.exists():
        models=_read(TRAIN)['sets']
    role=RoleModel(**_read(ROLE_MODEL))
    for i,offset in enumerate(OFFSETS):
        if i<len(models):
            if models[i]['offset']!=offset: raise ValueError('source mismatch')
            continue
        data=collect(cases(48,offset),{str(j):(j,None) for j in range(7)},role)
        model,_=train(data)
        values=np.array(model['value'])[::2]
        rng=np.random.default_rng(offset+7)
        noise=np.take_along_axis(values,rng.random(values.shape).argsort(axis=1),axis=1)
        shuffled=dict(x=model['x'],value=np.stack((noise,noise[:,::-1]),axis=1).reshape(-1,7).tolist())
        models.append(dict(offset=offset,rows=data,model=model,shuffled=shuffled))
        _write(TRAIN,dict(sets=models))
        print('finished native set',i+1,flush=True)


def assess():
    source=_read(TRAIN)['sets']
    if len(source)!=len(OFFSETS): raise ValueError('incomplete source')
    rows=_read(ARM)['rows']
    ready=MotorPrimitive(np.array(_read(MACRO)['parameters']),None).ready_world
    features=[target_context(r['sight'],ready) for r in rows]
    seen=np.array([x is not None for x in features])
    grid=np.array([np.zeros(4) if x is None else x for x in features])
    target=np.array([[[a['contacts']>0,a['legal']>0] for a in r['options']] for r in rows])
    def act(scores):
        choice=scores.argmax(axis=1)
        choice[~seen]=3
        return target[np.arange(len(rows)),choice].sum(axis=0)
    prepared=[]
    for run in source:
        pos=np.array(run['model']['x'])
        w=np.exp(-.5*np.minimum(100.,np.sum(((grid[:,None]-pos)/WIDTH)**2,axis=2)))
        y=np.array(run['model']['value'])
        prepared.append((w,y))
    learned=[w@y for w,y in prepared]
    shuffled=[w@r['shuffled']['value'] for (w,y),r in zip(prepared,source)]
    rng=np.random.default_rng(1584932)
    def control(w,labels):
        flat=labels[::2]
        y=np.take_along_axis(flat,rng.random(flat.shape).argsort(axis=1),axis=1)
        return w@np.stack((y,y[:,::-1]),axis=1).reshape(-1,7)
    n=[]
    pooled_null=[]
    for _ in range(1024):
        control_scores=[control(w,y) for w,y in prepared]
        pooled_null.append(act(sum(control_scores)))
        n.append(np.mean([act(s) for s in control_scores],axis=0))
    n=np.array(n)
    pooled_null=np.asarray(pooled_null)
    chosen=np.array([act(s) for s in learned])
    pooled=dict(x=sum((r['model']['x'] for r in source),[]),value=sum((r['model']['value'] for r in source),[]))
    result=dict(native_hash=_hash(TRAIN),arm_hash=_hash(ARM),replicate_scores=chosen.tolist(),
                shuffled_scores=[act(s).tolist() for s in shuffled],pooled=act(sum(learned)).tolist(),
                pooled_shuffled=act(sum(shuffled)).tolist(),neutral=target[:,3].sum(axis=0).tolist(),
                mean_null=n.mean(0).tolist(),p_mean=((1+(n>=chosen.mean(0)).sum(0))/(1+len(n))).tolist(),
                pooled_null_median=np.median(pooled_null,axis=0).tolist(),
                pooled_p=((1+(pooled_null>=act(sum(learned))).sum(axis=0))/(1+len(n))).tolist(),
                pooled_null=pooled_null.tolist(),null_draws=n.tolist(),pooled_model=pooled)
    _write(RESULT,result)
    print({k:v for k,v in result.items() if k not in ('pooled_model','null_draws','pooled_null')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',action='store_true')
    a=p.parse_args()
    run_source() if a.fit else assess()
