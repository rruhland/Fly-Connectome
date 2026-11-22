"""Learn the effect of each proposed target in the current optical context from one return per point."""
from pathlib import Path

import numpy as np
from scipy.special import betaincinv
from scipy.stats import rankdata

from .tt4_transfer_trial import ROOT, _read, _write
from .tt22_duration_transfer import MACRO,STREAMS
from .tt22_joint_practice import draw
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer_trial import MotorArena
from .tt22_population_transfer import BUDGETS, OUTPUT, games


def representation(optics, rows):
    features=[]
    for obs, trial in zip(optics,rows,strict=True):
        incoming=obs['mover']
        if incoming is None:
            features.append([[0., 0., 0., 0., 3.0]]*len(trial))
            continue
        p,v=np.array(incoming)
        # All features are available at decision time. Goal is the absolute requested
        # factory position computed from this exact prefix for each imagined action.
        features.append([[p[0]/.09,v[0]/.004,p[2]/.10,k['goal']/.085,0.] for k in trial])
    return np.asarray(features)


def kernels(train,test):
    # Each executed row corresponds to one chosen option on one practice start.
    flat=train.reshape(-1,train.shape[-1])
    def weights(x):
        dist2=((x[:,:,None,:]-flat[None,None,:,:])**2).sum(axis=3)
        return np.exp(-.5*np.minimum(dist2,60.)).astype(np.float32)
    return weights(train), weights(test)


def run(doc,test):
    n=len(doc['options'])+1
    source=np.r_[doc['source_scores'],np.median(doc['source_scores'])]
    native=(rankdata(source)-1)/(n-1)-.5
    train=representation(doc['practice_optics'],doc['practice'])
    held=representation(test['optics'],test['rows'])
    before,after=kernels(train,held)
    reward=np.asarray([[r['legal']>0 for r in row] for row in doc['practice']])
    held_reward=np.asarray([[r['legal']>0 for r in row] for row in test['rows']])
    rng=np.random.default_rng(3270969)
    curves={}
    for _ in range(160):
        order=rng.permutation(len(train))
        uniforms=rng.uniform(.0001,.9999,size=(len(order),n))
        ties=rng.uniform(size=(len(order)+1,n))*1.e-10
        for key,prior in [('native',native),('cold',np.zeros(n)),('permuted',rng.permutation(native))]:
            # Trust in native ranks is updated by predictive likelihood of the learner's actual scores.
            mixture=np.full(3,1/3)
            effectiveness=np.array([0.,.45,.85])
            traces=[]
            values=[]
            history=[]
            confirms=[]
            def posterior(weights):
                if history:
                    observed=weights[:,:,history]
                    wins=observed @ np.asarray(values,dtype=np.float32)
                    misses=observed.sum(axis=2)-wins
                else:
                    wins=misses=0.
                strength=mixture@effectiveness
                return (1.+2.*strength*prior+wins, 1.-2.*strength*prior+misses)
            for i in range(len(order)+1):
                if i in BUDGETS:
                    a,b=posterior(after)
                    scores=a/(a+b)+ties[i]
                    if scores.ndim==1:
                        scores=np.broadcast_to(scores,(len(held),n))
                    actions=np.argmax(scores,axis=1)
                    traces.append(dict(budget=i,legal=int(held_reward[np.arange(len(held)),actions].sum()),
                                       actions=actions.tolist(),trust=mixture.tolist()))
                if i==len(order):
                    break
                case=order[i]
                a,b=posterior(before[case:case+1])
                choice=int(np.argmax(betaincinv(a,b,uniforms[i])))
                success=int(reward[case,choice])
                aa,bb=float(np.ravel(a)[choice]),float(np.ravel(b)[choice])
                prob=(aa+2.*(effectiveness-mixture@effectiveness)*prior[choice])/(aa+bb)
                mixture*=prob if success else 1-prob
                mixture/=mixture.sum()
                history.append(case*n+choice)
                values.append(success)
                confirms.append(choice)
            curves.setdefault(key,[]).append(dict(eval=traces,chosen=confirms,returns=values))
    return curves


def main():
    doc=_read(OUTPUT)
    path=ROOT/'artifacts/tt22/contextual-motor.json.gz'
    rng=np.random.default_rng(121688702)
    fresh=[case for span in STREAMS.values() for case in draw(rng,32,span)]
    options=np.array(doc['options']+[[0.]*6])
    old={o['seed'] for k in ('practice_optics','held_optics') for o in doc[k]}
    if old.intersection(seed for seed,_ in fresh):
        raise ValueError('held starts reused')
    arena=MotorArena()
    test=dict(seed=121688702,optics=[],rows=[])
    try:
        test['optics']=[prefix(arena,seed,lane) for seed,lane in fresh]
        macro=np.asarray(_read(MACRO)['parameters'])
        for i in range(0,len(fresh),8):
            test['rows'].extend(games(arena,macro,options,test['optics'][i:i+8]))
            print('real starts',len(test['rows']),flush=True)
    finally:
        arena.close()
    test['curves']=run(doc,test)
    test['held']=len(fresh)
    _write(path,test)
    for name,trials in test['curves'].items():
        print(name,np.mean([[r['legal'] for r in t['eval']] for t in trials],axis=0),
              np.mean([sum(t['returns']) for t in trials]),flush=True)


if __name__=='__main__':
    main()
