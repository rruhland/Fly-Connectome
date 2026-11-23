"""Learn causal motion estimation from a native Pong player's optical experience."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil
from .tt24_optical_forecast import DIR, Forecast


def windows(positions, length=8):
    inputs, labels = [], []
    for col in np.asarray(positions).T:
        if len(col)<length+2:
            continue
        for win in sliding_window_view(col,length+2):
            if not np.all(np.isfinite(win)):
                continue
            dx = np.diff(win[:length])
            scale = max(float(np.sqrt(np.mean(dx*dx))),1.e-5)
            if scale < 1.e-4:
                continue
            inputs.append(dx/scale)
            # Symmetric optical measurement during source practice only.
            labels.append((win[-1]-win[-5])/(4*scale))
    return inputs, labels


class SourceEstimate:
    def __init__(self,x,y):
        n=x.shape[1]
        self.proj=np.random.default_rng(1581).normal(size=(n,48))*.67
        design=self.features(x)
        self.weights=np.linalg.solve(design.T@design+np.eye(design.shape[1])*2.,design.T@y)

    def features(self,x):
        return np.column_stack((np.ones(len(x)),x,np.sin(x@self.proj),np.cos(x@self.proj)))

    def estimate(self,x):
        return self.features(x)@self.weights


def native():
    import torch
    from .tt3_pong import cases_from_seeds,_native_rallies
    from .tt24_source_native import (CHECKPOINT,HELD,VisualConsequence,Collector,batches,
                                     CAP,OPPONENT_SPEED,_read,_hash)
    torch.set_num_threads(1)
    chk=_read(CHECKPOINT)
    horizon=_read(HELD)['horizon']
    policy=VisualConsequence.from_checkpoint('pong',chk)
    seeds=np.random.default_rng(6934278).integers(2**31-1,size=96).tolist()
    trains,targets,tests,labels=[],[],[],[]
    metrics=[]
    for index,batch in enumerate(batches(cases_from_seeds(seeds),16)):
        collectors=[Collector(c.exploration,policy,chk['plant'],horizon) for c in batch]
        outcomes=_native_rallies(batch,collectors,max_ticks=CAP,opponent_speed=OPPONENT_SPEED)
        for case,agent,result in zip(batch,collectors,outcomes,strict=True):
            x,y=windows(agent.path()[1])
            if index<4:
                trains.extend(x);targets.extend(y)
            else:
                tests.extend(x);labels.extend(y)
            metrics.append(dict(seed=case.serve,ticks=result['ticks'],track=agent.scene.signature.hexdigest(),samples=len(y)))
    x,y,xt,yt=map(np.asarray,(trains,targets,tests,labels))
    perm=np.random.default_rng(9138111).permutation(len(y))
    models={'pong':SourceEstimate(x,y),'shuffled':SourceEstimate(x,y[perm])}
    preds={name:m.estimate(xt) for name,m in models.items()}
    preds['untrained']=xt[:,-3:].mean(axis=1)
    results={k:float(np.sqrt(np.mean((v-yt)**2))) for k,v in preds.items()}
    np.savez_compressed(DIR/'pong-optics.npz',proj=models['pong'].proj,**{k:m.weights for k,m in models.items()})
    (DIR/'pong-optics-source.json').write_text(json.dumps(dict(source_sha256=_hash(CHECKPOINT),train=len(x),
        held=len(xt),splits=[64,32],rms=results,episodes=metrics),separators=(',',':'))+'\n')
    print(results,flush=True)


class OpticalRecall(Forecast):
    def __init__(self,decoder,motion,model):
        super().__init__(decoder,motion,.8)
        self.model=model

    def read(self,ellipse,joints):
        state=super().read(ellipse,joints)
        if len(self.points)<8:
            return state
        increments=np.diff(np.asarray(self.points),axis=0).T
        scales=np.maximum(1.e-5,np.sqrt(np.mean(increments**2,axis=1)))
        estimate=self.model.estimate(increments/scales[:,None])*scales*60
        state[3:6]=.5*state[3:6]+.5*estimate
        return state


def test(args):
    with np.load(DIR/'pong-optics.npz') as params:
        models={}
        for k in ('pong','shuffled'):
            m=SourceEstimate.__new__(SourceEstimate)
            m.proj,m.weights=params['proj'],params[k]
            models[k]=m
    arena=BounceArena(view='wrist',width=192,height=144,arm=args.arm,max_steps=900)
    regime='standard' if args.arm=='stock' else 'changed'
    seeds=np.random.default_rng(args.seed).integers(1,2**31-1,args.count).tolist()
    results={}
    try:
        for mode in ('online_only','pong','shuffled'):
            rows=[]
            for seed in seeds:
                agent=WristPupil.load(DIR/'bounce-ident-pupil.npz')
                dec,motion=agent.sensor.decoder,agent.sensor.motion
                agent.sensor=Forecast(dec,motion,.8) if mode=='online_only' else OpticalRecall(dec,motion,models[mode])
                obs=arena.reset(seed=seed,split='held',regime=regime)
                actions=[]
                while True:
                    u=agent.command(obs)
                    actions.append(u.tolist())
                    obs,_,done=arena.step(u)
                    if done:
                        break
                receipt=arena.judge_metrics()
                arena.reset(seed=seed,split='held',regime=regime,render=False)
                for u in actions:
                    arena.step(u,render=False)
                if receipt!=arena.judge_metrics():
                    raise ValueError('fixed command replay differs')
                rows.append(dict(seed=seed,run=receipt['longest_consecutive_rebounds'],ticks=len(actions),commands=actions))
            results[mode]=rows
            with gzip.open(args.out,'wt') as f:
                json.dump(dict(arm=args.arm,seeds=seeds,results=results,verified=True),f,separators=(',',':'))
            print(mode,[r['run'] for r in rows],flush=True)
    finally:
        arena.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['source','arm'])
    p.add_argument('--out',type=Path,default=DIR/'pong-optics-arm.json.gz')
    p.add_argument('--seed',type=int,default=93179857)
    p.add_argument('--count',type=int,default=16)
    p.add_argument('--arm',default='stock',choices=['stock','alternate'])
    args=p.parse_args()
    native() if args.phase=='source' else test(args)
