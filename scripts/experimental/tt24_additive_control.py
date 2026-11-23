"""Learn a compact motor readout by selecting sensory interactions from demonstrations."""
import argparse
import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_latent_control import collect


class AdditivePolicy:
    def __init__(self,x,targets,lengths,max_pairs=5,lookahead=0):
        # Omit issued position and per-episode identifiers when fitting a desired position.
        raw=self.read(x)
        self.center=raw.mean(0)
        self.scale=np.maximum(raw.std(0), .04)
        z=np.clip((raw-self.center)/self.scale,-3.,3.)
        begin=0
        seed_ids=[]
        output=[]
        for idx,n in enumerate(lengths):
            block=targets[begin:begin+n,:5]
            output.extend(block[np.minimum(np.arange(n)+lookahead,n-1)])
            seed_ids.extend([idx]*n)
            begin+=n
        if begin!=len(x):
            raise ValueError('example manifest length mismatch')
        y=np.array(output)
        self.y_scale=np.maximum(.015,y.std(0))
        y=y/self.y_scale
        tr=np.array(seed_ids)%4!=2
        va=~tr
        knots=np.quantile(z,[.05,.25,.50,.75,.95],axis=0).T
        self.knots=knots
        h=self.univariate(z)
        z0=np.column_stack((np.ones(len(raw)),z,h.reshape(len(z),-1)))
        current=z0
        pairs=list(itertools.combinations(range(z.shape[1]),2))
        self.chosen=[]
        for _ in range(max_pairs):
            w=self.fit(current[tr],y[tr])
            residual=y-current@w
            # Score candidate basis functions on complete, separate training drops.
            best=None
            for i,j in pairs:
                b=(h[:,i,:,None]*h[:,j,None,:]).reshape(len(z),-1)
                weights=self.fit(b[tr],residual[tr],.85)
                error=np.mean((residual[va]-b[va]@weights)**2)
                if best is None or error<best[0]:
                    best=(error,i,j,b)
            error,i,j,b=best
            if error >= np.mean(residual[va]**2):
                break
            self.chosen.append((i,j))
            pairs.remove((i,j))
            current=np.column_stack((current,b))
        self.weights=self.fit(current,y,1.)

    @staticmethod
    def read(x):
        # Optical position/size, two time differences, visibility, own motion.
        return np.atleast_2d(x)[:,:20]

    @staticmethod
    def fit(x,y,ridge=.5):
        return np.linalg.solve(x.T@x+ridge*np.eye(x.shape[1]), x.T@y)

    def univariate(self,z):
        return np.exp(-.5*((z[:,:,None]-self.knots[None,:,:])/.66)**2)

    def command(self,x,target,dt):
        z=np.clip((self.read(x)-self.center)/self.scale,-3.,3.)
        h=self.univariate(z)
        features=[np.ones(1),z.ravel(),h.ravel()]
        for i,j in self.chosen:
            features.append(np.outer(h[0,i],h[0,j]).ravel())
        position=(np.concatenate(features)@self.weights)*self.y_scale
        return np.clip((position-target)/(5.2*dt),-.70,.70)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('records',type=Path)
    p.add_argument('episodes',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=5490071)
    p.add_argument('--held',type=int,default=12)
    p.add_argument('--pairs',type=int,default=5)
    p.add_argument('--lead',type=int,default=0)
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args()
    lengths=[r['frames'] for r in json.load(gzip.open(args.episodes))['exposure']]
    with np.load(args.records) as data:
        policy=AdditivePolicy(data['x'],data['y'],lengths,args.pairs,args.lead)
    print('selected channel pairs',policy.chosen,flush=True)
    if args.checkpoint:
        np.savez_compressed(args.checkpoint,**policy.__dict__)
    arena=BounceArena(view='wrist',width=96,height=72)
    starts=np.random.default_rng(args.seed).integers(1,2**31-1,args.held).tolist()
    results={}
    try:
        for mode in ('live','blind','hold'):
            results[mode]=[collect(arena,s,None if mode=='hold' else policy,blind=mode!='live')[0] for s in starts]
            print(mode,[x['consecutive'] for x in results[mode]],flush=True)
        with gzip.open(args.output,'wt') as f:
            json.dump(dict(starts=starts,lead=args.lead,selected=policy.chosen,results=results),f)
    finally:
        arena.close()


if __name__=='__main__':
    main()
