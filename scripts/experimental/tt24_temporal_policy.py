"""Predict several future joint rates from each observation and execute one rate.

Fits an ordinary ridge readout to fixed nonlinear features. Predictions overlap
in time and are recomputed after each new frame. No action sequence is stored
or selected at deployment.
"""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_latent_control import collect


class TemporalActor:
    def __init__(self, x, targets, lengths, *, window=5, restricted=True, bandwidth=1., smoothing=2.5):
        self.window = window
        self.smoothing = smoothing
        self.restricted = restricted
        valid, chunks = [], []
        begin = 0
        for length in lengths:
            for i in range(begin, begin+length-window+1):
                valid.append(i)
                chunks.append(targets[i:i+window,5:].ravel())
            begin += length
        if begin != len(x):
            raise ValueError('episode boundaries and source labels disagree')
        x = self.state(x[valid])
        self.center = x.mean(0)
        self.scale = np.maximum(.06,x.std(0))
        r = np.random.default_rng(391593)
        self.proj = r.normal(size=(x.shape[1],512))*bandwidth/np.sqrt(x.shape[1])
        self.phase = r.uniform(-np.pi,np.pi,self.proj.shape[1])
        z = self.design(x)
        self.params = np.linalg.solve(z.T@z + .6*np.eye(z.shape[1]), z.T@np.array(chunks))
        self.reset()

    def state(self, x):
        # Absolute reset context provides a strong accidental episode identifier.
        return np.atleast_2d(x)[:,:25] if self.restricted else np.atleast_2d(x)

    def design(self, raw):
        z=np.clip((raw-self.center)/self.scale,-4.,4.)
        return np.column_stack((np.ones(len(z)), z, np.cos(z @ self.proj + self.phase)))

    def reset(self):
        self.pending = deque(maxlen=self.window)

    def command(self, x, target, dt):
        estimate = (self.design(self.state(x))@self.params).reshape(self.window,5)
        self.pending.appendleft(estimate)
        predictions = [v[i] for i,v in enumerate(self.pending)]
        weight = np.exp(-np.arange(len(predictions))/self.smoothing)
        return np.clip(np.average(predictions,axis=0,weights=weight),-.75,.75)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('records',type=Path)
    p.add_argument('episodes',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=5490071)
    p.add_argument('--held',type=int,default=12)
    p.add_argument('--window',type=int,default=5)
    p.add_argument('--wide',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args()
    lengths=[r['frames'] for r in json.load(gzip.open(args.episodes))['exposure']]
    with np.load(args.records) as data:
        pupil=TemporalActor(data['x'],data['y'],lengths,window=args.window,restricted=not args.wide)
    if args.checkpoint:
        np.savez_compressed(args.checkpoint,center=pupil.center,scale=pupil.scale,proj=pupil.proj,
                            phase=pupil.phase,params=pupil.params,window=pupil.window,restricted=pupil.restricted)
    arena=BounceArena(view='wrist',width=96,height=72)
    seeds=np.random.default_rng(args.seed).integers(1,2**31-1,args.held).tolist()
    results={}
    try:
        for label in ['live','blind','hold']:
            table=[]
            for s in seeds:
                pupil.reset()
                table.append(collect(arena,s,pupil if label!='hold' else None,blind=label!='live')[0])
            results[label]=table
            print(label,[r['consecutive'] for r in table],flush=True)
        with gzip.open(args.output,'wt') as out:
            json.dump(dict(held=seeds,window=args.window,full_state=args.wide,results=results),out)
    finally:
        arena.close()


if __name__=='__main__':
    main()
