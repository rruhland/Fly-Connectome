"""Infer changed motor effects from the player's own gaming commands during play."""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve

from .tt24_bounce import BounceArena
from .tt24_effects import transmission
from .tt24_ongoing_game import History, controls, fit_native, residual, stream, DIR


class OnlineDrive:
    """Fit the actuator mixing that best predicts recent measured joint changes."""

    def __init__(self, reference):
        self.response = reference[:20].reshape(4, 5, 5)
        self.history = reference[20:]
        self.precision = np.eye(25)*.002
        self.cross = np.eye(5).ravel()*.002
        self.samples = 0
        self.mix = np.eye(5)
        self.inverse = np.eye(5)
        self.commands = np.eye(5)*.01
        self.direction = np.zeros(5)

    def update(self, features, delta):
        actions = features[:20].reshape(4,5)
        # Each physical actuator responds to a linear combination of public channels.
        design = np.einsum('lio,lj->oij',self.response,actions).reshape(5,25)
        self.precision += design.T@design
        self.cross += design.T@(delta - features[20:]@self.history)
        self.commands += np.outer(actions[0],actions[0])
        self.samples += 1
        if self.samples % 6 == 0:
            self.mix = solve(self.precision,self.cross,assume_a='pos').reshape(5,5)
            u,s,v = np.linalg.svd(self.mix,full_matrices=False)
            self.inverse = (v.T*(s/(s*s+.035)))@u.T
            self.direction = np.linalg.eigh(self.commands)[1][:,0]
            if np.dot(self.direction, np.arange(1,6))<0:
                self.direction *= -1

    def request(self, desired, explore):
        command = self.inverse@desired
        if explore and self.samples < 105:
            polarity = 1 if (self.samples//6)%2==0 else -1
            command += polarity*.23*self.direction
        return np.clip(command,-1,1)

    def predictor(self, reference):
        out = reference.copy()
        out[:20] = np.einsum('ij,lio->ljo',self.mix,self.response).reshape(20,5)
        return out


def play(arenas,record,native,starts,wiring,explore):
    _, reference, threshold, _ = native
    learner = None
    adapt = False
    rows = []
    for i,seed in enumerate(starts):
        changed = 7 <= i < 22
        arena = arenas[int(changed)]
        obs = arena.reset(seed=seed,split='held',regime='changed' if changed else 'standard')
        hist = History(obs)
        recent = deque(maxlen=8)
        policy = controls(record)
        switches = []
        ticks_learning = 0
        while True:
            tick = hist.ticks
            desired = policy.command(obs,sampled=tick%2==0)
            req = learner.request(desired,explore) if adapt else desired
            actual = wiring@req if changed else req
            obs,_,done = arena.step(actual,render=(tick+1)%2==0)
            pair = hist.transition(req,obs)
            if hist.ticks >= 5:
                recent.append(pair)
                if len(recent)==8 and hist.ticks%4==0:
                    native_error = residual(reference,recent)
                    if not adapt and native_error>threshold:
                        learner = OnlineDrive(reference)
                        for x,y in recent:
                            learner.update(x,y)
                        adapt = True
                        switches.append((hist.ticks,'new'))
                    elif (adapt and learner.samples>30 and native_error<threshold*.65 and
                          residual(learner.predictor(reference),recent) > max(threshold,2*native_error)):
                        adapt = False
                        switches.append((hist.ticks,'recall'))
                        recent.clear()
                if adapt:
                    learner.update(*pair)
                    ticks_learning += 1
            if done:
                break
        judge=arena.judge_metrics()
        rows.append(dict(seed=seed,score=judge['rebounds'],run=judge['longest_consecutive_rebounds'],
                         ticks=hist.ticks,learn_ticks=ticks_learning,switches=switches))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--seed',type=int,default=618043)
    p.add_argument('--streams',type=int,default=2)
    p.add_argument('--confirm',action='store_true')
    args=p.parse_args()
    with gzip.open(DIR/'recurrence-arm.json.gz','rt') as f:
        policy=json.load(f)['policies']['pong']
    with gzip.open(DIR/'motor-rate-target-only.json.gz','rt') as f:
        gains=json.load(f)['motor_effects']
    record=dict(weights=policy,motor_effects=gains)
    arenas=[BounceArena(view='wrist',width=64,height=48,arm=m) for m in ('stock','alternate')]
    rows=[]
    try:
        native=fit_native(arenas[0],record,args.seed+1)
        rng=np.random.default_rng(args.seed)
        for index in range(args.streams):
            wire=transmission(rng)
            starts=rng.integers(1,2**31-1,30).tolist()
            seed=int(rng.integers(2**31-1))
            modes={}
            conditions = ('voluntary','adaptive','frozen','familiar_wiring') if args.confirm else ('voluntary','concurrent','adaptive','frozen','familiar_wiring')
            for name in conditions:
                if name in ('voluntary','concurrent'):
                    games=play(arenas,record,native,starts,wire,name=='concurrent')
                else:
                    games=stream(arenas,record,native,starts,wire,seed,name)
                modes[name]=games
                print(index,name,[sum(q['score'] for q in games[l:r]) for l,r in ((7,12),(12,22),(22,30))],flush=True)
            rows.append(dict(wire=wire.tolist(),modes=modes))
            with gzip.open(args.out,'wt') as f:
                json.dump(dict(nominal=native[3],threshold=native[2],rows=rows),f,separators=(',',':'))
    finally:
        for arena in arenas:
            arena.close()


if __name__=='__main__':
    main()
