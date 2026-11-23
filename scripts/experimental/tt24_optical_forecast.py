"""Use online next-image error to choose velocity estimators in the wrist pupil."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, WristState

DIR = Path('artifacts/tt24')


def extrapolators():
    models = []
    for length in (3, 5, 8):
        for order in (1, 2):
            times = np.arange(1 - length, 1, dtype=float)
            design = np.polynomial.polynomial.polyvander(times, order)
            inv = np.linalg.pinv(design)
            models.append((length, np.ones(order + 1) @ inv, inv[1] * 60, inv[0]))
    return models


class Forecast(WristState):
    """Track prediction errors using visible observations, including after abrupt changes."""

    def __init__(self, decoder, motion, mix):
        super().__init__(decoder, motion)
        self.mix = mix
        self.points = []
        self.models = extrapolators()
        self.next_point = None
        self.error = np.full((len(self.models), 3), 4.e-5)

    def read(self, ellipse, joints):
        state = super().read(ellipse, joints)
        if ellipse is None:
            self.points.clear()
            self.next_point = None
            return state
        point = state[:3].copy()
        if self.next_point is not None:
            error = np.square(self.next_point - point)
            self.error = .8 * self.error + .2 * np.minimum(error, .04)
        self.points = (self.points + [point])[-8:]
        if len(self.points) < 8:
            return state
        positions = np.asarray(self.points)
        estimates = []
        predictions = []
        centers = []
        for length, extrapolate, derivative, center in self.models:
            samples = positions[-length:]
            predictions.append(extrapolate @ samples)
            estimates.append(derivative @ samples)
            centers.append(center @ samples)
        weights = 1. / (self.error + 2.5e-6)**2
        weights /= weights.sum(axis=0)
        self.next_point = np.asarray(predictions)
        velocity = np.sum(weights * estimates, axis=0)
        smooth = np.sum(weights * centers, axis=0)
        state[:3] = (1 - self.mix)*point + self.mix*smooth
        state[3:6] = (1 - self.mix)*state[3:6] + self.mix*velocity
        return state


def run(arena, seed, mode, regime, evidence, checkpoint):
    agent = WristPupil.load(checkpoint)
    if mode:
        agent.sensor = Forecast(agent.sensor.decoder, agent.sensor.motion, mode)
    obs = arena.reset(seed=seed, split='held', regime=regime)
    tape = []
    while True:
        u = agent.command(obs)
        tape.append(u.tolist())
        obs, _, done = arena.step(u)
        if done:
            break
    result = arena.judge_metrics()
    if evidence:
        arena.reset(seed=seed, split='held', regime=regime, render=False)
        for u in tape:
            _, _, done = arena.step(u, render=False)
        if arena.judge_metrics() != result:
            raise ValueError('command replay changed the physical result')
    return dict(seed=seed, score=result['rebounds'], run=result['longest_consecutive_rebounds'],
                ticks=len(tape), tape=tape if evidence else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--mix',type=float,nargs='+',default=[0.,.35,.8])
    p.add_argument('--seed',type=int,default=420317)
    p.add_argument('--count',type=int,default=6)
    p.add_argument('--arm',choices=['stock','alternate'],default='stock')
    p.add_argument('--horizon',type=int,default=300)
    p.add_argument('--verify',action='store_true')
    p.add_argument('--checkpoint',type=Path,default=DIR/'bounce-ident-pupil.npz')
    args = p.parse_args()
    arena = BounceArena(view='wrist',width=192,height=144,arm=args.arm,max_steps=args.horizon)
    regime = 'changed' if args.arm=='alternate' else 'standard'
    seeds = np.random.default_rng(args.seed).integers(1,2**31-1,args.count).tolist()
    doc = dict(arm=args.arm,regime=regime,horizon=args.horizon,checkpoint=args.checkpoint.name,seeds=seeds,verified=args.verify,results={})
    try:
        for mix in args.mix:
            rows = [run(arena,s,mix,regime,args.verify,args.checkpoint) for s in seeds]
            doc['results'][str(mix)] = rows
            print(mix,[r['run'] for r in rows],flush=True)
            with gzip.open(args.out,'wt') as f:
                json.dump(doc,f,separators=(',',':'))
    finally:
        arena.close()


if __name__=='__main__':
    main()
