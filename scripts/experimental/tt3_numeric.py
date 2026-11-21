"""Learn a numeric spatial preference by direct search using only native Pong scores."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from tt3_learning import approach_normal
from tt3_pong import cases_from_seeds, practice_pong, summarize_pong
from tt3_visual import RoleModel


class NumericMemory:
    """Generic desired displacement as a function of visible relative kinematics.

    No coefficient has a preferred sign at initialization. Opposite coefficients
    generate opposite motion; the public game score determines which survive.
    """
    def __init__(self, coefficients, totals=None, counts=None):
        self.coefficients = np.asarray(coefficients, dtype=float)
        n = len(self.coefficients)
        self.scores = np.zeros(n) if totals is None else np.array(totals,dtype=float)
        self.counts = np.zeros(n,dtype=int) if counts is None else np.array(counts,dtype=int)

    def state(self):
        return dict(coefficients=self.coefficients.tolist(), totals=self.scores.tolist(), counts=self.counts.tolist())

    @classmethod
    def restore(cls, state):
        return cls(**state)

    def choose(self, rng, explore=0.):
        if explore and rng.random() < explore:
            return int(rng.integers(len(self.coefficients)))
        means = self.scores / np.maximum(1, self.counts)
        return int(rng.choice(np.flatnonzero(np.isclose(means,means.max()))))

    def observe(self, option, reward):
        self.scores[option] += reward
        self.counts[option] += 1

    def goal(self, option, actor_position, mover_position, mover_velocity, ready_position, *,
             normal=None, acceleration=None, motion_bounds=None, response_axis=None, **unused):
        origin = np.asarray(ready_position)
        if mover_position is None or mover_velocity is None:
            return origin.copy()
        position = np.asarray(mover_position)
        velocity = np.asarray(mover_velocity)
        delta = position-origin
        if normal is None:
            normal = approach_normal(position, velocity, origin)
        direction = np.asarray(normal)
        rate = velocity @ direction
        horizon = np.clip(-(delta @ direction)/rate, 0., 90.) if abs(rate) > 1e-6 else 0.
        bias, displacement, motion = self.coefficients[option]
        # A finite set of dimensions is shared by different visual spaces.
        # The candidate can ignore, follow, or oppose each term independently.
        forecast = position + velocity*horizon
        if acceleration is not None:
            forecast += .5 * np.asarray(acceleration) * horizon**2
        if motion_bounds is not None:
            low, high = (np.array(bound) for bound in motion_bounds)
            span = high-low
            forecast = low + span - np.abs(np.mod(forecast-low, 2*span)-span)
        target_delta = displacement*delta + motion*(forecast-position)
        if response_axis is not None:
            axis = np.asarray(response_axis)
            target_delta += bias * axis / max(1e-8,np.linalg.norm(axis))
        return origin + target_delta - (target_delta @ direction)*direction


def sample_cases(rng, n):
    return cases_from_seeds(rng.choice(170_000_000,n,replace=False)+990_000_000)


def train(args):
    import torch
    torch.set_num_threads(1)
    raw_roles = json.loads(args.roles.read_text())
    roles = RoleModel(**raw_roles.get('model',raw_roles))
    rng = np.random.default_rng(args.seed)
    mean = np.zeros(3)
    scale = np.array([.11,1.4,1.4])
    records = []
    best = None
    args.out.mkdir(parents=True, exist_ok=True)
    for generation in range(args.generations):
        # Stratified, independent, zero-centered initial search. Elites update
        # center and scale from public returns on paired native starting cases.
        n = args.population
        noise = np.stack([(rng.permutation(n)+rng.random(n))*2/n-1 for _ in range(3)],axis=1)
        candidates = mean+noise*scale
        if best is not None:
            candidates[0]=best
        candidates = np.clip(candidates,[-.16,-2.5,-2.5],[.16,2.5,2.5])
        memory = NumericMemory(candidates)
        base = sample_cases(rng,args.starts)
        cases = [case for case in base for _ in candidates]
        ids = np.tile(np.arange(len(candidates)),len(base)).tolist()
        rows = practice_pong(cases,roles,memory,options=ids,batch_size=32,max_ticks=args.ticks,
                             points_per_match=args.points_per_match)
        performance = [summarize_pong([r for r in rows if r['relation']==j]) for j in range(n)]
        public = np.array([sum(r['score'] for r in rows if r['relation']==j) for j in range(n)])
        order = np.argsort(-public,kind='stable')
        best = candidates[order[0]].copy()
        elite = order[:max(2,n//5)]
        mean = .25*mean+.75*candidates[elite].mean(axis=0)
        scale = np.maximum([.035,.26,.25],scale*.75)
        records.append(dict(cases=[asdict(c) for c in base],public_points=public.tolist(),performance=performance,
                            state=memory.state()))
        (args.out/'numeric-training.json').write_text(json.dumps(records,separators=(',',':'))+'\n')
        selected = order[:6]
        portable = NumericMemory(memory.coefficients[selected],memory.scores[selected],memory.counts[selected])
        (args.out/'numeric-state.json').write_text(json.dumps(portable.state(),indent=2)+'\n')
        print(generation,public.tolist(),best.round(3).tolist(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roles',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--generations',type=int,default=5)
    parser.add_argument('--population',type=int,default=6)
    parser.add_argument('--points-per-match',type=int,default=1)
    parser.add_argument('--starts',type=int,default=8)
    parser.add_argument('--ticks',type=int,default=1080)
    parser.add_argument('--seed',type=int,default=6617352)
    train(parser.parse_args())
