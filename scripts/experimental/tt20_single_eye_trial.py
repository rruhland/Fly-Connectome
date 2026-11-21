"""Matched, single-stream reward learning and bidirectional camera transfer."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _read, _write, _hash
from .tt20_curriculum import retarget_programs
from .tt20_fixture import TT20Arena
from .tt20_single_eye import Memory, image_ball, cue, motor_library, SAMPLES

OUT = ROOT/'artifacts/tt20'
OLD = ROOT/'artifacts/tt19'
EYES = ('wrist', 'left')
TRAIN = 10971346
ADAPT = 995371
HELD = 23137016
REPEAT = 45201139
ARENA = dict(mount='clamp', wrist_optics='outboard', marked=False, detail='preview', venue='hall',
             width=320, height=240, strict_endline=True, max_steps=190)


def schedule(seed, pairs):
    gen = np.random.default_rng(seed)
    return [(int(gen.integers(2**31)), role) for _ in range(pairs) for role in (-1,1)]


class Trial:
    def __init__(self, eye):
        self.eye = eye
        self.arena = TT20Arena(**ARENA, view_mode=eye)
        self.programs = {r:retarget_programs(motor_library(OLD,r)) for r in SAMPLES}

    def memories(self, state=None):
        return {r:Memory(p, state[str(r)] if state else None) for r,p in self.programs.items()}

    def point(self, seed, role, memory, rng, learn, *, blind=False):
        arena = self.arena
        arena.reset(seed=seed,server=role,render=False)
        samples=[]
        digest=sha256()
        silence=np.zeros(6)
        for tick in range(SAMPLES[role][-1]+1):
            if tick in SAMPLES[role]:
                images = arena.render_cameras()
                if tuple(images) != (self.eye,):
                    raise ValueError('A second camera entered a single-view trial')
                rgb = images[self.eye]
                digest.update(rgb)
                samples.append([tick,image_ball(rgb)])
            if tick < SAMPLES[role][-1]:
                arena.step(silence,render=False)
        seen = cue(samples)
        x = memory.features(self.eye, seen, remember=True)
        if blind:
            x = np.zeros_like(x)
        index = memory.choose(x, explore=learn, rng=rng)
        actions = memory.commands(index,x,role,arena.max_steps)
        for tick in range(SAMPLES[role][-1],arena.max_steps):
            _, reward, done, timeout, _ = arena.step(actions[tick],render=False)
            if done or timeout:
                break
        result = arena.point_result()
        if learn:
            memory.record(x,index,reward)
        return dict(seed=seed, role=role, samples=samples, cue=x.tolist(), selection=index,
                    digest=digest.hexdigest(), action=sha256(actions[:tick+1].tobytes()).hexdigest(),
                    elapsed=tick+1, outcome=result, reward=reward)


def state_dict(memories):
    return {str(r):m.state() for r,m in memories.items()}


def summary(rows):
    receive=[r for r in rows if r['role']<0]; serve=[r for r in rows if r['role']>0]
    return dict(first_returns=sum(r['outcome']['legal_returns']>0 for r in receive),
                points_won=sum(r['reward']>0 for r in rows),
                legal_serves=sum(r['outcome']['legal_serve'] for r in serve),
                two_return=sum(r['outcome']['legal_returns']>=2 for r in rows),
                ball_seen=sum(any(p is not None for _,p in row['samples']) for row in rows),
                attempts=len(rows), frames=sum(r['elapsed'] for r in rows))


def train(bench, count):
    memories=bench.memories()
    rng=np.random.default_rng(3241761)
    rows=[]
    for seed, role in schedule(TRAIN,count):
        rows.append(bench.point(seed,role,memories[role],rng,True))
    path=OUT/f'one-eye-{bench.eye}-practice.json.gz'
    _write(path,dict(arena=ARENA,camera=bench.eye,practice=rows,summary=summary(rows),
                     state=state_dict(memories),programs={str(k):p.tolist() for k,p in bench.programs.items()}))
    print(bench.eye,'practice',summary(rows),flush=True)


def evaluate(bench,count,budget,fresh):
    for condition in ('cold', 'same', 'other','other-blind','other-shuffled'):
        source = bench.eye if condition=='same' else next(e for e in EYES if e!=bench.eye)
        path = OUT/f'one-eye-{source}-practice.json.gz'
        state=None if condition=='cold' else _read(path)['state']
        memories=bench.memories(state)
        if condition=='other-shuffled':
            rng=np.random.default_rng(7724831)
            for mem in memories.values():
                labels=rng.permutation([r[3] for r in mem.rows])
                for row,v in zip(mem.rows,labels,strict=True):
                    row[3]=float(v)
        rng=np.random.default_rng(81741115)
        practice=[]
        for seed,role in schedule(ADAPT,budget):
            practice.append(bench.point(seed,role,memories[role],rng,True))
        stored = state_dict(memories)
        rows=[]
        for seed,role in schedule(fresh,count):
            rows.append(bench.point(seed,role,memories[role],None,False,blind=condition=='other-blind'))
        name=f'one-eye-{bench.eye}-{condition}-{budget}-{fresh}'
        report=dict(arena=ARENA, eye=bench.eye, condition=condition, source=None if state is None else source,
                    source_sha256=None if state is None else _hash(path), local_practice=practice,
                    source_points=0 if state is None else sum(len(m['rows']) for m in state.values()),
                    target_points=2*budget, held=rows,summary=summary(rows), state_after_practice=stored)
        _write(OUT/(name+'.json.gz'),report)
        print(name,report['summary'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eye',choices=EYES)
    parser.add_argument('--train',type=int,metavar='POINTS_PER_ROLE')
    parser.add_argument('--held',type=int,default=24)
    parser.add_argument('--adapt',type=int,default=0)
    parser.add_argument('--seed',type=int,default=HELD)
    a=parser.parse_args()
    bench=Trial(a.eye)
    try:
        if a.train is not None:train(bench,a.train)
        else:evaluate(bench,a.held,a.adapt,a.seed)
    finally:bench.arena.close()
