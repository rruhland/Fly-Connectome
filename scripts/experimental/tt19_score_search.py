"""Learn a timed SO-101 stroke using only ordinary point scores on one stream."""
import argparse
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt19_so101 import SO101Arena


# Excursions are expressed in cumulative public command units. The player gets
# a public service turn and point clock. Its subsequent strokes are open loop.
def action_tape(parameters, steps=220):
    offset, stroke, tick, width = np.split(np.asarray(parameters), [5, 10, 11])
    start = int(np.round(tick[0]))
    duration = max(2, int(np.round(width[0])))
    tape = np.zeros((steps, 6))
    tape[:5, :5] = offset / 5
    tape[start:start + duration,:5] = stroke / duration
    tape[start + duration + 4:start + 2*duration + 4,:5] = -stroke / duration
    return np.clip(tape,-1.,1.)


def play(arena, seed, server, tape):
    arena.reset(seed=seed, server=server, render=False)
    for command in tape:
        _, reward, done, timeout, _ = arena.step(command, render=False)
        if done or timeout:
            return float(reward), arena.point_result()
    raise RuntimeError('incomplete point')


class ScoreSearch:
    """Cross-entropy search on fresh point seeds with noisy binary reward."""
    def __init__(self, role, rng, population=16):
        self.rng = rng
        self.population = population
        self.base_std = np.array([.65,.67,.67,.67,.65,1.2,1.7,1.7,1.7,1.,3.,1.2])
        self.mean = np.r_[np.zeros(10), 14. if role == 1 else 21., 5.]
        self.std = self.base_std.copy()
        self.bounds = np.r_[np.full(5,3.),np.full(5,4.5),40.,9.]
        self.pool = self.rewards = None
        self.epoch = 0
        self.draw()

    def draw(self):
        noise = self.rng.normal(size=(self.population,len(self.mean)))
        self.pool = np.clip(self.mean + self.std * noise, -self.bounds, self.bounds)
        self.pool[:,10:] = np.clip(self.pool[:,10:], [2.,2.], [40.,9.])
        self.pool[0] = self.mean
        self.rewards = np.zeros(len(self.pool))

    def finish(self):
        # Shrink only when positive observations distinguish candidates.
        if np.ptp(self.rewards) > 0:
            ranked = np.argsort(self.rewards + self.rng.uniform(0,.01,len(self.pool)))
            elite = self.pool[ranked[-max(3,len(self.pool)//4):]]
            self.mean = .25*self.mean + .75*elite.mean(axis=0)
            self.std = np.maximum(.16*self.base_std, .72*self.std + .28*elite.std(axis=0))
        else:
            self.std = np.minimum(1.65*self.base_std, self.std*1.15)
        self.epoch += 1
        self.draw()


def train(folder, seed, epochs, repeat, population):
    arena = SO101Arena(venue='hall', detail='preview', max_steps=210)
    rng = np.random.default_rng(seed)
    learners = {str(role):ScoreSearch(role, np.random.default_rng([seed,role+1]),population)
                for role in (-1,1)}
    history = []
    try:
        for epoch in range(epochs):
            marks = {}
            for role in (-1,1):
                search = learners[str(role)]
                data = []
                for _ in range(repeat):
                    for index in rng.permutation(population):
                        point = int(rng.integers(2**31))
                        reward, result = play(arena,point,role,action_tape(search.pool[index]))
                        search.rewards[index] += reward
                        data.append([point,int(index),int(reward),int(result['legal_serve']),result['legal_returns'],arena.steps])
                marks[str(role)] = dict(before=search.mean.tolist(),std=search.std.tolist(),
                    candidates=search.pool.tolist(),reward=search.rewards.tolist(),points=data)
                search.finish()
            history.append(marks)
            checkpoint=dict(seed=seed, epochs=epoch+1,population=population,repeat=repeat,
                            stroke={role:learner.mean.tolist() for role,learner in learners.items()},history=history)
            _write(folder/'score-search.json.gz',checkpoint)
            print(epoch+1,{role:round(np.mean(np.array(mark['points'])[:,2]>0),2) for role,mark in marks.items()},flush=True)
    finally:
        arena.close()


def scored(arena, program, seed):
    rng = np.random.default_rng(seed)
    score = ServiceScore(first_server=-1)
    rows = []
    while not score.finished:
        point = int(rng.integers(2**31))
        role = score.server
        reward,result = play(arena, point, role, action_tape(program[str(role)]))
        events = [dict(step=e['step'], kind=e['kind'], side=e.get('side')) for e in arena.events]
        rows.append(dict(seed=point, server=role, frames=arena.steps, result=result, events=events))
        score.record(result)
        if len(rows) >= 100:
            raise ValueError('game did not end')
    return dict(score=list(score.score), rows=rows, own_serves=sum(x['server']==1 for x in rows),
                legal_serves=sum(x['server']==1 and x['result']['legal_serve'] for x in rows),
                returns=sum(x['result']['legal_returns'] for x in rows),
                multiple=sum(x['result']['legal_returns']>=2 for x in rows))


def evaluate(folder, seeds):
    learned=_read(folder/'score-search.json.gz')
    arena=SO101Arena(venue='hall',detail='preview',max_steps=210)
    games={}
    try:
        for label,prog in [('idle',{role:[0.]*10+[20.,5.] for role in ('1','-1')}),('learned',learned['stroke'])]:
            games[label]=[scored(arena,prog,seed) for seed in seeds]
            print(label,[(g['score'],g['legal_serves'],g['returns'],g['multiple']) for g in games[label]],flush=True)
    finally:
        arena.close()
    _write(folder/'score-held.json.gz',dict(training_sha256=_hash(folder/'score-search.json.gz'), seeds=seeds, games=games))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--seed',type=int,default=1913415)
    p.add_argument('--epochs',type=int,default=13)
    p.add_argument('--repeat',type=int,default=2)
    p.add_argument('--population',type=int,default=18)
    p.add_argument('--evaluate',action='store_true')
    a=p.parse_args()
    a.folder.mkdir(parents=True,exist_ok=True)
    if not a.evaluate:
        train(a.folder,a.seed,a.epochs,a.repeat,a.population)
    evaluate(a.folder,[9130973,32000694,830371,6291321,8411661,3277159])
