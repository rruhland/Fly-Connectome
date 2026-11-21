"""Learn controlled serves and returns across successive physical feeder spreads."""
import argparse
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import _read,_write,_hash
from .tt8_service_arena import ServiceScore
from .tt19_consolidate import arena_and_eyes,setup,TrialBank
from .tt19_contextual import training_point
from .tt19_skill_course import install_feeder
from .tt19_stroke_trial import DONOR


def sources(folder,role,use_prior):
    if use_prior:
        rows=[]
        for mode in ('gantry','local'):
            path=folder/f'consolidated-{mode}-bank.json.gz'
            archive=_read(path)['banks'][str(role)]
            trials=np.array(archive['trials'])
            value=(trials[:,5]+2*(trials[:,7]>0.))/3
            quality=[np.mean(value[trials[:,0]==j]) for j in range(len(archive['programs']))]
            rows.extend(np.asarray(archive['programs'])[np.argsort(quality)[-8:]])
        return np.asarray(rows)
    return np.tile(np.r_[np.zeros(10),14 if role==1 else 21,4,0],(16,1))


class PracticeBank:
    def __init__(self,base):
        self.mean=np.asarray(base)
        self.force=self.mean[0]
    def choose(self,cue,learning):
        return self.force


def practice(folder,tag,mode,course,seed,epochs=8,base=None):
    arena,eyes,cam=arena_and_eyes(False)
    if course!='full':install_feeder(arena,course)
    law,slopes=setup(folder,'gantry' if mode!='local' else 'local')
    rng=np.random.default_rng(seed)
    archive={str(r):[] for r in (-1,1)}
    if base:
        with_prior=_read(base)
    try:
        for r in (-1,1):
            init=sources(folder,r,mode=='experienced') if not base else np.asarray(with_prior['champions'][str(r)])
            bank=PracticeBank(init)
            for epoch in range(epochs):
                sigma=max(.12,.8*.82**epoch)
                if epoch==0:
                    pop=init.copy()
                    if mode!='experienced' and not base:
                        pop+=rng.normal(0.,1.,pop.shape)*[.25,.55,.55,.6,.5,.7,1.8,1.8,2.2,1.,2.2,1.,4.]
                else:
                    parents=np.repeat(elite,4,axis=0)
                    pop=parents+rng.normal(size=parents.shape)*sigma*[.3,.57,.57,.67,.5,.6,1.8,1.8,1.9,.75,2.2,1.,4.]
                    pop[::4]=elite
                pop=np.clip(pop,[-3]*5+[-6]*5+[4,2,-22],[3]*5+[6]*5+[35,10,22])
                trials=[]
                for _ in range(4):
                    for j in rng.permutation(len(pop)):
                        bank.force=pop[j]
                        point=training_point(arena,eyes,cam,bank,int(rng.integers(2**31)),r,False,law,slopes[r])
                        trials.append([int(j),point['seed'],point['frames'],*point['cue'],point['merit'],
                                       point['reward'],point['result']['legal_returns'],int(point['result']['legal_serve'])])
                t=np.array(trials)
                utility=1.2*(t[:,6]>0)+.9*(t[:,5]>2.2)+.25*t[:,5]
                order=np.argsort([np.mean(utility[t[:,0]==j]) for j in range(len(pop))])[::-1]
                elite=pop[order[:4]]
                archive[str(r)].append(dict(programs=pop.tolist(),trials=trials,order=order.tolist()))
                print(tag,r,epoch,int((t[:,6]>0).sum()),int(t[:,7 if r<0 else 8].sum()),flush=True)
            # Keep distinct successful neighbors as a prior for later courses.
        champions={r:[ep['programs'][i] for ep in epochs[-4:] for i in ep['order'][:4]]
                   for r,epochs in archive.items()}
        _write(folder/f'feeder-{tag}-practice.json.gz',dict(seed=seed,course=course,mode=mode,archive=archive,
               parent=None if base is None else _hash(base),champions=champions))
    finally: arena.close()


def qualify(folder,tag,seed,repeat=12):
    path=folder/f'feeder-{tag}-practice.json.gz'
    prior=_read(path);course=prior['course'];mode=prior['mode']
    law,slopes=setup(folder,'gantry' if mode!='local' else 'local')
    arena,eyes,cam=arena_and_eyes(False)
    if course!='full':install_feeder(arena,course)
    rng=np.random.default_rng(seed);banks={}
    try:
        for r in (-1,1):
            pop=prior['champions'][str(r)];bank=PracticeBank(pop);trials=[]
            for _ in range(repeat):
                for j in rng.permutation(len(pop)):
                    bank.force=pop[j]
                    point=training_point(arena,eyes,cam,bank,int(rng.integers(2**31)),r,False,law,slopes[r])
                    # Selection uses measurements obtainable from RGB and score.
                    success=.60*(point['reward']>0)+.40*(point['merit']>2.2)
                    trials.append([int(j),point['seed'],point['frames'],*point['cue'],success,point['merit'],
                                   point['reward'],point['result']['legal_returns'],int(point['result']['legal_serve'])])
            banks[str(r)]=dict(programs=pop,trials=trials)
            print('qualified',tag,r,flush=True)
    finally: arena.close()
    _write(folder/f'consolidated-{tag}-bank.json.gz',dict(source_sha256=_hash(path),banks=banks,
           keep=len(prior['champions']['1']),repeats=repeat))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('tag')
    p.add_argument('--mode',choices=('experienced','gantry','local'),default='experienced')
    p.add_argument('--course',choices=('center','wide','full'),default='center')
    p.add_argument('--parent',type=Path);p.add_argument('--epochs',type=int,default=7)
    p.add_argument('--repeats',type=int,default=12);p.add_argument('--seed',type=int,default=85125171)
    a=p.parse_args();practice(a.folder,a.tag,a.mode,a.course,a.seed,a.epochs,a.parent)
    qualify(a.folder,a.tag,a.seed+1,a.repeats)
