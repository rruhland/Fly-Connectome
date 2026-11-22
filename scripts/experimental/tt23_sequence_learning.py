"""Transfer a native duration prior and adapt on single, fully played SO-101 points."""

import argparse

import numpy as np
from scipy.special import betaincinv
from scipy.stats import rankdata

from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_pong import ROLE_MODEL, cases, play
from .tt23_optical_contact import OUT
from .tt23_sequence import MODES

BANK = OUT/'tt23_sequence.json.gz'
SOURCE = OUT/'sequence-learning-native.json.gz'
CURVE = OUT/'sequence-learning.json.gz'
BUDGETS = (0,2,4,8,12)
CAP = 300


def fit_source():
    import torch
    torch.set_num_threads(1)
    setup = _read(BANK)
    candidates = {k:[0.,*setup['gains'][k][:2]] for k in MODES}
    runs = play(cases(18,2090000000), candidates, RoleModel(**_read(ROLE_MODEL)), 54)
    means = [np.mean([r['ticks'] for r in runs[k]]) for k in MODES]
    result = dict(source='native Pong', options=candidates, seeds=[r['case']['serve'] for r in runs[MODES[0]]],
                  rows=runs, mean_ticks=means, rank=rankdata(means).tolist(),role_sha256=_hash(ROLE_MODEL))
    _write(SOURCE,result)
    print(candidates,means,flush=True)


def learning(bank,source):
    groups = bank['bank_seeds']
    names = list(groups)
    if len(names)!=2 or len(bank['cases'])!=24 or not bank['verified']:
        raise ValueError('expected two verified complete twelve-point banks')
    ranks = np.array(source['rank'])
    means = .25 + .25*(ranks-1)
    prior={'pong':means,'cold':np.full(3,.5),'reversed':1.-means}
    curves={}
    for train_group,held_group in (names,names[::-1]):
        train=[case for case in bank['cases'] if case['cohort']==train_group]
        held=[case for case in bank['cases'] if case['cohort']==held_group]
        if len(train)!=12 or len(held)!=12 or {c['seed'] for c in train}&{c['seed'] for c in held}:
            raise ValueError('expected two disjoint completed cohorts')
        score = np.array([[[r['longest']>=10,r['longest']>=20,r['outcome']['legal_returns']]
                           for option in MODES for r in [c['results'][option]]] for c in held],dtype=float)
        rng=np.random.default_rng(184882183)
        sampler=rng.uniform(1e-8,1-1e-8,size=(len(train),512,len(MODES)))
        ties=rng.uniform(0,1e-8,size=(13,512,len(MODES)))
        traces={}
        for mode,start in prior.items():
            a=np.tile(2*start,(512,1))
            b=2-a
            elapsed=np.zeros(512,dtype=int)
            first=np.zeros(512,dtype=int)
            row_idx=np.arange(512)
            checks=[]
            for i in range(len(train)+1):
                if i in BUDGETS:
                    greedy=np.argmax(a/(a+b)+ties[i],axis=1)
                    aggregate=score.sum(axis=0)[greedy]
                    checks.append(dict(practice=i,held_ten=aggregate[:,0].mean(),held_twenty=aggregate[:,1].mean(),
                                       held_own_legal=aggregate[:,2].mean(),options=np.bincount(greedy,minlength=3).tolist()))
                if i==len(train):
                    break
                chosen=np.argmax(betaincinv(a,b,sampler[i]),axis=1)
                results=[train[i]['results'][k] for k in MODES]
                length=np.array([len(r['actions']) for r in results])[chosen]
                tenth=np.array([next((e['step'] for j,e in enumerate(z for z in r['events'] if z['kind']=='legal_return') if j==9),0)
                                for r in results])[chosen]
                just=(first==0)&(tenth>0)
                first[just]=elapsed[just]+tenth[just]
                elapsed+=length
                # Each simulated learner receives exactly one clock interval.
                utility=np.minimum(length/CAP,1.)
                a[row_idx,chosen]+=utility
                b[row_idx,chosen]+=1-utility
            traces[mode]=dict(held=checks,training_ten_probability=(first>0).mean(),
                              frames_to_first_ten_if_reached=None if not np.any(first) else float(np.mean(first[first>0])),
                              mean_practice_frames=elapsed.mean())
        curves[train_group]=dict(held_cohort=held_group,traces=traces)
    return curves


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',action='store_true')
    a=p.parse_args()
    if a.source:
        fit_source()
    else:
        results=learning(_read(BANK),_read(SOURCE))
        _write(CURVE,dict(bank_sha256=_hash(BANK),source_sha256=_hash(SOURCE),curves=results))
        print(results)
