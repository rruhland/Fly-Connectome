"""Repeat promising strokes on fresh incoming balls before trusting them."""
import argparse
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import _hash,_read,_write
from .tt8_service_arena import ServiceScore
from .tt19_contextual import training_point
from .tt19_course import guide
from .tt19_flight import FlightLaw
from .tt19_practice_camera import BallEyes, PracticeCamera
from .tt19_stroke_trial import DONOR
from .tt19_view_trial import SO101ViewArena


class TrialBank:
    def __init__(self, programs, trials=()):
        self.programs=np.asarray(programs)
        self.trials=list(trials)
        self.force=None

    def choose(self,cue,learning):
        if self.force is not None:
            return self.programs[self.force]
        rows=np.asarray(self.trials)
        index=rows[:,0].astype(int)
        q=rows[:,3:5]-cue
        # A cross-validated movement must work repeatedly. Pool the full
        # estimate with evidence from nearby incoming balls.
        local=np.exp(-.5*np.sum((q/[.042,.055])**2,axis=1))
        weight=.20+local
        wins=np.bincount(index,weights=weight*rows[:,5],minlength=len(self.programs))
        count=np.bincount(index,weights=weight,minlength=len(self.programs))
        rank=(wins+.65)/(count+2.)
        return self.programs[int(np.argmax(rank))]


def arena_and_eyes(native):
    arena=SO101ViewArena(venue='hall',width=240,height=180,view_mode='wrist+fixed' if native else 'fixed',
                        detail='preview',strict_endline=True,max_steps=190)
    return arena,BallEyes(arena.reset(seed=20017)),PracticeCamera(arena)


def setup(folder,mode):
    styles,_=guide() if mode=='gantry' else ({},[])
    law=FlightLaw.from_state(_read(folder/'flight-source.json.gz')['source_law']) if mode=='gantry' else FlightLaw()
    return law,{r:styles.get(r,{}).get('slope') for r in (-1,1)}


def practice(folder,mode,keep,repeats):
    source=folder/f'context-{mode}-practice.json.gz'
    prior=_read(source)
    law,slopes=setup(folder,mode)
    rng=np.random.default_rng(39107462)
    arena,eyes,cam=arena_and_eyes(False)
    banks={}
    try:
        for r in (-1,1):
            rows=sorted(prior['bank'][str(r)],key=lambda x:-x[2])
            programs=[row[1] for row in rows[:keep]]
            bank=TrialBank(programs)
            for n in range(repeats):
                for j in rng.permutation(len(programs)):
                    bank.force=int(j)
                    point=training_point(arena,eyes,cam,bank,int(rng.integers(2**31)),r,False,law,slopes[r])
                    # RGB far-half rebounds give a larger signal than a long
                    # shot ending in a rival error. Simulator events are audit only.
                    success=float(point['merit']>2.2)
                    bank.trials.append([int(j), point['seed'],point['frames'],*point['cue'],
                                        success,point['merit'],point['reward'],
                                        point['result']['legal_returns'],int(point['result']['legal_serve'])])
                if (n+1)%6==0: print('qualification',r,n+1,flush=True)
            banks[str(r)]=dict(programs=programs,trials=bank.trials)
    finally:
        arena.close()
    path=folder/f'consolidated-{mode}-bank.json.gz'
    _write(path,dict(source_sha256=_hash(source),banks=banks,keep=keep,repeats=repeats))


def evaluate(folder,mode,seeds,native,render_wrist=False):
    path=folder/f'consolidated-{mode}-bank.json.gz'; trained=_read(path)
    law,slopes=setup(folder,mode)
    banks={int(r):TrialBank(**record) for r,record in trained['banks'].items()}
    arena,eyes,cam=arena_and_eyes(native)
    games=[]
    try:
        for seed in seeds:
            score=ServiceScore(first_server=-1); rng=np.random.default_rng(seed); rows=[]
            while not score.finished and len(rows)<90:
                r=score.server
                row=training_point(arena,eyes,cam,banks[r],int(rng.integers(2**31)),r,False,law,slopes[r],native)
                rows.append(row);score.record(row['result'])
            g=dict(seed=seed,score=list(score.score),points=rows,finished=score.finished,
                   returns=sum(r['result']['legal_returns'] for r in rows),
                   multiple=sum(r['result']['legal_returns']>=2 for r in rows),
                   serves=sum(r['role']==1 and r['result']['legal_serve'] for r in rows))
            games.append(g)
            print('held',seed,score.score,g['returns'],g['multiple'],g['serves'],flush=True)
    finally:
        arena.close()
    _write(folder/f'consolidated-{mode}-{"native" if native else "fast"}.json.gz',
           dict(practice=_hash(path),strict_endline=True,native=native,games=games))
    checkpoint=_read(DONOR)
    checkpoint['tt19_consolidated']=dict(mode=mode,banks=trained['banks'],practice_sha256=_hash(path),
                                         donor_sha256=_hash(DONOR))
    _write(folder/f'consolidated-{mode}-player.json.gz',checkpoint)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path);p.add_argument('mode',choices=('local','gantry'))
    p.add_argument('--held',action='store_true');p.add_argument('--native',action='store_true')
    p.add_argument('--keep',type=int,default=28);p.add_argument('--repeat',type=int,default=18)
    p.add_argument('--seeds',type=int,nargs='+',default=[7920488,1193881,1474315,2918869,37620411,993144])
    a=p.parse_args()
    if not a.held: practice(a.folder,a.mode,a.keep,a.repeat)
    evaluate(a.folder,a.mode,a.seeds,a.native)
