"""Retrieve and refine strokes from nearby optical approaches on one arm."""
import argparse
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import _read,_write,_hash
from .tt8_service_arena import ServiceScore
from .tt19_course import Actor, guide
from .tt19_flight import FlightLaw, optical_start
from .tt19_practice_camera import BallEyes, PracticeCamera
from .tt19_stroke_trial import DONOR
from .tt19_view_trial import SO101ViewArena


def context(sightings,server):
    if not sightings:
        return [0.,.32]
    _,p=sightings[-1]
    if len(sightings)<2:
        return [float(p[0]),float(p[2])]
    _,a=sightings[-2]
    if server>0:
        return [float(p[0]),float(a[2])]
    dx,dy,_=p-a
    travel=np.clip((-.90-p[1])/min(-.02,dy),0,12)
    return [float(np.clip(p[0]+dx*travel,-.29,.29)),float(a[2])]


def optical_placement(sightings, server, reward, law, learned_slope):
    if len(sightings)<3:
        return reward*.20
    ticks,points=zip(*sightings)
    points=np.asarray(points); ticks=np.asarray(ticks)
    velocity=np.diff(points,axis=0)/np.diff(ticks)[:,None]
    indices=np.flatnonzero((points[:-1,1]<-.50)&(velocity[:,1]>.008))
    if not len(indices):
        return .20*reward
    i=int(indices[0])
    p=points[i+1]; v=velocity[i]
    # A real forward stroke must be seen across at least one subsequent
    # capture. Estimate the landing through the net from that outgoing flight.
    after=points[i:]
    progress=np.clip(np.max(after[:,1]-points[i,1]-.9*abs(after[:,0])),0,1.8)
    value=.3+.3*progress
    if i+2<len(points) and velocity[i+1,1]>.010:
        p=points[i+2]; v=(points[i+2]-points[i])/(ticks[i+2]-ticks[i])
        dt=(ticks[i+2]-ticks[i])/2
        v=v-np.array([0,0,dt*law.gravity])
    predicted,_=law.trajectory(p,v,42,bounds=(.424,.80))
    positive=np.flatnonzero(predicted[:,1]>=0.)
    z=predicted[:,2]
    bounces=np.flatnonzero((z[1:-1]<=z[:-2])&(z[1:-1]<z[2:])&(z[1:-1]<.11))+1
    y_cross=predicted[positive[0],2] if len(positive) else -.3
    if len(bounces):
        landing=predicted[bounces[0]]
        first_good=(server<0 and landing[1]>.015) or (server>0 and -.8<landing[1]<-.02)
        value+=.32*float(first_good)
    if len(positive):
        value+=.34*np.exp(-abs(y_cross-.24)/.23)
    # Detect an actual opponent-side bounce optically, including curved
    # paths. Early rival volleys provide little placement credit.
    actual=np.flatnonzero((after[1:-1,2]<after[:-2,2])&(after[1:-1,2]<after[2:,2])&
                          (after[1:-1,2]<.15)&(after[1:-1,1]>.03)&(abs(after[1:-1,0])<.45))
    value+=1.7*bool(len(actual))
    # Residual initial source knowledge is learned from past positive strokes.
    if learned_slope is not None:
        slope=v[2]/max(.008,v[1])
        value+=.30*np.exp(-abs(slope-learned_slope)/.6)
    return float(value+.20*reward)


class StrokeMemory:
    def __init__(self, seed, role, rows=None):
        self.rng=np.random.default_rng(seed)
        self.role=role
        self.rows=[] if rows is None else list(rows)
        self.start=np.r_[np.zeros(10),14. if role>0 else 21.,4.,0.]
        self.scale=np.array([.36,.61,.61,.65,.55, .7,1.9,1.9,2.1,.8, 2.7,1.,4.5])

    def choose(self,cue,learning):
        n=len(self.rows)
        if n<8:
            mean=self.start
        else:
            cues=np.array([r[0] for r in self.rows])
            diff=(cues-cue)/([.043,.065] if self.role<0 else [.028,.05])
            d=np.sqrt((diff**2).sum(axis=1))
            nearest=np.argsort(d)[:min(n,max(12,int(np.sqrt(n)*1.8)))]
            scores=np.array([self.rows[j][2] for j in nearest])
            ranks=scores-.23*np.minimum(4.,d[nearest])
            idx=nearest[np.argmax(ranks)]
            mean=np.asarray(self.rows[idx][1])
            if learning and self.rng.random()<.16:
                mean=self.start
        if not learning:
            return mean
        radius=self.rng.choice([.18,.38,.7,1.25],p=[.2,.35,.3,.15]) if n>24 else 1.
        response=mean+self.rng.normal(size=13)*self.scale*radius
        return np.clip(response,[-3]*5+[-6]*5+[4,2,-22],[3]*5+[6]*5+[35,10,22])


def training_point(arena,eyes,camera,bank,seed,role,learn,law,slope,native=False):
    arena.reset(seed=seed,server=role,render=False)
    prefix=[]
    actor=None
    for tick in range(arena.max_steps):
        if tick%2==0:
            rgb=arena.render_cameras() if native else camera.capture()
            xyz=eyes.position(rgb,not native)
            if xyz is not None and actor is None:
                prefix.append((tick,xyz))
            if tick>=4 and actor is None:
                cue=context(prefix,role)
                program=bank.choose(cue,learn)
                actor=Actor(program,role,True)
                for t,p in prefix: actor.observe(t,p)
            elif actor is not None:
                actor.observe(tick,xyz)
        action=actor.action(tick) if actor is not None else np.zeros(6)
        _,reward,done,timeout,_=arena.step(action,render=False)
        if done or timeout: break
    merit=optical_placement(actor.ball,role,reward,law,slope)
    if learn:
        bank.rows.append([cue,np.asarray(program).tolist(),merit])
    return dict(seed=seed,frames=arena.steps,role=role,cue=cue,program=np.asarray(program).tolist(),
                optical=[[t,p.tolist()] for t,p in actor.ball],merit=merit,reward=float(reward),
                result=arena.point_result())


def launch(folder,mode,points,seeds,native,budgets,seed):
    path=folder/f'context-{mode}-practice.json.gz'
    arena=SO101ViewArena(venue='hall',width=240,height=180,view_mode='wrist+fixed' if native else 'fixed',
                        detail='preview',max_steps=190)
    camera=PracticeCamera(arena)
    eyes=BallEyes(arena.reset(seed=20017))
    styles,_=guide() if mode=='gantry' else ({},[])
    law=FlightLaw.from_state(_read(folder/'flight-source.json.gz')['source_law']) if mode=='gantry' else FlightLaw()
    banks={r:StrokeMemory(seed+(r+1),r) for r in (-1,1)}
    try:
        if points:
            rng=np.random.default_rng(seed)
            rows=[]
            for n in range(points):
                role=1 if n%2 else -1
                row=training_point(arena,eyes,camera,banks[role],int(rng.integers(2**31)),role,True,law,
                                   styles.get(role,{}).get('slope'))
                rows.append([row[k] for k in ('seed','role','frames','reward','result','cue','merit')])
                if (n+1)%100==0 or n==points-1:
                    print('practice',n+1,[(r,sum(x[4]['legal_returns'] for x in rows[-100:] if x[1]==r),
                           sum(x[4]['legal_serve'] for x in rows[-100:] if x[1]==r)) for r in (-1,1)],flush=True)
                    _write(path,dict(seed=seed,rows=rows,bank={str(r):b.rows for r,b in banks.items()}))
        stored=_read(path)
        for budget in budgets:
            for r in banks: banks[r]=StrokeMemory(seed+(r+1),r,stored['bank'][str(r)][:budget//2])
            games=[]
            for test in seeds:
                rng=np.random.default_rng(test)
                score=ServiceScore(first_server=-1)
                rows=[]
                while not score.finished and len(rows)<90:
                    role=score.server
                    row=training_point(arena,eyes,camera,banks[role],int(rng.integers(2**31)),role,False,
                                       law,styles.get(role,{}).get('slope'),native)
                    rows.append(row);score.record(row['result'])
                g=dict(seed=test,score=list(score.score),points=rows,finished=score.finished,
                       returns=sum(r['result']['legal_returns'] for r in rows),
                       multiple=sum(r['result']['legal_returns']>=2 for r in rows),
                       serves=sum(r['role']==1 and r['result']['legal_serve'] for r in rows))
                print('test',budget,test,score.score,g['returns'],g['serves'],g['multiple'],flush=True)
                games.append(g)
            _write(folder/f'context-{mode}-{budget}-{"native" if native else "fast"}.json.gz',
                   dict(source=_hash(path),native=native,budget=budget,games=games))
    finally:
        arena.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('mode',choices=('gantry','local'))
    p.add_argument('--points',type=int,default=1000)
    p.add_argument('--seed',type=int,default=424116959)
    p.add_argument('--budgets',type=int,nargs='+',default=[24,120,1000])
    p.add_argument('--seeds',type=int,nargs='+',default=[129043,97732032,15530926,409102,8299737,127720])
    p.add_argument('--native',action='store_true')
    a=p.parse_args()
    launch(a.folder,a.mode,a.points,a.seeds,a.native,a.budgets,a.seed)
