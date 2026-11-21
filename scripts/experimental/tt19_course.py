"""Learn strokes from their camera-observed effect and ordinary point reward."""
import argparse
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import ROOT,_read,_write,_hash
from .tt8_service_arena import ServiceScore
from .tt19_practice_camera import BallEyes, PracticeCamera
from .tt19_score_search import ScoreSearch
from .tt19_stroke import gantry_style
from .tt19_stroke_trial import TAPES,DONOR
from .tt19_view_trial import SO101ViewArena


def guide():
    rows = [r for p in TAPES for r in _read(p)['points']]
    styles,examples = gantry_style(rows,1/30)
    return styles,examples


class Actor:
    def __init__(self, params, server, see):
        self.params = np.asarray(params)
        self.server = server
        self.see = see
        self.ball = []
        self.position = np.zeros(5)
        self.strokes = []
        self.center_x = 0.
        self.base = self.params[:5]
        self.speed = self.params[5:10]
        self.last_incoming = None
        self.phase = 'opening'
        self.contact = None
        self.cycle_start = 0

    def observe(self,tick,xyz):
        if xyz is None:
            return
        point = np.asarray(xyz)
        if self.ball:
            prev_tick,old=self.ball[-1]
            speed=(point-old)/(tick-prev_tick)
            if speed[1] < -.025 and self.last_incoming is None:
                self.last_incoming=tick
            if speed[1] > .023 and self.phase=='recovery' and self.contact is not None and tick>self.contact+3:
                self.phase='away'
            if self.phase=='away' and speed[1]<-.025 and point[1] > -.42:
                self.phase='incoming'
                self.cycle_start=tick
                self.last_incoming=tick
            if self.see and self.phase in ('opening','incoming'):
                if self.server<0 or self.phase=='incoming':
                    if speed[1]<-.028:
                        eta=(-.90-point[1])/speed[1]
                        if 1. <= eta <= 26.:
                            self.center_x=float(np.clip(point[0]+speed[0]*eta,-.21,.21))
                            if self.contact is None or self.phase=='incoming':
                                self.contact=tick+eta
                elif tick<12:
                    self.center_x=float(point[0])
        self.ball.append((tick,point))

    def action(self,tick):
        offset,start,duration = self.params[12],self.params[10],max(2,round(self.params[11]))
        if self.phase=='opening':
            onset = start
            if self.see and self.contact is not None and self.server<0:
                onset=self.contact-4 + start-21
        elif self.phase=='incoming' and self.contact is not None:
            onset=self.contact-4 + start-21
        else:
            onset=None
        if onset is not None and tick>=onset:
            self.strokes.append(tick)
            self.contact=tick+4
            self.phase='recovery'
        goal = self.base.copy()
        if self.see:
            goal[0] += offset*self.center_x
        if self.strokes:
            age=tick-self.strokes[-1]
            strength=min(1.,(age+1)/duration,max(0.,1.-(age-duration-3)/duration))
            goal+=strength*self.speed
        action = np.clip(goal-self.position,-.85,.85)
        self.position+=action
        return np.r_[action,0.]


def evaluation(ball,role,reward,style):
    if len(ball)<3:
        return float(reward)
    pos = np.asarray([r[1] for r in ball]); delta=np.diff(pos,axis=0)
    times=np.diff([r[0] for r in ball])
    # Detect the ball leaving the arm side from the optical depth reversal.
    source=np.flatnonzero((pos[:-1,1]<-.55) & (delta[:,1]/times>.008))
    if not len(source):
        return float(reward)
    i=source[0]
    segment=pos[i:]
    x,y,z = segment.T
    # Maximum progress along the length while inside the playable width.
    progress=np.max(y-y[0]-1.5*np.maximum(abs(x)-.25,0.)-1.4*np.maximum(z-.8,0.))
    value=.20+min(1.45,max(0.,progress))
    if style is not None:
        good=source[source<=i+2]
        launch=np.median(delta[good]/times[good,None],axis=0)
        desired=float(style['slope'])
        slope=np.clip(launch[2]/max(.008,launch[1]),-3,3)
        value+=.35*np.exp(-abs(slope-desired)/.55)
    return float(reward+.70*value)


def point(arena,eyes,camera,parameters,seed,server,*,native=False,style=None,see=True):
    arena.reset(seed=seed,server=server,render=False)
    actor=Actor(parameters,server,see)
    for tick in range(arena.max_steps):
        if tick%3==0:
            rgb=arena.render_cameras() if native else camera.capture()
            actor.observe(tick, eyes.position(rgb,not native))
        _,reward,done,timeout,_=arena.step(actor.action(tick),render=False)
        if done or timeout: break
    trace=[[t,b.tolist()] for t,b in actor.ball]
    return dict(seed=seed,server=server,frames=arena.steps,reward=float(reward),result=arena.point_result(),
                optical=trace,strokes=actor.strokes,
                credit=evaluation(actor.ball,server,reward,style))


def new_search(role,seed,pop):
    s=ScoreSearch(role,np.random.default_rng([seed,role+1]),pop)
    s.base_std=np.r_[ [.29,.48,.48,.55,.42, .68,2.2,2.2,2.5,.8, 2.2,.9], 4.0]
    s.mean=np.r_[np.zeros(10), 14. if role==1 else 21., 4.,0.]
    s.std=s.base_std.copy()
    s.bounds=np.r_[np.full(5,3.),np.full(5,6.),39.,10.,17.]
    def draw():
        values=s.rng.normal(0.,1.,(s.population,len(s.mean)))
        s.pool=np.clip(s.mean+s.std*values,-s.bounds,s.bounds)
        s.pool[:,10:12]=np.clip(s.pool[:,10:12],[3.,2.],[39.,10.])
        s.pool[0]=s.mean
        s.rewards=np.zeros(s.population)
    s.draw=draw
    draw()
    return s


def train(folder,mode,seed,epochs,pop,rep):
    styles,examples=guide() if mode=='gantry' else ({},[])
    arena=SO101ViewArena(venue='hall',width=240,height=180,view_mode='fixed',detail='preview',max_steps=180)
    camera=PracticeCamera(arena)
    eyes=BallEyes(arena.reset(seed=20017))
    rng=np.random.default_rng(seed)
    searches={role:new_search(role,seed,pop) for role in (-1,1)}
    history=[]
    try:
        for epoch in range(epochs):
            stats={}
            for role,s in searches.items():
                data=[]
                for _ in range(rep):
                    for j in rng.permutation(pop):
                        r=point(arena,eyes,camera,s.pool[j],int(rng.integers(2**31)),role,style=styles.get(role))
                        s.rewards[j]+=r['credit']
                        data.append([int(j),r['seed'],r['frames'],r['reward'],r['credit'],r['result']['legal_returns'],
                                     int(r['result']['legal_serve'])])
                stats[str(role)]=dict(candidates=s.pool.tolist(), trials=data)
                s.finish()
            history.append(stats)
            program={str(k):s.mean.tolist() for k,s in searches.items()}
            _write(folder/f'course-{mode}-practice.json.gz',dict(seed=seed,epochs=epoch+1,population=pop,
                   source_hashes=[_hash(p) for p in TAPES] if examples else [],examples=examples,
                   program=program,history=history))
            print(epoch+1,[(role,round(np.mean(np.array(r['trials'])[:,3]>0),2),
                            np.sum(np.array(r['trials'])[:,5])) for role,r in stats.items()],flush=True)
    finally:
        arena.close()


def held(folder,mode,seeds,native):
    path=folder/f'course-{mode}-practice.json.gz'
    prog=_read(path)['program']
    rig='wrist+fixed' if native else 'fixed'
    arena=SO101ViewArena(venue='hall',width=240,height=180,view_mode=rig,detail='preview',max_steps=180)
    camera=PracticeCamera(arena)
    eyes=BallEyes(arena.reset(seed=20017))
    games=[]
    try:
        for seed in seeds:
            rng=np.random.default_rng(seed)
            score=ServiceScore(first_server=-1)
            rows=[]
            while not score.finished and len(rows)<90:
                server=score.server
                row=point(arena,eyes,camera,prog[str(server)],int(rng.integers(2**31)),server,native=native)
                rows.append(row)
                score.record(row['result'])
            g=dict(seed=seed,score=list(score.score),points=rows,finished=score.finished,
                   returns=sum(r['result']['legal_returns'] for r in rows),
                   multiple=sum(r['result']['legal_returns']>=2 for r in rows),
                   serves=sum(r['server']==1 and r['result']['legal_serve'] for r in rows))
            games.append(g)
            print('held',seed,score.score,g['returns'],g['multiple'],g['serves'],flush=True)
    finally:
        arena.close()
    _write(folder/f'course-{mode}-{"native" if native else "fast"}.json.gz',
           dict(source=_hash(path),native=native,games=games))
    # Preserve the previous body addresses. The trained motor program is the
    # only additional state used when returning to this installation.
    checkpoint=_read(DONOR)
    checkpoint['tt19_course']=dict(mode=mode,program=prog,training_sha256=_hash(path),donor_sha256=_hash(DONOR))
    _write(folder/f'course-{mode}-player.json.gz',checkpoint)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('mode',choices=['gantry','local'])
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--seed',type=int,default=2051039)
    parser.add_argument('--pop',type=int,default=18)
    parser.add_argument('--repeat',type=int,default=2)
    parser.add_argument('--held',action='store_true')
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+',default=[9130973,32000694,830371,6291321,8411661,3277159])
    a=parser.parse_args(); a.folder.mkdir(parents=True,exist_ok=True)
    if not a.held: train(a.folder,a.mode,a.seed,a.epochs,a.pop,a.repeat)
    held(a.folder,a.mode,a.seeds,a.native)
