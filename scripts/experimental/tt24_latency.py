"""Learn an optical clock and recover usable state from late wrist RGB."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _ellipse
from .tt24_public_body import load_public
from .tt24_wrist_egomotion import (CALIBRATION, HEIGHT, MAP, WIDTH, camera_pose,
                                 discover_edges, reprojected_lines)
from .tt4_transfer_trial import _hash, _read, _write

OUT = Path('artifacts/tt24')
SOURCE = OUT/'wrist-egomotion-trial.json.gz'
PUPIL = OUT/'bounce-ident-pupil.npz'
MODEL = OUT/'latency-model.npz'
RECEIPT = OUT/'latency-fit.json.gz'
FIT_CLIPS = 6
FIT_TICKS = 180
MAX_AGE = 5
DR = 60.
TRAIN_MODES = (.5,2.,8.)


def edge_features(lines):
    """Encode three frequently visible natural scene lines in pixel units."""
    return np.asarray(lines)[..., :3, (0,2)] * np.asarray((64.,1.))


def context(q, center):
    z=(q-center)/.18
    return np.concatenate((np.ones((*q.shape[:-1],1)),z,z*z),axis=-1)


def ridge(x,y,amount):
    return np.linalg.solve(x.T@x+amount*np.eye(x.shape[1]),x.T@y)


class SceneClock:
    """Track a hidden image age by matching wrist edges to encoder history."""

    def __init__(self,geometry,theta,lines,center,offset,scale):
        self.geometry,self.theta,self.lines=geometry,theta,lines
        self.center,self.offset,self.scale=center,offset,scale
        self.edges=deque(maxlen=MAX_AGE+1)
        self.last_observed=None
        self.previous=None
        self.last_frame=None
        initial=np.full(MAX_AGE+1,.018)
        self.transition=np.log(initial[None,:]+.892*np.eye(MAX_AGE+1))
        self.posterior=np.full(MAX_AGE+1,-100.)
        self.posterior[0]=0.

    def predict(self,q):
        p,r=camera_pose(q,self.geometry,self.theta)
        geometric=edge_features(reprojected_lines(self.lines,p,r))
        adjustment=(context(q,self.center)@self.offset).reshape(-1,3,2)
        return geometric+adjustment

    def read(self,obs):
        rgb=obs['rgb']
        if self.last_frame is None or not np.array_equal(rgb,self.last_frame):
            observed=edge_features(discover_edges(rgb))
        else:
            observed=self.last_observed
        self.edges.appendleft(self.predict(np.asarray(obs['joints'])[None,:])[0])
        count=len(self.edges)
        estimated=np.asarray(self.edges)
        residual=np.nan_to_num((observed-estimated)/self.scale,nan=0.)
        observed_loss=-.60*np.sum(np.log1p(residual**2/3.),axis=(1,2))
        probabilities=self.posterior[:,None]+self.transition[:,:count]
        if self.previous is not None:
            differential=observed[None,None]-self.last_observed[None,None]
            residual=np.nan_to_num((differential-estimated[None]+self.previous[:,None])/(1.45*self.scale),nan=0.)
            pair_loss=-.45*np.sum(np.log1p(residual**2/3.),axis=(2,3))
            probabilities[:len(self.previous)]+=pair_loss
        score=logsumexp(probabilities,axis=0)+observed_loss
        self.posterior[:]=-100.
        self.posterior[:count]=score-logsumexp(score)
        self.previous=estimated
        self.last_observed=observed
        self.last_frame=rgb
        return int(np.argmax(self.posterior))


def forecasts_features(position,velocity,body):
    v=np.clip(velocity/2.3,-2.,2.)
    r=np.clip((position-body)/np.array((.075,.075,.19)),-2.,2.)
    return np.r_[1.,r,v,r*v,v*np.abs(v)]


def training_replay(path,checkpoint):
    """Regenerate old source views. Command tapes do not enter either fit."""
    archive=_read(path)
    indicated=archive['conditions']['stock_original']
    chosen=[r for r in indicated if len(r['actions'])>=100][:FIT_CLIPS+2]
    if len(chosen)!=FIT_CLIPS+2:
        raise ValueError('need eight previously collected source motion tapes with adequate duration')
    arena=BounceArena(view='wrist',width=WIDTH,height=HEIGHT,max_steps=600)
    gathered=[]
    try:
        for row in chosen:
            agent=WristPupil.load(checkpoint)
            obs=arena.reset(seed=row['seed'],split='held',regime='standard')
            samples=[]
            for command in row['actions'][:FIT_TICKS]:
                image=obs['rgb']
                state=agent.sensor.observe(obs)
                samples.append((obs['joints'].copy(),edge_features(discover_edges(image)),
                                state.copy(),_ellipse(image) is not None))
                obs,_,_=arena.step(command)
            entry={key:np.asarray(value) for key,value in
                   zip(('q','edges','state','visible'),zip(*samples,strict=True),strict=True)}
            gathered.append(entry)
            print('old stock RGB replay',row['seed'],len(samples),flush=True)
    finally:
        arena.close()
    return [r['seed'] for r in chosen],gathered


def train_forecast(rows,age):
    X,Y=[],[]
    for row in rows:
        state=row['state']
        visible=row['visible']
        for t in range(3,len(state)-age):
            if not np.all(visible[t-3:t+age+1]):
                continue
            pos,vel=state[t,:3],state[t,3:6]
            future=state[t+age]
            X.append(forecasts_features(pos,vel,state[t,6:9]))
            delta_pos=(future[:3]-pos)*DR/age
            Y.append(np.r_[delta_pos-vel,future[3:6]-vel])
    return np.asarray(X),np.asarray(Y)


def fit(args):
    geometry=load_public(args.sensor).decoder
    with np.load(args.scene,allow_pickle=False) as scene:
        theta,lines=scene['source_theta'],scene['points']
    seeds,rows=training_replay(args.source,args.pupil)
    fit_groups,holdout=rows[:FIT_CLIPS],rows[FIT_CLIPS:]
    q=np.vstack([r['q'] for r in fit_groups])
    edge=np.vstack([r['edges'] for r in fit_groups]).reshape(-1,6)
    center=q.mean(axis=0)
    p,r=camera_pose(q,geometry,theta)
    residual=edge-edge_features(reprojected_lines(lines,p,r)).reshape(-1,6)
    valid=np.isfinite(residual)
    X=context(q,center)
    offset=np.column_stack([ridge(X[mask],value[mask],.9)
                            for mask,value in zip(valid.T,residual.T,strict=True)])
    gap=residual-X@offset
    scale=np.maximum(.17,1.4826*np.nanmedian(np.abs(gap-np.nanmedian(gap,axis=0)),axis=0)).reshape(3,2)
    model={}
    summary={}
    for age in range(1,MAX_AGE+1):
        A,B=train_forecast(fit_groups,age)
        C,D=train_forecast(holdout,age)
        fits=[ridge(A,B,penalty) for penalty in TRAIN_MODES]
        # Select fit regularization on whole old gameplay clips. Target performance is unseen.
        errors=[np.mean((C@w-D)**2) for w in fits]
        chosen=int(np.argmin(errors))
        model[f'h{age}']=fits[chosen]
        summary[str(age)]=dict(train_samples=len(B),validation_samples=len(D),ridge=TRAIN_MODES[chosen],
                               held_displacement_velocity_rmse=float(np.sqrt(errors[chosen])),
                               held_inertial_rmse=float(np.sqrt(np.mean(D**2))))
    np.savez_compressed(args.model,center=center,offset=offset,scale=scale,**model)
    doc=dict(schema='tt24-late-wrist-self-supervision-1',source_sha256=_hash(args.source),
             scene_sha256=_hash(args.scene),model_sha256=_hash(args.model),
             calibration_sha256=_hash(args.sensor),motor_sha256=_hash(args.pupil),
             fit_seeds=seeds[:FIT_CLIPS],validation_seeds=seeds[FIT_CLIPS:],
             fit_frames=sum(len(r['q']) for r in fit_groups),
             selection_frames=sum(len(r['q']) for r in holdout),
             labels='future optical positions and derived wrist velocity on old stock play',
             used_fields=['delivered wrist RGB','joint encoders'], scale=scale.tolist(),
             forecast=summary)
    _write(args.receipt,doc)
    print(doc,flush=True)


class LateOptics:
    """Decode images at their estimated capture pose and forecast to the current tick."""

    def __init__(self,sensor,clock,model):
        self.sensor=sensor
        self.clock=clock
        self.model=model
        self.encoders=deque(maxlen=MAX_AGE+1)
        self.points={}
        self.tick=-1
        self.latest=-1
        self.home=None
        self.velocity=np.zeros(3)
        self.age=0

    def older(self,tick):
        return self.points.get(tick,self.points[self.latest]+(tick-self.latest)*self.velocity/DR)

    def observe(self,obs,true_age=None):
        self.tick+=1
        q=np.asarray(obs['joints'])
        self.encoders.appendleft(q.copy())
        body=self.sensor.decoder.body(q)
        if self.home is None:
            self.home=body[12:15].copy()
        self.age=self.clock.read(obs) if true_age is None else true_age
        seen_at=self.tick-self.age
        ellipse=_ellipse(obs['rgb'])
        if seen_at>=self.latest:
            capture=self.sensor.decoder.body(self.encoders[self.age])
            point=self.sensor.decoder.ball(ellipse,capture)
            if point is None:
                point=self.home+self.sensor.decoder.default if self.latest<0 else self.older(seen_at)
            old_latest=self.latest
            self.latest=seen_at
            self.points[seen_at]=point
            past=[]
            for back in range(1,4):
                second=seen_at-back
                if second<0:
                    earlier=self.points[min(self.points)]
                elif second not in self.points and old_latest>=0 and second>old_latest:
                    alpha=(second-old_latest)/max(1,seen_at-old_latest)
                    earlier=(1-alpha)*self.points[old_latest]+alpha*point
                else:
                    earlier=self.older(second)
                past.append((point-earlier)*(DR/back))
            self.velocity=self.sensor.motion.predict(np.r_[point-self.home,*past])
            self.points={k:p for k,p in self.points.items() if k>=seen_at-8}
        point=self.points[self.latest]
        back=self.tick-self.latest
        velocity=self.velocity.copy()
        if back:
            offset=self.model[min(back,MAX_AGE)]
            features=forecasts_features(point,velocity,body[12:15])
            shift=features@offset
            moving=np.clip(velocity+.76*shift[:3],[-2.,-2.,-5.],[2.,2.,5.])
            point=point+(back/DR)*moving
            velocity=velocity+.76*shift[3:]
        return np.r_[point-self.home,velocity,body[12:15]-self.home,body[15:17]]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--pupil',type=Path,default=PUPIL)
    parser.add_argument('--scene',type=Path,default=MAP)
    parser.add_argument('--sensor',type=Path,default=CALIBRATION)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--receipt',type=Path,default=RECEIPT)
    fit(parser.parse_args())
