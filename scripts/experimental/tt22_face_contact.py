"""Learn whether a planned racket contact is robust from the arm's own scored strokes."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

from .tt4_arena import BALL_MASS,BALL_INERTIA,AIR_DRAG,MAGNUS_COEFFICIENT,SPIN_DECAY,mujoco
from .tt4_transfer_trial import ROOT,_read,_write
from .tt19_flight import FlightLaw,optical_start
from .tt19_model_rollout import ORIGIN,observed_ball
from .tt19_shoot import HORIZON
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import Station,replay

DIR=ROOT/'artifacts/tt22'


class ContactProbe:
    def __init__(self,sim):
        self.sim=sim
        self.data=mujoco.MjData(sim.model)
        self.b=sim.model.geom('ball').id
        self.r=sim.model.geom('learner_racket').id

    def measure(self,obs,tape):
        sim=self.sim
        model,d=sim.model,self.data
        mujoco.mj_copyData(d,model,sim.twin.data)
        pos,vel=sim.arena._ball_joint,sim.arena._ball_dof
        d.qpos[pos:pos+3]=obs.center+ORIGIN;d.qpos[pos+3:pos+7]=[1.,0.,0.,0.]
        d.qvel[vel:vel+3]=obs.velocity;d.qvel[vel+3:vel+6]=obs.spin
        mujoco.mj_forward(model,d)
        spin=np.empty(3)
        for tick,command in enumerate(tape):
            inc=sim.increments*command
            for j in range(sim.timestep):
                d.ctrl[sim.axes]=np.clip(d.ctrl[sim.axes]+inc,sim.low,sim.high)
                v=d.qvel[vel:vel+3]
                mujoco.mju_rotVecQuat(spin,d.qvel[vel+3:vel+6],d.qpos[pos+3:pos+7])
                d.xfrc_applied[sim.arena._ball_body,:3]=BALL_MASS*(-AIR_DRAG*np.linalg.norm(v)*v+MAGNUS_COEFFICIENT*np.cross(spin,v))
                d.xfrc_applied[sim.arena._ball_body,3:]=-BALL_INERTIA*SPIN_DECAY*spin
                before=v.copy()
                mujoco.mj_step(model,d)
                for c in d.contact:
                    if c.efc_address>=0 and {self.b,self.r}==set(c.geom):
                        R=d.geom_xmat[self.r].reshape(3,3)
                        local=(c.pos-d.geom_xpos[self.r])@R
                        normal=np.asarray(c.frame[:3])
                        face=R[:,2]
                        incidence=abs(normal@before)/max(.1,np.linalg.norm(before))
                        return [1.,np.linalg.norm(local[:2])/.07,abs(normal@face),
                                incidence,local[0]/.07,local[1]/.07,face[2],
                                before[2]/3.,np.linalg.norm(v-before)/4.,(tick+j/sim.timestep)/12.]
        return [0.]*10


def features(x):
    a=np.array(x)
    return np.concatenate((np.ones(a.shape[:-1]+(1,)),a,a[...,1:4]**2,
                           (a[...,1]*a[...,2])[...,None]),axis=-1)


def model(x,y):
    f=features(x)
    reg=3.*np.eye(f.shape[1]);reg[0,0]=.01
    w=np.zeros(f.shape[1])
    for _ in range(10):
        p=expit(f@w)
        curvature=f.T@(f*(p*(1-p))[:,None])+reg
        w+=np.linalg.solve(curvature,f.T@(y-p)-reg@w)
    return w


def auc(y,score):
    y=np.array(y,dtype=bool)
    r=rankdata(score)
    n=y.sum()
    return float((r[y].sum()-n*(n+1)/2)/(n*(len(y)-n))) if 0<n<len(y) else None


def train():
    paths=['rally-anchor-612423769','rally-anchor-8351422','rally-anchor-220381117','rally-anchor-1505337903-zero-guard',
           'paired-course','benchmark-learned']
    player=RallyPlayer(FlightLaw());probe=ContactProbe(player.sim)
    rows=[];used=set()
    try:
        for f in paths:
            d=_read(DIR/(f+'.json.gz'))
            runs=d.get('rows',[])
            if isinstance(runs,dict):
                runs=runs['frozen']
            for r in runs:
                if r['seed'] in used:
                    continue
                used.add(r['seed']);player.reset()
                hist=[];optical=dict(r['optical'])
                shots=r['shots'];by_t={s['tick']:i for i,s in enumerate(shots)}
                actions=np.array(r['actions'])
                for t,a in enumerate(actions):
                    hist.append((t,optical.get(t)))
                    if t in by_t:
                        n=by_t[t]
                        stop=shots[n+1]['tick'] if n+1<len(shots) else len(actions)+1
                        ev=[e for e in r['events'] if t<e['step']<=stop and e.get('side')==1]
                        obs=observed_ball(hist,optical_start(hist,player.law.gravity),player.law)
                        values=probe.measure(obs,actions[t:min(t+HORIZON,stop),:5])
                        rows.append(dict(seed=r['seed'],tick=t,x=values,
                            legal=any(e['kind']=='legal_return' for e in ev),contact=any(e['kind']=='contact' for e in ev),opening=(n==0)))
                    player.sim.twin.advance(a)
    finally:
        player.close()
    x=np.array([r['x'] for r in rows]);y=np.array([r['legal'] for r in rows],dtype=float)
    ids=np.array([r['seed'] for r in rows])
    pred=np.zeros(len(rows))
    for g in set(ids):
        held=ids==g
        pred[held]=features(x[held])@model(x[~held],y[~held])
    score=auc(y,pred)
    diagnostics=dict(n=len(rows),seeds=len(used),label=int(sum(y)),held_auc=score,
                     predicted_face_hits=int(x[:,0].sum()),median_legal=np.median(x[y>0],axis=0).tolist(),
                     median_fault=np.median(x[y==0],axis=0).tolist())
    print(diagnostics,flush=True)
    _write(DIR/'face-practice.json.gz',dict(rows=rows,model=model(x,y).tolist(),held=pred.tolist(),summary=diagnostics))


if __name__=='__main__':
    train()
