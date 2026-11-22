"""Update the timing of an already committed joint stroke from continuing stereo measurements."""
from pathlib import Path

import numpy as np

from .tt4_arena import AIR_DRAG,BALL_INERTIA,BALL_MASS,MAGNUS_COEFFICIENT,SPIN_DECAY,TABLE_HEIGHT,mujoco
from .tt4_transfer_trial import ROOT,_read,_write
from .tt19_factory import RATE
from .tt22_flight_inverse import FlightInverse
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import Station,replay


def crossing(fitter,state,y):
    p,v,w,_=state
    sim=fitter.data
    model=fitter.model
    mujoco.mj_resetData(model,sim)
    sim.qpos[:3]=p+[0.,0.,TABLE_HEIGHT]
    sim.qpos[3:7]=[1.,0.,0.,0.]
    sim.qvel[:3]=v
    sim.qvel[3:6]=w
    mujoco.mj_forward(model,sim)
    spin=np.empty(3)
    previous=sim.qpos[:3].copy()
    for i in range(18*fitter.micro):
        velocity=sim.qvel[:3]
        mujoco.mju_rotVecQuat(spin,sim.qvel[3:6],sim.qpos[3:7])
        sim.xfrc_applied[1,:3]=BALL_MASS*(-AIR_DRAG*np.linalg.norm(velocity)*velocity+
                                           MAGNUS_COEFFICIENT*np.cross(spin,velocity))
        sim.xfrc_applied[1,3:]=-BALL_INERTIA*SPIN_DECAY*spin
        mujoco.mj_step(model,sim)
        new=sim.qpos[:3]
        if previous[1]>=y>new[1]:
            frac=(y-previous[1])/(new[1]-previous[1])
            return (i+frac)/fitter.micro, previous+frac*(new-previous)-[0.,0.,TABLE_HEIGHT]
        previous[:]=new
    return None


class TrimPlayer(RallyPlayer):
    def __init__(self,law, *, gain):
        super().__init__(law)
        self.late_refine=True
        self.lens=FlightInverse()
        self.gain=gain

    def search(self,estimate,observation,target):
        chosen=super().search(estimate,observation,target)
        if chosen is None:
            return None
        tape,note=chosen
        self.original=np.vstack((target,np.clip(target+np.cumsum(tape*RATE[:5],axis=0),
                                                self.factory.low,self.factory.high)))
        self.correction=np.zeros(3)
        self.phase=0.
        self.previous_update=-1
        note['trims']=[]
        return chosen

    def refine_shot(self,tick,estimate,current):
        start,tape=self.plan
        note=self.shots[-1]
        elapsed=tick-start
        remaining=note['eta']-elapsed
        if not 1.0<remaining<11 or elapsed<1 or self.history[-1][1] is None:
            return
        if tick-self.previous_update>=2:
            fit=self.lens.fit(self.history)
            predicted=crossing(self.lens,fit,note['depth']) if fit is not None else None
            if predicted is not None:
                eta,center=predicted
                # Retain the chosen swing and its originally intended contact offset.
                self.phase=float(self.gain*np.clip(eta-remaining,-1.4,1.4))
                delta=center-np.asarray(note['anchor'])
                self.correction=self.gain*np.clip(delta,[-.028,0.,-.032],[.028,0.,.032])
                self.previous_update=tick
                note['trims'].append(dict(tick=tick,phase=self.phase,translation=self.correction.tolist()))
        if self.previous_update<0:
            return
        phase=np.clip(elapsed+1-self.phase,0,len(self.original)-1)
        lo=int(phase)
        frac=phase-lo
        q=(1-frac)*self.original[lo]+frac*self.original[min(lo+1,len(self.original)-1)]
        _,_,j,jn=self.factory.pose(q)
        a=np.vstack((j,.10*jn))
        change=np.linalg.solve(a.T@a+.0005*np.eye(5),a.T@np.r_[self.correction,np.zeros(3)])
        goal=np.clip(q+change,self.factory.low,self.factory.high)
        tape[elapsed]=np.clip((goal-current)/RATE[:5],-1.,1.)


def run(count,seed,gain,path):
    doc=dict(rows=[],seed=seed,gain=gain,privileged=False,spread=.30,pace=.82,rival='anchor')
    station=Station(.30,.82,'anchor')
    law=station.player.law
    station.player.close()
    station.player=TrimPlayer(law,gain=gain)
    try:
        for s in np.random.default_rng(seed).integers(2**31,size=count):
            r=station.point(int(s)); doc['rows'].append(r)
            print(s,r['longest'],sum(len(p['trims']) for p in r['shots']),flush=True)
            _write(path,doc)
    finally:
        station.close()
    doc['verified']=replay(doc['rows'],.30,.82,'anchor')
    _write(path,doc)


if __name__=='__main__':
    import argparse
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--count',type=int,default=5)
    arg.add_argument('--seed',type=int,default=8351422)
    arg.add_argument('--gain',type=float,default=.65)
    arg.add_argument('--output',type=Path,default=ROOT/'artifacts/tt22/trim-pilot.json.gz')
    args=arg.parse_args()
    run(args.count,args.seed,args.gain,args.output)
