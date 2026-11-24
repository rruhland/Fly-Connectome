"""Seek earlier, contact-compatible low strokes with alternate body configurations."""

import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from . import tt38_joint_mpc as mpc
from . import tt39_terminal_fork as forks
from . import tt39_terminal_servo as servo
from .tt39_terminal_reachable import shoot
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import NOMINAL_NO_LOAD

HORIZON = 29
SCALE = np.r_[.055, .070, .052, .40, .40, .40]


def advance_score(p, incoming, n, normal_speed):
    """Cheap setup screen only. The optical court decides actual contact/legality."""
    outgoing = incoming + 1.80*(normal_speed - incoming @ n)*n
    if outgoing[1] <= 2.25:
        return np.inf
    g, v = 9.81, outgoing[2]
    landing_t = (v + np.sqrt(max(0.,v*v + 2*g*(p[2]-.765))))/g
    across = -p[1]/(.89*outgoing[1])
    net_z = p[2]+v*across-.5*g*across*across
    at = p[:2]+.88*landing_t*outgoing[:2]
    return (2.*abs(at[0]) + 1.8*abs(at[1]-.68) + 8.*max(0.,.93-net_z)
            + 2.*max(0.,.08-at[1]) + 3.*max(at[1]-1.26,0.))


class ExtendedBody(servo.GroundedBody):
    def endpoints(self, commands):
        x = np.ascontiguousarray(commands, dtype=float)
        n, frames, axes = x.shape
        if not (1 <= frames <= HORIZON and axes == 6) or not np.isfinite(x).all():
            raise ValueError("Invalid extended body reference batch")
        out = np.empty((n,9))
        self.call(self.model._address,self.data._address,x.ctypes.data,n,frames,self.micro,self.racket,
                  self.motors.ctypes.data,self.dofs.ctypes.data,self.step.ctypes.data,
                  self.low.ctypes.data,self.high.ctypes.data,self.limits.ctypes.data,
                  NOMINAL_NO_LOAD,.8,out.ctypes.data)
        return out


class EarlyServo(servo.TerminalServo):
    """A longer body-only inverse solve, with static pose seeds but no stroke templates."""

    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.body=ExtendedBody(self.engine)
        # These are candidate starting points for nonlinear pose inversion only.
        self.q_seeds = qmc.scale(qmc.Sobol(5,scramble=True,seed=391092).random_base2(13),
                                self.geometry.low,self.geometry.high)
        poses = np.array([np.r_[*self.geometry.pose(q)] for q in self.q_seeds])
        self.pose_index = cKDTree(poses/SCALE)

    def early_targets(self,q):
        t=self.engine.twin
        path=self.engine.nominal.flight.path(t.ball,t.velocity,t.spin,1,bounced=self.engine.bounced,horizon=.49)
        if not len(path):
            return []
        time,p=path[:,0],path[:,1:4]
        # Prefer the available part of the rebound in front of the common deep stroke.
        low_exists = np.any((p[:,1] < -1.50) & (p[:,1] > -1.68) & (p[:,2] < .995) & (abs(p[:,0]) < .155))
        inside = (time>=.105)&(time<HORIZON*mpc.DT)&(p[:,1]>-1.56)&(p[:,1]<-1.03)
        inside &= (p[:,2]>.81)&(p[:,2]<1.09)&(abs(p[:,0])<.155)
        eligible=np.flatnonzero(inside)
        if not len(eligible) or not low_exists:
            return []
        rows=[]
        for k in np.unique(np.rint(np.linspace(eligible[0],eligible[-1],5)).astype(int)):
            frames=round(time[k]/mpc.DT)
            if not inside[k] or not 6<=frames<=HORIZON:
                continue
            j=int(np.argmin(abs(time-frames*mpc.DT)))
            ball,vin=path[j,1:4],path[j,4:7]
            n0=servo.desired_pose(ball,vin,4.20)[3:6]
            vtan=np.array([0.,0.,1.])-n0*n0[2]
            for pitch in (-.185,.0,.15):
                desired_n=n0+pitch*vtan
                desired_n/=np.linalg.norm(desired_n)
                center=ball-.026*desired_n
                _, inds=self.pose_index.query(np.r_[center,desired_n]/SCALE,6)
                nearest=self.q_seeds[inds]
                picked=[q[:5],nearest[0]]
                i=int(np.argmax(np.max(abs(nearest-nearest[0]),axis=1)))
                if np.max(abs(nearest[i]-nearest[0]))>.36:
                    picked.append(nearest[i])
                for q0 in picked:
                    angles, fit=self.geometry.fit(center,desired_n,q0)
                    cp,n,J,_=fit
                    delta=ball-cp
                    normal=float(delta@n)
                    rss=np.linalg.norm(delta-normal*n)
                    joint_travel=np.max(abs(q[:5]-angles))
                    if not (.007<normal<.051 and rss<.055 and joint_travel<3.75*frames*mpc.DT+.15):
                        continue
                    budget=min(1.85,float(3.7*np.linalg.norm(n@J,ord=1)))
                    for speed in (.55,.85,1.15,1.45,1.72):
                        if speed>budget:
                            continue
                        merit=advance_score(ball,vin,n,speed)
                        if not np.isfinite(merit) or merit>2.25:
                            continue
                        rank=merit+5.*rss+4.*abs(normal-.025)+.3*joint_travel + .18*(speed/budget)**2
                        rows.append((rank,frames,np.r_[cp,n,speed],angles,fit))
        rows.sort(key=lambda r:r[0])
        unique=[]
        for r in rows:
            if any(abs(r[1]-old[1])<3 and np.linalg.norm(r[2][3:6]-old[2][3:6])<.14 for old in unique):
                continue
            unique.append(r)
            if len(unique)>=3:
                break
        return unique

    def schedule(self, q, reference, frames, goal, angles, nominal):
        _, _, j, a=nominal
        j=np.vstack((j,.10*a))
        vgoal=np.r_[goal[3:6]*goal[6],np.zeros(3)]
        drive=np.clip(j.T@np.linalg.solve(j@j.T+.017**2*np.eye(6),vgoal),-6.9,6.9)
        dq=self.engine.body_state.qvel[self.engine.dofs[:5]]
        t=np.arange(1,frames+1)*mpc.DT
        u=t/t[-1]
        aim=(2*u**3-3*u*u+1)[:,None]*q[:5]+(-2*u**3+3*u*u)[:,None]*angles
        aim+=t[-1]*((u**3-2*u*u+u)[:,None]*dq+(u**3-u*u)[:,None]*drive)
        aim+=np.gradient(aim,mpc.DT,axis=0)*.040
        return np.c_[aim,np.full(frames,reference[5])],drive

    def continued_review(self, goals):
        model = self.engine.twin
        p,v=model.ball.copy(),model.velocity.copy()
        w=model.data.qvel[model._ball_dof+3:model._ball_dof+6]
        old=w.copy()
        records=[]
        commands=np.asarray([mpc.tail(g,HORIZON+mpc.TAIL) for g in goals])
        try:
            for off in (0.,-1.,1.):
                model.ball[:]=p+off*np.array([.0016,.002,.0032])
                model.velocity[:]=v+off*np.array([.021,.040,.085])
                w[:]=old+np.array([17.*(off+1.) if off else 0.,0.,0.])
                response=np.zeros((len(goals),9))
                _, values=self.engine.trajectories(commands,mpc.AIM,(HORIZON+.1)*mpc.DT,feedback=response)
                ranks, legal=mpc.cost(values,response)
                records.append((ranks,legal,values,response[:,1].copy()))
                self.diag["native_candidates"]+=len(goals)
        finally:
            model.ball[:]=p
            model.velocity[:]=v
            w[:]=old
        ranks=np.array([r[0] for r in records])
        legal=np.array([r[1] for r in records])
        return ranks,legal,records[0][2],sum(np.sum(r[3]) for r in records)

    def attempt(self,tick,x,q,reference,note):
        if tick-self.last_shoot<6 or abs(x[0])>.23 or x[4]>=-.9 or not -1.42<x[1]<.23:
            return False
        self.last_shoot=tick
        start=process_time()
        self.body.snapshot()
        targets=self.early_targets(q)
        proposals, proof=[],[]
        for _, frames, goal, angle, kin in targets:
            base,drive=self.schedule(q,reference,frames,goal,angle,kin)
            result=shoot(self.body,base,goal)
            if result is None:
                continue
            seq, trace=result
            follow=np.arange(1,HORIZON-frames+1)*mpc.DT
            ext=np.c_[seq[-1,:5] + np.minimum(follow,.075)[:,None]*drive[None]*.56,np.full(len(follow),reference[5])]
            proposals.append(np.vstack((seq,ext)))
            proof.append(dict(frames=frames,**trace))
        note.update(early_target_count=len(targets),shooting=proof)
        self.diag["shoot_solves"]+=len(proof)
        self.diag["early_attempts"]+=1
        accepted=False
        if proposals:
            rank,legal,value,contacts=self.continued_review(proposals)
            wins=legal.sum(axis=0)
            ids=np.flatnonzero(legal[0]&(wins>=2))
            self.diag["shoot_legal"]+=int(legal[0].sum())
            note.update(shoot_legal=int(legal[0].sum()),tested_contacts=int(contacts),stress_legals=wins.tolist())
            if len(ids):
                k=min(ids,key=lambda i:np.mean(np.where(legal[:,i],rank[:,i],2.9)))
                self.best=proposals[k]
                self.tick=tick
                self.next=tick+6
                self.lease=6
                self.diag["shoot_issued"]+=1
                note.update(chosen="terminal_body_shooting", robust=int(wins[k]-1),terminal_index=int(k),
                            terminal_contact_s=float(value[k,1]),terminal_center_radius_m=float(value[k,12]))
                accepted=True
        self.diag["shoot_cpu_s"]+=process_time()-start
        return accepted

    def search(self,tick,x,q,reference,base):
        mpc.MPC.search(self,tick,x,q,reference,base)
        if self.notes and self.notes[-1]["tick"]==tick and not self.notes[-1].get("chosen"):
            self.attempt(tick,x,q,reference,self.notes[-1])

    def step(self,tick,eye,x,q,reference,base):
        self.engine.observe(tick,eye,q,reference.copy())
        if x is not None and x[4]>.9:
            self.best=None
            return base
        if x is not None and x[4]<-.9 and -.49<=x[1]<.23 and abs(x[0])<=.23 and tick-self.last_shoot>=6:
            if self.engine.sync():
                note=dict(tick=tick,p=x[:3].tolist(),chosen=None,early=True)
                self.attempt(tick,x,q,reference,note)
                self.notes.append(note)
        if (x is not None and x[4]<-.9 and -1.62<x[1]<-.49 and x[2]>.77 and tick>=self.next):
            self.next=tick+5
            self.search(tick,x,q,reference,base)
        if self.best is not None and 0<=tick-self.tick<self.lease:
            self.diag["screened_ticks"]+=1
            return self.best[tick-self.tick]
        return base


def diagnostic_times(row):
    existing=forks_original(row)
    last=row["placements"]["delivered"][-1]
    if not existing:
        return []
    # Analysis-only fork selection. The whole-game actor uses the optical state gate.
    return sorted(set(range(last["far_contact"]["step"]+9,existing[0],6))|set(existing))


forks_original=forks.eligible


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--limit",type=int,default=4)
    p.add_argument("--bank",type=Path,default=mpc.BANK)
    p.add_argument("--optics",type=Path,default=Path("artifacts/tt37/rebound.npz"))
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--deck",choices=("development","held_openings","guarded_challenge"))
    args=p.parse_args()
    if args.deck is None:
        with patch.object(forks,"eligible",diagnostic_times),patch.object(forks,"TerminalServo",EarlyServo):
            forks.course(args.bank,args.optics,args.output,args.limit)
    else:
        with patch.object(mpc,"MPC",EarlyServo):
            mpc.run(args.feeds,args.bank,mpc.PLAN,args.deck,"live",args.output,variant="elastic",optic=args.optics,limit=args.limit)
    data=_read(args.output)
    data["terminal_early_sha256"]=_hash(Path(__file__))
    _write(args.output,data)
