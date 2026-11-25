"""Propose early low motions from bounded optical trajectory hypotheses and earned motor examples."""

import numpy as np

from .tt38_motorcoach import AIM, answerable
from .tt43_stance_chart import path_options
from .tt43_stance_live import OneHub
from .tt43_stance_native import dynamic_options

SHALLOW = (-1.015, -1.074, -1.151, -1.232, -1.33)
# (z, toward-near v, vertical v, topspin). Fixed assumptions about
# uncertain early flight, not samples of the measured current ball state.
OFFSETS = np.array([[0.,0.,0.,0.], [.014,-.4,.34,17.],
                    [-.013,.28,-.32,-18.], [.025,-.50,-.18,-33.],
                    [-.012,-.48,.53,31.]])


def projected_height(path):
    if not len(path):
        return None
    use=path[abs(path[:,2]+1.42)<.04]
    return None if not len(use) else float(use[np.argmin(abs(use[:,2]+1.42)),3])


class LowHub(OneHub):
    def __init__(self,*args,warm=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.warm=warm

    def begin(self):
        super().begin()
        self.uncertain_next=0
        self.probe_next=0
        self.probe_q=None
        self.uncertain_commit=False
        self.low_warm=[]

    def step(self,tick,eye,x,q,reference,base):
        original=super().step(tick,eye,x,q,reference,base)
        if x is not None and x[4]>.9:
            self.probe_q=None
            self.uncertain_commit=False
            self.low_warm=[]
        if ((x is None or x[4]>=-.9 or not -.96<x[1]<.69)
                or not self.home_started or self.tape is not None or self.deep_supervisor
                or (self.best is not None and tick-self.tick<self.lease)
                or not self.engine.sync()):
            return original
        if tick<self.uncertain_next:
            if self.probe_q is not None and tick<self.probe_next:
                return self.probe_q
            return original

        self.uncertain_next=tick+4
        t=self.engine.twin
        if t.launches or t.card is not None:
            raise ValueError("External RGB/encoder-only actor requires separate physics")
        w=t.data.qvel[t._ball_dof+3:t._ball_dof+6]
        p=t.ball.copy()
        v=t.velocity.copy()
        s=w.copy()

        # Keep the regular broad-depth controller if the existing optical
        # estimate has a robustly ordinary projected bounce.
        path=t.partner.flight.path(p,v,s,1,bounced=self.engine.bounced,horizon=.66)
        height=projected_height(path)
        if height is not None and height>1.027:
            self.probe_q=None
            return original

        diag=dict(tick=tick,mode="hub_low_ensemble",y=float(x[1]),
                  projected_height_m=height,bounced=bool(self.engine.bounced),hypotheses=[])
        self.notes.append(diag)
        candidates=[]
        prep=[]
        try:
            for mode,offset in enumerate(OFFSETS):
                # Once a table rebound is visible the perturbed predictions are
                # much narrower. No hypothesis marks a bounce not seen by RGB.
                dz,dy,dv,ds=offset*(.37 if self.engine.bounced else 1.)
                t.ball[:]=p+[0.,0.,dz]
                t.velocity[:]=v+[0.,dy,dv]
                w[:]=s+[ds,0.,0.]
                h=t.partner.flight.path(t.ball,t.velocity,w,1,bounced=self.engine.bounced,horizon=.66)
                P=path_options(self.chart,h,quick=(2 if self.warm else True),depths=SHALLOW)
                P=[r for r in P if .050<r["eta"]<.61 and r["p"][2]<1.025 and
                   np.max(abs(q[:5]-r["q"]))/4.16<r["eta"]-.017 and
                   self.chart.guard.route(q[:5],np.array(r["q"]),steps=8)]
                for r in P:
                    slack=r["eta"]-np.max(abs(q[:5]-r["q"]))/4.16
                    prep.append((slack-.019*mode,r,mode))
                hit,log=dynamic_options(t,self.engine.predict,self.chart.ik,P,None,32,2,
                                        [433832,tick,mode],memories=self.earned)
                diag["hypotheses"].append(dict(poses=len(P),tested=sum(r["tried"] for r in log),
                                           contacts=sum(r["contact"] for r in log),
                                           legal=sum(r["legal"] for r in log),usable=hit is not None))
                if hit is not None:
                    candidates.append((hit,mode))
        finally:
            t.ball[:]=p;t.velocity[:]=v;w[:]=s
        self.diag["low_epochs"]+=1
        self.diag["low_trials"]+=sum(a["tested"] for a in diag["hypotheses"])

        previous=list(candidates)
        if self.warm:
            for origin, old in self.low_warm:
                delta=tick-origin
                left=old["quality"][1]-delta*t.control_dt
                trace=np.asarray(old["goals"])
                if delta<=0 or not .041<left<.53:
                    continue
                for shift in (0,-2,2):
                    idx=max(0,delta+shift)
                    rem=trace[idx:]
                    if len(rem)<3:
                        continue
                    q=old["quality"].copy()
                    q[1]=left-shift*t.control_dt
                    candidates.append((dict(old,goals=rem.tolist(),quality=q),5+abs(shift)))
            self.low_warm=[(origin,old) for origin,old in self.low_warm
                           if old["quality"][1]-(tick-origin)*t.control_dt>.045]
            if previous:
                self.low_warm=( [(tick,p) for p,_ in previous[:2]] + self.low_warm )[:3]
            diag["retained_and_retimed"]=len(candidates)-len(previous)

        if candidates:
            n=max(len(r["goals"]) for r,_ in candidates)
            goals=np.array([np.vstack((r["goals"],np.broadcast_to(r["goals"][-1],(n-len(r["goals"]),6))))
                            for r,_ in candidates])
            goals=np.concatenate((goals,np.broadcast_to(goals[:,-1:, :],(len(goals),60,6))),axis=1)
            cutoff=float(max(r["quality"][1]+.14 for r,_ in candidates))
            feedback=np.zeros((len(goals),9))
            _,value=self.engine.trajectories(goals,AIM,cutoff,feedback=feedback)
            nominal=np.isfinite(value[:,0]) & (value[:,5]>.012)
            if nominal.any():
                idx=min(np.flatnonzero(nominal),key=lambda i:answerable(value[i]))
                result,mode=candidates[idx]
                eta=value[idx,1]
                # Before an observed rebound recheck on a new exposure and
                # issue only a bounded prefix. No referee outcome is read.
                if self.engine.bounced or eta<.090:
                    self.tape=np.asarray(result["goals"])
                    self.tape_start=tick
                    self.pending={**result,"contact_tick":tick+round(eta/t.control_dt)}
                    self.uncertain_commit=True
                    self.diag["low_commits"]+=1
                    diag.update(issued="nominal commit",selected=int(mode),eta=float(eta))
                    return self.tape[0]
                self.probe_q=goals[idx,0]
                self.probe_next=tick+4
                self.diag["low_updates"]+=1
                diag.update(issued="temporary nominal",selected=int(mode))
                return self.probe_q
            if self.warm:
                far_from_contact=[(i,c) for i,c in enumerate(candidates)
                                  if c[0]["quality"][1]>.106 and x[1]>-.78]
                if not far_from_contact:
                    self.probe_q=None
                    return original
                _,(best,mode)=min(far_from_contact,key=lambda z:
                    2.6*min(feedback[z[0],0],.40) - .18*feedback[z[0],1]+
                    .04*z[1][0]["cost"]+.008*z[1][1])
            else:
                best,mode=min(candidates,key=lambda r:r[0]["cost"]+.14*r[1])
            self.probe_q=np.asarray(best["goals"][0])
            self.probe_next=tick+4
            self.diag["low_updates"]+=1
            diag.update(issued="temporary alternate",selected=int(mode))
            return self.probe_q
        if prep and x[1]>-.69:
            # Geometry only purchases three ticks of positioning. A later
            # estimate must pass a landing check before it can strike.
            _,r,mode=max(prep,key=lambda a:a[0])
            self.probe_q=np.r_[r["q"],self.hub[-1]]
            self.probe_next=tick+3
            self.diag["low_geometry_prep"]+=1
            diag.update(issued="temporary geometry",selected=int(mode))
            return self.probe_q
        self.probe_q=None
        return original


def fitted_camera_examples(chart, rows, payload):
    """Compress rewarded stereo-and-reference strips into pose-relative drive residuals."""
    from .tt41_shallow_model import impulse

    examples=[]
    for mode,row in enumerate(rows):
        hit=row.get("measured_contact")
        if hit is None or not (row["teacher_actual_legal"] and row["teacher_actual_answered"]):
            continue
        keep=payload["stamps"][:,0]==mode
        t=payload["stamps"][keep,1]/60.
        q=payload["encoders"][keep,:5]
        targets=payload["previous_references"][keep,:5]
        if not len(t):
            continue
        idx=np.argmin(abs(t-hit["time"]))
        nearby=(t-hit["time"]>-.070)&(t-hit["time"]<.044)
        if nearby.sum()<4:
            continue
        p=np.asarray(hit["center"])
        v=np.asarray(hit["incoming"])
        real=q[idx]
        sign=1 if chart.ik.pose(real)[1][1]>=0 else -1
        z=[r for r in chart.poses_at(p,v,4.4,.15,real) if r["sign"]==sign
           and np.max(abs(r["q"]-real))<.46]
        if not z:
            continue
        guess=min(z,key=lambda r:np.max(abs(r["q"]-real)))
        _,u=impulse(p,v,4.4,.13)
        _,_,J,N=chart.ik.pose(guess["q"],jacobian=True)
        biomech=np.linalg.lstsq(np.r_[J,.065*N,.023*np.eye(5)],np.r_[u,np.zeros(8)],rcond=None)[0].clip(-5.8,5.8)
        fit=np.linalg.lstsq(np.c_[t[nearby]-hit["time"],np.ones(nearby.sum())],targets[nearby],rcond=None)[0]
        drift,at_hit=fit
        params=np.r_[np.clip(at_hit-.04*drift-guess["q"],-.43,.43),
                     np.clip(drift-biomech,-3.6,3.6),0.,0.,0.]
        examples.append(dict(p=p.tolist(),v=v.tolist(),face=sign,parameters=params.tolist(),drive_mode=2))
    return examples
