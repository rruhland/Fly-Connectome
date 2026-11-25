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
    def begin(self):
        super().begin()
        self.uncertain_next=0
        self.probe_next=0
        self.probe_q=None
        self.uncertain_commit=False

    def step(self,tick,eye,x,q,reference,base):
        original=super().step(tick,eye,x,q,reference,base)
        if x is not None and x[4]>.9:
            self.probe_q=None
            self.uncertain_commit=False
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
                P=path_options(self.chart,h,quick=True,depths=SHALLOW)
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

        if candidates:
            n=max(len(r["goals"]) for r,_ in candidates)
            goals=np.array([np.vstack((r["goals"],np.broadcast_to(r["goals"][-1],(n-len(r["goals"]),6))))
                            for r,_ in candidates])
            goals=np.concatenate((goals,np.broadcast_to(goals[:,-1:, :],(len(goals),60,6))),axis=1)
            cutoff=float(max(r["quality"][1]+.14 for r,_ in candidates))
            _,value=self.engine.trajectories(goals,AIM,cutoff)
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
