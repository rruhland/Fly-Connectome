"""Recall physically rewarded opening strokes only after a protected optical preflight fails."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt33_joint as joint, tt37_motor_clinic as clinic
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt37_belief import ArrivalHistory
from .tt37_opponent import arena
from .tt37_visual_inertia import LearnedOptics
from .tt38_impact_margin import ROOT, SHIFTS, games, quality
from .tt38_motorcoach import setup


def eligible(row):
    out=[]
    for decision in row['records']:
        n=decision.get('impact_margin') or {}
        if n.get('status')!='issued':
            continue
        t=decision['tick']/60
        shot=next((s for s in row['score']['shots'] if s['side']==1 and t<s['hit']<=t+decision['score'][1]+.18),None)
        if shot is not None and any(s['side']==-1 and s['hit']>shot['landing'] for s in row['score']['shots']):
            out.append((decision,shot))
    return out


def collect(source, out):
    d=_read(source)
    if not d['completed'] or d['stage']!='margin':
        raise ValueError('Training requires a complete physical impact-margin tape')
    s=setup(d['depth']); joint.RECEIVE_Y=d['depth']
    doc=dict(source_sha256=_hash(source),practice_release_ids=sorted({c['id'] for c in d['cards']}),
             source=str(source),receipts=[],samples=[],original_ready=d['ready'])
    with centered(d['ready']):
        a=arena(); render,rig,stride,lo,hi=joint.resources(a)
        try:
            for c,row in zip(d['cards'],d['rows'],strict=True):
                teach=eligible(row)
                if not teach:
                    continue
                a.launch(row['shot'],card=c); a.partner.reset(seed=c['seed'])
                eye=joint.Vision(rig,a.control_dt)
                eye.eye=LearnedOptics(eye.eye,d['visual'])
                history=ArrivalHistory()
                qref=np.asarray(d['ready']); refs=[qref.copy()]; trace={}
                desired=set()
                for decision,shot in teach:
                    clock=decision['tick']
                    desired.update(range(max(0,clock-46),clock+25))
                for tick,u in enumerate(row['actions']):
                    q=a.joints.copy()
                    frame=render.capture() if tick%2==0 and tick in desired else None
                    opt=history.update(tick*a.control_dt,eye.update(tick,frame,q))
                    trace[tick]=(opt,q)
                    a.exchange(u,clinic.FAR)
                    qref=np.clip(qref+stride*np.asarray(u),lo,hi); refs.append(qref.copy())
                if digest_json(a.events)!=digest_json(row['events']):
                    raise ValueError('Practice command replay did not reproduce physical returns')
                n_before=len(doc['samples'])
                for decision,shot in teach:
                    candidates=[]
                    for tick in range(max(0,decision['tick']-22),decision['tick']):
                        opt,q=trace[tick]
                        if opt is not None and .255<opt[0]<.380 and .76<opt[2]<1.38:
                            candidates.append((tick,opt,q))
                    if not candidates:
                        continue
                    tick,opt,q=min(candidates,key=lambda v:abs(v[1][0]-.32))
                    chart,_=clinic.geometry(s,opt)
                    current_camera_eta=tick+round(opt[0]*60)
                    local=refs[tick+1:min(len(refs),current_camera_eta+7)]
                    if len(local)<13:
                        continue
                    doc['samples'].append(dict(x=np.r_[opt[1:6],q[:5]].tolist(),chart=chart.tolist(),eta=float(opt[0]),
                        targets=np.asarray(local).tolist(),actual_departure=shot['outgoing_mps'],
                        actual_landing=shot['landing_xyz'],answered=True,
                        mark=dict(id=c['id'],tick=tick,coach_tick=decision['tick'])))
                doc['receipts'].append(dict(id=c['id'],points_sha256=digest_json(row['events']),
                                             rewarded=len(teach),examples=len(doc['samples'])-n_before))
                _write(out,doc)
                print(c['id'],len(teach),len(doc['samples'])-n_before,flush=True)
        finally:
            render.close(); a.close()
    doc['complete']=True
    _write(out,doc)


class ProtectedPractice:
    def __init__(self, path):
        d=_read(path)
        if not d.get('complete') or len(d['samples'])<4:
            raise ValueError('Need four or more physically rewarded control histories')
        self.samples=d['samples']
        self.practice_ids=set(d['practice_release_ids'])
        self.provenance=dict(bank_sha256=_hash(path),samples=len(self.samples),
                             source_sha256=d['source_sha256'],trigger_eta=[.259,.367])
        self.x=np.array([r['x'] for r in self.samples]); self.known=np.array([r['chart'] for r in self.samples])
        self.begin()

    def begin(self):
        self.active=None; self.notes=[]; self.last=-100.

    @staticmethod
    def guard(engine):
        if engine.twin.launches or engine.twin.card is not None:
            raise ValueError('Motion recall can only see the independent optical arena')

    @staticmethod
    def predict(engine, goals, cutoff):
        t=engine.twin
        f0=np.zeros((len(goals),9))
        _,r0=engine.trajectories(goals,clinic.AIM,cutoff,feedback=f0)
        q,valid=quality(r0,f0)
        score=[q]; legal=[valid]
        p,v=t.ball.copy(),t.velocity.copy()
        w=t.data.qvel[t._ball_dof+3:t._ball_dof+6]; initial=w.copy()
        try:
            for d in SHIFTS[1:5]:
                t.ball[:]=p+d[:3]; t.velocity[:]=v+d[3:6]; w[:]=initial+d[6:]
                f=np.zeros((len(goals),9))
                _,r=engine.trajectories(goals,clinic.AIM,cutoff,feedback=f)
                s,h=quality(r,f); score.append(s); legal.append(h)
        finally:
            t.ball[:]=p; t.velocity[:]=v; w[:]=initial
        return r0,f0,np.mean(score,axis=0),np.sum(legal,axis=0)

    def matched(self, obs, encoder):
        x=np.r_[obs[1:6],encoder[:5]]
        d=np.linalg.norm((self.x-x)/clinic.SCALE,axis=1)
        return [(i,float(d[i])) for i in np.argsort(d)[:4] if d[i]<1.75],x

    def move(self, index, eta, dt, n, chart):
        d=self.samples[index]
        refs=np.asarray(d['targets']).copy()
        refs[:,:5]+=np.clip(chart-self.known[index],-.18,.18)[None,:]
        a=np.arange(n)+(d['eta']-eta)/dt
        steps=np.arange(len(refs))
        return np.stack([np.interp(a,steps,v) for v in refs.T],axis=-1)

    def prepare(self, engine, surface, obs, bank, ordinary):
        self.guard(engine)
        dt=engine.twin.control_dt; now=engine.tick*dt
        if self.active is not None:
            age=engine.tick-self.active['tick']
            if 0<=age<self.active['frames'] and obs[0]>.18:
                return self.active['tape'][age]
            if now-self.last>.45:
                self.active=None
        if now-self.last<.45 or not (.259<obs[0]<.367):
            return ordinary
        self.last=now
        match,x=self.matched(obs,engine.previous)
        note=dict(tick=engine.tick,eta=float(obs[0]),status='no nearby preparation')
        self.notes.append(note)
        if not match or not engine.sync():
            return ordinary
        cut=float(obs[0]); fit=clinic.geometry(surface,obs)
        length=round((cut+1.05)/dt)
        from .tt37_reuse import options
        old=options(bank,x)
        old=np.vstack([old[:12],np.zeros(10),clinic.generate(16)[1:]])
        prep=max(1,round((cut-.169)/dt))
        late=clinic.schedules(*fit,cut-prep*dt,old,dt,surface.ready,length-prep)
        stock=np.concatenate([np.broadcast_to(ordinary,(len(late),prep,6)),late],axis=1)
        rolls,h,q,viable=self.predict(engine,stock,cut+.13)
        note.update(stock_modes=len(stock),stock_nominal=int(np.isfinite(rolls[:,0]).sum()),stock_robust=int(viable.max()))
        if np.any(np.isfinite(rolls[:,0])&(viable>=3)):
            note['status']='stock protected early'; return ordinary
        tapes=np.array([self.move(i,cut,dt,length,fit[0]) for i,d in match])
        out,f,quality,good=self.predict(engine,tapes,cut+.13)
        valid=np.isfinite(out[:,0])&(good>=4)
        note['borrowed_nominal']=int(np.isfinite(out[:,0]).sum()); note['borrowed_robust']=int(good.max())
        if not valid.any():
            note['status']='no robust earned preparation'; return ordinary
        selected=int(np.argmax(np.where(valid,quality,-np.inf)))
        i,dist=match[selected]
        self.active=dict(tick=engine.tick,frames=prep,tape=tapes[selected],index=i)
        note.update(status='applied early preparation',nearest_distance=dist,frames=prep,
                    early_predicted_landing=out[selected,3:5].tolist(),historical_contact_answered=True)
        return tapes[selected,0]

    def plan(self, engine, surface, obs, selected, old_score, fit, eta):
        self.guard(engine)
        if self.active is None or engine.tick-self.active['tick']>15:
            return dict(tape=None,score=None,eta=None,trace=[],note=dict(status='stock unchanged'))
        dt=engine.twin.control_dt; n=round((eta+1.05)/dt)
        gs=clinic.schedules(*fit,eta,selected[None],dt,surface.ready,n)
        t=self.move(self.active['index'],eta,dt,n,fit[0])
        r,f,q,v=self.predict(engine,np.concatenate([gs,t[None]]),eta+.13)
        trace=[dict(stage='impact_margin_early_recheck',trials=10,contacts=int(f[:,1].sum()),legal=int(v.sum()))]
        note=dict(status='stock protected late',stock_legal_cases=int(v[0]),candidate_legal_cases=int(v[1]),
                  stock_nominal_legal=old_score is not None)
        if np.isfinite(r[0,0]) and v[0]>=3 or not (np.isfinite(r[1,0]) and v[1]>=4 and q[1]-q[0]>.32):
            self.notes.append(dict(tick=engine.tick,**note))
            return dict(tape=None,score=None,eta=None,trace=trace,note=note)
        contact=float(r[1,1]); frames=round((contact+.08)/dt)
        note.update(status='issued',candidate=dict(family='earned_early_history'),feedback=f[1].tolist(),
                    nominal_candidates=1,entered_nominal_legal=old_score is not None)
        self.notes.append(dict(tick=engine.tick,**note))
        return dict(tape=t[:frames],score=r[1].tolist(),eta=contact,trace=trace,note=note)


def assess(bank, stock, output, n):
    p=ProtectedPractice(bank)
    ref=_read(stock)
    if p.practice_ids & {c['id'] for c in ref['cards'][:n]}:
        raise ValueError('Evaluation roster contains a physical practice release')
    games(stock,output,'early-protected',count=n,policy=p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('collect','test'))
    p.add_argument('bank',type=Path)
    p.add_argument('tape',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--n',type=int,default=6)
    a=p.parse_args()
    if a.phase=='collect':
        collect(a.tape,a.bank)
    else:
        if a.out is None:
            p.error('The physical assessment needs an output path')
        assess(a.bank,a.tape,a.out,a.n)


if __name__=='__main__':
    main()
