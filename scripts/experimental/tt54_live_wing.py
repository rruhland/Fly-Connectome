"""Advance along a rewarded wing preparation using only early stereo and encoders."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt52_stock_repertory as rp
from .tt37_opponent_audit import CourtClearance, guarded
from .tt39_opportunity_map import projected_height
from .tt51_low_prep_bank import EncoderPreparation, OUT as BANK
from .tt51_low_early_neutrals import restore_readiness_from_actual_images

root=Path('artifacts/tt54_live_wing')
mpc=rp.mpc
LOOK=np.array([.26,.18,.22,.63,.94,.85])


class Wing:
    def __init__(self,packet,checker):
        self.bank=EncoderPreparation(packet)
        self.starts=np.array([s['rgb'][0] for s in packet['trajectories']])
        self.checker=checker
        self.reset()

    def reset(self):
        self.chosen=None;self.opened=None;self.inward=None;self.last_out=None
        self.declined=False;self.ended=False;self.recover=False;self.level=.22
        self.forecast=None;self.plan_phase=0;self.last_risk=None;self.logs=[]

    def follow(self,q,ref,want):
        lo,hi=self.checker.ik.low,self.checker.ik.high
        applied=np.clip(ref[:5]+np.clip(want[:5]-ref[:5],-4.16/60,4.16/60),lo,hi)
        if not self.checker.route(q[:5],applied,steps=4):return None
        return np.r_[applied,ref[-1]]

    def close(self,tick,why):
        if not self.ended:self.logs.append(dict(tick=tick,action='release',reason=why))
        self.ended=True

    def look(self,tick,x,ready):
        p=ready.predictor
        if p is None or not p.estimates:return
        z=p.estimates[-1]
        if z['tick']!=tick or z['gate']>.45:return
        f=np.asarray(z['distribution']);lane,h=f[5:7];risk=f[-1]
        if not np.isfinite(f).all():return
        self.last_risk=float(risk)
        self.forecast=(float(lane),float(h),float(risk))
        if lane>.105 or h>1.092:
            self.declined=True;self.recover=True
        else:
            lateral=np.clip((.095-lane)/.22,0.,1.)*np.clip((lane+.43)/.17,0.,1.)
            height=np.clip((1.075-h)/.084,0.,1.)
            conf=.68*float(np.clip(risk/.28,0.,1.))+.32*float(height)
            self.level=max(.22,min(.85,.28+.72*conf*lateral))
        self.logs.append(dict(tick=tick,action='far optical evidence',gate=z['gate'],lane=float(lane),
                              estimate_height=float(h),risk=float(risk),level=self.level,declined=self.declined))

    def observed_incoming(self,tick,x,engine):
        if not engine.sync():return None
        a=engine.twin
        path=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=engine.bounced,horizon=.69)
        row=path[abs(path[:,2]+1.10)<.045]
        if not len(row):return None
        r=row[np.argmin(abs(row[:,2]+1.10))]
        f=np.r_[r[1],r[3:7]]
        if not rp.pf.prior.bounded_forecast(f):return None
        lane,h=rp.pf.prior.deep_coordinates(f)
        return float(lane),float(h)

    def command(self,tick,x,q,ref,plain,p):
        if x is None:return None
        outward=x[4]>.9
        if outward and self.inward is not None:
            keep=self.logs
            self.reset();self.logs=keep
        if outward:self.last_out=tick
        if self.opened is None:
            if self.declined or not (outward and p.ready.seen_inbound and -.99<x[1]<-.65):return None
            optics=np.mean(((self.starts-x)/LOOK)**2,axis=1)
            v=self.bank.preview(q,ref)
            plausible=[r for r in v if r['supported'] and r['phase']<17 and optics[r['trajectory']]<1.35]
            if not plausible:return None
            self.chosen=min(plausible,key=lambda r:r['body_distance']+.35*optics[r['trajectory']])['trajectory']
            self.opened=tick
            self.logs.append(dict(tick=tick,action='supported outbound reserve',source=self.chosen,
                                  q=q.tolist(),ref=ref.tolist(),outgoing=x.tolist()))
        if tick-self.opened>116:
            self.close(tick,'optical rally timeout')
        if self.ended:
            if self.recover and x[1]>-.49:
                return self.follow(q,ref,plain)
            return None
        if outward:
            self.look(tick,x,p.ready)
        elif x[4]<-.9:
            if self.inward is None:
                if self.last_out is None or tick-self.last_out>21:
                    self.close(tick,'stale outward track');return None
                self.inward=tick
            if x[1]<-.49:
                self.close(tick,'ordinary stock window');return None
            if tick%2==0:
                estimate=self.observed_incoming(tick,x,p.engine)
                if estimate is not None:
                    lane,h=estimate
                    if lane>-.060 or lane<-.38 or h>1.073:
                        self.declined=True;self.recover=True
                    elif (tick-self.inward<=7 and h<1.035) or h<1.016:
                        self.level=1.;self.declined=False;self.recover=False
                    self.logs.append(dict(tick=tick,action='observed inbound forecast',lane=lane,
                                          estimate_height=h,level=self.level,declined=self.declined))
        if self.declined:
            self.recover=True
            return self.follow(q,ref,plain)
        rows=self.bank.rows[self.chosen];n=len(rows['q'])
        v=next(r for r in self.bank.preview(q,ref) if r['trajectory']==self.chosen)
        target=int(np.floor(self.level*(n-1)))
        # This speed-up is a separate causal control candidate, not inherited
        # evidence from the source replay. Encoders set phase and the unchanged
        # servo clips every reference step. The next step has an independent CAD check.
        phase=max(0,v['phase'],self.plan_phase-5)
        look=11 if self.inward is not None else 8
        use=int(min(target,phase+look))
        g=self.follow(q,ref,rows['next'][use])
        if g is None:
            self.declined=True;self.recover=True
            self.logs.append(dict(tick=tick,action='blocked by CAD step',goal_phase=use))
            return self.follow(q,ref,plain)
        self.plan_phase=use
        self.logs.append(dict(tick=tick,action='toward acquired body',phase=int(v['phase']),goal_phase=use,
                              scale=self.level,encoder_error=float(np.max(abs(q[:5]-rows['q'][-1,:5]))),
                              q=q.tolist(),ref=ref.tolist(),command=g.tolist()))
        return g


class Arm(rp.Arm):
    def __init__(self,*a,body=None,**kw):
        super().__init__(*a,**kw);self.wing=body

    def begin(self):
        super().begin()
        if self.wing is not None:self.wing.reset()
        self.body_trace=[] if self.wing is None else self.wing.logs

    def step(self,tick,eye,x,q,ref,base):
        plain=super().step(tick,eye,x,q,ref,base)
        if self.wing is None:return plain
        g=self.wing.command(tick,x,q,ref,plain,self)
        return plain if g is None else g


def install(raw,checker,reference,*,enabled,frames):
    prior=rp.pf.prior
    wing=Wing(rp._read(BANK),checker) if enabled else None
    p=Arm(mpc.BANK,raw,mode='partial',finish=rp.pf.Finish(),checker=checker,
          predictor=rp.old.logged_predictor(rp.risk_model),repertory=rp.Joined(None,rp.low_field(rp.low_artifact)),
          weak=True,body=wing,variant=reference['variant'],budget=reference['budget'],
          coverage=prior.fresh.shot.Coverage(rp._read(prior.fresh.RISK)))
    p.tap=frames
    return p


def run(path):
    prior=rp.pf.prior;reg=rp._read(prior.PLAN);source=rp._read(prior.ARCHIVE)
    case,original=prior.terminal(source)
    begin=reg['anchor'];keep=reg['keep_old_commands_through'];leg=case['leg']
    doc=dict(schema='tt54-live-wing-exact-1',source_sha256=rp._hash(Path(__file__)),bank_sha256=rp._hash(BANK),
             risk_sha256=rp._hash(rp.risk_model),case=case['id'],stream=case['card']['reply_seed'],rows=[],
             role='Original normal launch and scored in-point history. Prior swing preserved. New arm decisions from actual public RGB and encoders only.')
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    with mpc.centered(source['ready']):
        a=mpc.arena();cad=mpc.arena(detail='cad');guard=CourtClearance(cad.model,articulated=True)
        camera,rig,stride,lo,hi=mpc.resources(a);tap=rp.old.Tap(camera);axes=a._arm_ids[1][2]
        check=prior.Reach(cad,source['ready'],source['ready'])
        history=prior.prefix(a,case['card'],original,begin,camera,axes,window=34)
        saved=prior.State(a)
        for name,warm in (('prepared-stock',False),('causal-wing',True)):
            saved.restore(a);guarded(a,guard)
            with patch.object(mpc,'Fragments',lambda f,e:prior.Carried(f,e,practice=[prior.course.PRACTICE],held={case['id']})):
                p=install(raw,check,source,enabled=warm,frames=tap)
            carried={**source,'rows':[None]*len(rp._read(prior.season.CARRY)['rows'])+source['rows']}
            prior.course.resume(p,prior.BANK,carried)
            obs=prior.Observer(a,rig,p,history); eye=obs.restore(p)
            restore_readiness_from_actual_images(p,rig,a.control_dt,history)
            commands=[];observed=[];sensed=False
            try:
                for tick in range(begin,begin+240):
                    q=a.joints.copy();ref=a.data.ctrl[axes].copy();frames=tap.capture() if tick%2==0 else None
                    x=mpc.observed(eye,tick,frames,q)
                    if tick<=keep:
                        p.observe_only(tick,frames,eye.eye,x,q,ref);u=original['actions'][tick]
                    else:
                        if x is not None and x[4]<-.9 and x[1]<-.9:sensed=True
                        base=source['ready']
                        if x is not None and (x[4]<-.9 or (sensed and x[1]<.50)):
                            target,c=raw.get(x,q)
                            if target is not None:base=np.array(base)+c*(target-base)
                        g=p.step(tick,eye.eye,x,q,ref,base)
                        u=np.clip((np.clip(g,lo,hi)-ref)/stride,-.8,.8).tolist()
                    commands.append(u)
                    if x is not None:observed.append([tick,*x,*q,*ref])
                    _,_,done,capped,_=a.exchange(u,mpc.FAR_COMMAND)
                    if done or capped or a.rival_returns>=leg+2:break
                ps=a.point_placements();got=ps['delivered'][leg]
                same=all(prior.same_trace(got[k],case['original_offer'][k]) for k in ('far_contact','near_bounce'))
                if not same or ps['requested'][leg]['target']!=original['placements']['requested'][leg]['target']:
                    raise ValueError('Shifted original physical offer')
                row=dict(mode=name,legal=bool(got['near_landed']),answered=a.rival_returns>=leg+2,height=projected_height(got['receive_line']),
                         same_offer=same,offer=got,commands=commands,observed=observed,opts=p.repertory_log,
                         body_trace=p.body_trace,risk=p.ready.predictor.estimates,clearance=a.clearance_result(),
                         event_sha256=prior.digest_json(a.events))
                doc['rows'].append(row);rp._write(path,doc)
                print(name,row['legal'],row['answered'],len(row['body_trace']),len(row['opts']),flush=True)
            finally:p.close()
        camera.close();a.close();cad.close()
    doc['first_changed_tick']=next((begin+i for i,(u,v) in enumerate(zip(*(r['commands'] for r in doc['rows']))) if u!=v),None)
    doc['complete']=True;rp._write(path,doc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=root/'exact.json.gz')
    a=p.parse_args();root.mkdir(parents=True,exist_ok=True);run(a.out)
