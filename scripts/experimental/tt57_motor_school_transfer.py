"""Select learned direct movements with RGB and body encoders, never the native teacher state."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt57_motor_school as school
from . import tt57_motor_school_mature as maturity
from . import tt56_postures as recorded
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_optical_clinic import OpticalClinic
from .tt37_opponent_audit import CourtClearance
from .tt39_carry_joints import Carried
from .tt44_paired_credit import State

mpc=school.mpc
season=school.season
course=school.course
ROOT=school.ROOT
SCHOOL=ROOT/'readout.npz'
HELD=ROOT/'held-forks.json.gz'
CONTROL_PLAN=ROOT/'routine.protocol.json'
CONTROL_OUT=ROOT/'routine-forks.json.gz'
GAMES=ROOT/'games.json.gz'
DIAG=ROOT/'optical-witnesses.json'
ATLAS=school.atlas
H=school.H
K=12
SCALE=np.r_[np.ones(6),np.ones(5)*.62,np.ones(5)*.46,np.ones(5)*.34,.65]


def fit(out):
    design=_read(school.PLAN); train=_read(maturity.OUT)
    if train['original_practice_sha256']!=_hash(school.OUT):
        raise ValueError('Changed original practice receipt')
    if train['protocol_sha256']!=_hash(school.PLAN) or any(x['case']['id'] not in {v['id'] for v in design['practice']} for x in train['rows']):
        raise ValueError('School contains a release outside the practice roster')
    keep=[(r,a) for r in train['rows'] for a in r.get('arms',[]) if a['answered'] and a.get('issued_relative')]
    if len({r['case']['id'] for r,_ in keep})<4:
        raise ValueError('Insufficient physically answered movements for transfer')
    np.savez_compressed(out,keys=[r['inputs']['features'] for r,a in keep],
                        x=[r['inputs']['x'] for r,a in keep],
                        ref=[r['inputs']['ref'] for r,a in keep],
                        paths=[a['issued_relative'] for r,a in keep],
                        ids=[r['case']['id'] for r,a in keep],
                        source=_hash(maturity.OUT),protocol=_hash(school.PLAN))
    print('distilled',len(keep),len({r['case']['id'] for r,_ in keep}),flush=True)


def eta(x):
    return np.clip((x[1]+1.55)/max(1.6,-x[4]),.065,.47)


class Readout:
    def __init__(self,path=SCHOOL):
        with np.load(path) as f:
            self.keys=f['keys']*SCALE; self.x=f['x']; self.ref=f['ref']; self.paths=f['paths']; self.releases=f['ids']
            self.source=str(f['source'])
        self.knn=cKDTree(self.keys)

    def __call__(self,x,features,ref):
        distance,index=self.knn.query(features*SCALE,k=len(self.keys))
        distance,index=np.atleast_1d(distance),np.atleast_1d(index)
        pair=[]; ids=set()
        for d,j in zip(distance,index,strict=True):
            release=int(self.releases[j]); delta=self.x[j]-x
            if abs(delta[0])>.145 or abs(delta[2])>.13 or abs(delta[4])>1.23 or release in ids:
                continue
            if d>4.2:
                break
            ids.add(release)
            pair.append((float(d),j,school.warp(self.paths[j],np.clip(eta(self.x[j])/eta(x),.72,1.36),0.)))
            if len(pair)==4:
                break
        out=[]; meta=[]
        for d,j,path in pair:
            # Mostly body-relative, with an absolute geometry correction that
            # cannot jump current actuator bounds or import unknown physics.
            refshift=np.clip(self.ref[j]-ref,-.28,.28)
            mix=np.minimum(1.,school.PHASE/(.50*H))[:,None]
            for w in (.0,.54):
                out.append(path+mix*w*refshift)
                meta.append(dict(distance=d,release=int(self.releases[j]),absolute=w))
        if len(pair)>=2:
            for (_,j,a),(_,k,b) in [(pair[0],v) for v in pair[1:]]:
                if np.max(abs(a[5::5,:5]-b[5::5,:5]))<.46:
                    out.append(.72*a+.28*b)
                    meta.append(dict(distance=pair[0][0],release=int(self.releases[j]),mixed_with=int(self.releases[k])))
        return out[:K],meta[:K]


def archive(bank,x,key,release=None):
    roots,distances=bank.paths(key,x,release)
    out=[m for _,m in roots]; notes=[dict(distance=d,release=i) for (i,_),d in zip(roots,distances,strict=True)]
    return out[:K],notes[:K]


def choose(engine,paths,metadata,ref):
    if not paths:
        return None,dict(attempts=0,legal=0,robust=0)
    if not engine.sync():
        return None,dict(attempts=0,legal=0,robust=0)
    nominal=engine.twin
    u,y,f,loss,legal=school.native(engine.predict,np.asarray(paths),ref)
    evidence=dict(attempts=len(paths),legal=int(legal.sum()),robust=0)
    ii=np.flatnonzero(legal)
    if not len(ii):return None,evidence
    ii=ii[np.argsort(loss[ii])[:5]]
    p=nominal.ball.copy(); v=nominal.velocity.copy()
    dof=nominal._ball_dof
    w=nominal.data.qvel[dof+3:dof+6].copy()
    votes=np.zeros(len(ii),dtype=int); risk=loss[ii].copy()
    try:
        for s in (-1.,1.):
            nominal.ball[:]=p+s*np.array([.0018,-.0023,.0028]); nominal.velocity[:]=v+s*np.array([.023,.043,.073]); nominal.data.qvel[dof+3:dof+6]=w+[16.+17.*s,0.,0.]
            _,_,_,other,valid=school.native(engine.predict,np.asarray(paths)[ii],ref)
            votes+=valid
            risk+=np.where(valid,other,4.)*.65
    finally:
        nominal.ball[:]=p; nominal.velocity[:]=v; nominal.data.qvel[dof+3:dof+6]=w
    cost=np.where(votes>=1,risk-.55*votes,np.inf)
    evidence['robust']=int(np.sum(votes>=1))
    if np.isfinite(cost).any():
        k=int(np.argmin(cost)); j=int(ii[k]); m=metadata[j]
        evidence.update(m,selected=int(j),vote_count=int(votes[k]),score=float(cost[k]),contact_y=float(f[j,6]))
        return u[j,:H],evidence
    return None,evidence


def check(p):
    plan=_read(school.PLAN)
    if any(_hash(Path(f))!=s for f,s in plan['sources'].items()):
        raise ValueError('Changed physical evaluator tapes')
    with np.load(p) as f:
        if str(f['protocol'])!=_hash(school.PLAN) or {q['id'] for q in plan['held']}&set(f['ids'].tolist()):
            raise ValueError('Held release entered the readout')
    return plan


def plan_routine():
    plan=_read(school.PLAN); ids={r['id'] for r in plan['held']}; available=[]
    for name in school.SOURCES['held']:
        doc=_read(name)
        if _hash(name)!=plan['sources'][str(name)]:
            raise ValueError('Changed routine control source')
        for i,(c,r) in enumerate(zip(doc['cards'],doc['rows'])):
            if c['id'] not in ids or c['profile']!='reach' or any(d['draw']!='random' for d in r['placements']['requested']):
                continue
            for o in r['placements']['delivered'][:4]:
                if not o['near_landed'] or not all(o.get(k) for k in ('near_contact','far_contact','near_bounce','receive_line')):
                    continue
                z=float(school.projected_height(o['receive_line'])); x=float(o['receive_line']['xyz'][0])
                if not 1.026<=z<=1.14:continue
                windows=[n for n in r['searches'] if n.get('p') and -.99<n['p'][1]<-.51
                         and o['far_contact']['step']+3<n['tick']<o['near_contact']['step']-9]
                if not windows:continue
                s=windows[0]
                lane='left' if x<=-.10 else 'right' if x>=.10 else 'middle'
                available.append(dict(id=c['id'],i=i,leg=o['leg'],tick=s['tick'],x=x,height=z,lane=lane,
                       source=str(name),stock_robust=int(s.get('robust',0)),stock_scored=True,
                       late_offer_sha256=school.digest_json(o['near_bounce'])))
    selected=[]; left=available.copy()
    while left and len(selected)<8:
        r=min(left,key=lambda v:(v['stock_robust']>0,sum(w['lane']==v['lane'] for w in selected),v['tick'],abs(v['height']-1.06)))
        selected.append(r); left=[v for v in left if v['id']!=r['id']]
    if len(selected)<6:raise ValueError('Too few independent routine held positive controls')
    _write(CONTROL_PLAN,dict(base_sha256=_hash(school.PLAN),rows=selected,
        policy='Routine already-scored ordinary RGB windows, one per held release ID, balancing lanes and prioritizing unprotected stock decisions. Frozen without new holdout actions or outcomes. Six to eight trials use identical donor check, command budget, native referee, and matched incoming requirement as the terminal misses.'))
    print('routine control roster',[(r['id'],r['lane'],round(r['height'],3),r['stock_robust']) for r in selected],flush=True)


def warm_clinic(clinic,rig,dt,history,eye,tick,q,ref):
    prefix_eye=mpc.Vision(rig,dt,raw=True)
    prefix_eye.eye=mpc.LearnedOptics(prefix_eye.eye,course.OPTIC)
    for j,im,q0,r0 in history:
        mpc.observed(prefix_eye,j,im,q0)
        clinic.observe(j,prefix_eye.eye,q0,r0.copy())
    clinic.observe(tick,eye.eye,q,ref.copy())


def audit_training(count):
    """Can RGB/encoder simulation even recognize a known physical practice win?"""
    data={}; practice=_read(school.OUT)
    result=dict(practice_sha256=_hash(school.OUT),rows=[],
                scope='Recognition on physically earned training commands only, not transfer.')
    with school.enabled():
        for r in practice['rows']:
            cmd=next((v for v in r.get('arms',[]) if v.get('issued_relative') and v['answered']),None)
            if cmd is None:continue
            spec=r['case']; source=spec['source']; tick=spec['tick']
            if source not in data:data[source]=_read(Path(source))
            doc=data[source]; old=doc['rows'][spec['i']]
            with mpc.centered(spec['ready']):
                a=mpc.arena(); cam,rig,*_=mpc.resources(a)
                clinic=OpticalClinic(body=True)
                try:
                    eye,x,q,ref,_,features,hist=school.observed_prefix(a,cam,rig,spec['card'],old,tick,a._arm_ids[1][2])
                    warm_clinic(clinic,rig,a.control_dt,hist,eye,tick,q,ref)
                    selected,d=choose(clinic,[cmd['issued_relative']],[dict(release=spec['id'])],ref)
                    result['rows'].append(dict(id=spec['id'],lane=spec['lane'],tick=tick,x=x.tolist(),
                            exact_native_legal=cmd['native_legal'],observed_native=d,would_issue=selected is not None))
                    _write(DIAG,result)
                    print('optical witness',spec['id'],cmd['native_legal'],d,flush=True)
                finally:
                    clinic.twin.close(); cam.close(); a.close()
            if len(result['rows'])>=count:break


def test(readout_path,count,out=HELD,routine=False):
    plan=check(readout_path)
    if routine:
        c=_read(CONTROL_PLAN)
        if c['base_sha256']!=_hash(school.PLAN):raise ValueError('Changed routine controls')
        cohort=c['rows']
    else:cohort=plan['held']
    receipt=dict(schema='tt57-rgb-encoders-only-action-1',plan_sha256=_hash(school.PLAN),
                 motor_readout_sha256=_hash(readout_path),rows=[],complete=False,
                 cohort_sha256=_hash(CONTROL_PLAN) if routine else _hash(school.PLAN))
    if out.exists():
        old=_read(out)
        if any(old[k]!=receipt[k] for k in ('plan_sha256','motor_readout_sha256','cohort_sha256')):
            raise ValueError('Changed held readout')
        receipt=old
    library=Readout(readout_path); donors=school.Donors(); data={}
    with school.enabled():
        cad=mpc.arena(width=64,height=48,detail='cad')
        try:
            guard=CourtClearance(cad.model,articulated=True)
            for spec in cohort[len(receipt['rows']):count]:
                src=spec['source']; doc=data.setdefault(src,_read(Path(src))) if src not in data else data[src]
                idx=spec['i']; card,row=doc['cards'][idx],doc['rows'][idx]; tick=spec['tick']
                with mpc.centered(doc['ready']):
                    a=mpc.arena(); cam,rig,*_=mpc.resources(a)
                    clinic=OpticalClinic(body=True)
                    try:
                        axes=a._arm_ids[1][2]
                        eye,x,q,ref,old,feature,hist=school.observed_prefix(a,cam,rig,card,row,tick,axes)
                        saved=State(a)
                        baseline=school.physical(a,saved,card,row,spec,row['actions'][tick:],guard=None)
                        n=baseline['stock_prefix_events']
                        stock_robust=next(v.get('robust',0) for v in row['searches'] if v['tick']==tick)
                        if (baseline['events']!=row['events'][n:n+len(baseline['events'])]
                            or bool(baseline['landed'])!=bool(routine)
                            or (not routine and baseline['event_sha256']!=school.digest_json(row['events']))):
                            raise ValueError('Held exact baseline failed')
                        arms={}; optics=x is not None and feature is not None and x[4]<-.9 and -1.03<x[1]<-.47
                        if optics and stock_robust<=0:
                            warm_clinic(clinic,rig,a.control_dt,hist,eye,tick,q,ref)
                            key=ATLAS.key(tick,x[:3],bool(eye.eye.latest.bounced),ref,old)
                            offers={'school':library(x,feature,ref),'atlas':archive(donors,x,key)}
                        else:offers={k:([],[]) for k in ('school','atlas')}
                        for mode,(paths,meta) in offers.items():
                            decision,verdict=choose(clinic,paths,meta,ref)
                            if decision is None:
                                result=baseline
                            else:
                                cmd=np.vstack((decision,np.zeros((school.T+94-H,6))))
                                result=school.physical(a,saved,card,row,spec,cmd)
                                if result['landed']:
                                    audit=school.physical(a,saved,card,row,spec,cmd,guard)
                                    if audit['event_sha256']!=result['event_sha256']:
                                        raise ValueError('CAD changed actual rescue tape')
                                    result.update(clearance=audit['clearance'])
                            arms[mode]=dict(decision=verdict,issued=decision is not None,**result)
                        r=dict(case=dict(spec,card=card,ready=doc['ready']),stock=baseline,arms=arms,optical=optics,
                               stock_robust=stock_robust,public_context=None if not optics else dict(x=x.tolist(),q=q.tolist(),
                               ref=ref.tolist(),features=feature.tolist(),bounced=bool(eye.eye.latest.bounced)))
                        receipt['rows'].append(r)
                        receipt['complete']=len(receipt['rows'])==len(cohort)
                        _write(out,receipt)
                        print('routine' if routine else 'held',len(receipt['rows']),spec['id'],spec['lane'],stock_robust,
                              [(k,v['decision'],v['landed'],v['answered']) for k,v in arms.items()],flush=True)
                    finally:
                        clinic.twin.close(); cam.close(); a.close()
        finally:
            cad.close()


class Player(season.Player):
    def __init__(self,*args,readout,**kwargs):
        super().__init__(*args,**kwargs)
        self.school=readout

    def begin(self):
        super().begin()
        self.school_pick=None
        self.school_tick=-100
        self.school_log=[]

    def search(self,tick,x,q,ref,base):
        super().search(tick,x,q,ref,base)
        if not self.notes or self.notes[-1]['tick']!=tick or self.notes[-1].get('robust',0)>0:
            return
        if not -1.03<x[1]<-.47 or x[4]>=-.9:
            return
        feature=school.context(x,q,ref,self.engine.rate,self.engine.bounced)
        paths,meta=self.school(x,feature,ref)
        if not paths:
            return
        choice,evidence=choose(self.engine,paths,meta,ref)
        self.school_log.append(dict(tick=tick,issued=choice is not None,**evidence))
        if choice is not None:
            self.school_pick=np.clip(ref+np.cumsum(choice,axis=0)*self.school_stride,self.school_lo,self.school_hi)
            self.school_tick=tick
            self.best=self.school_pick; self.tick=tick; self.lease=H

    def step(self,tick,eye,x,q,ref,base):
        if self.school_pick is not None and 0<=tick-self.school_tick<H and (x is None or x[4]<.9):
            self.engine.observe(tick,eye,q,ref.copy())
            return self.school_pick[tick-self.school_tick]
        self.school_pick=None
        return super().step(tick,eye,x,q,ref,base)


def games(path,out=GAMES):
    plan=check(path); immutable=recorded.STOCK; stock=_read(immutable)
    train_ids={q['id'] for q in plan['practice']}
    # First six standard assessment cards. No treated result chooses a card.
    roster=list(range(6)); cap=14
    if any(stock['cards'][i]['id'] in train_ids for i in roster):
        raise ValueError('Physical whole-game cohort overlaps practice IDs')
    result=dict(plan_sha256=_hash(school.PLAN),readout_sha256=_hash(path),stock_sha256=_hash(immutable),indexes=roster,cap=cap,rows=[])
    if out.exists():
        old=_read(out)
        if any(old[k]!=result[k] for k in result if k!='rows'):
            raise ValueError('Changed game comparison')
        result=old
    with np.load(mpc.BANK) as f:raw=mpc.DeepField(f,'local')
    with mpc.centered(stock['ready']),school.enabled():
        a=mpc.arena(); cam,rig,stride,lo,hi=mpc.resources(a)
        rank=season.fresh.shot.Coverage(_read(season.fresh.RISK))
        ids={stock['cards'][i]['id'] for i in roster}
        with patch.object(mpc,'Fragments',lambda b,e:Carried(b,e,practice=[course.PRACTICE],held=ids)):
            actor=Player(mpc.BANK,raw,variant=stock['variant'],budget=stock['budget'],coverage=rank,readout=Readout(path))
        actor.school_stride=stride; actor.school_lo=lo; actor.school_hi=hi
        try:
            course.resume(actor,season.CARRY.with_suffix('').with_suffix('.npz'),_read(season.CARRY))
            for i in roster[len(result['rows']):]:
                card=stock['cards'][i]; old=recorded.baseline_truncated(stock['rows'][i],cap)
                r=mpc.play(a,cam,rig,stride,lo,hi,stock['ready'],card,old['shot'],raw,actor,course.OPTIC,'frozen',cap)
                paired=recorded.compare(old,r)
                r.update(paired,school=actor.school_log)
                if r['memory']!=old['memory'] or any(v['target']!=list(recorded.draw(card,j,'random',None)[0]) for j,v in enumerate(r['placements']['requested'])):
                    raise ValueError('Changed frozen bank or opponent target law')
                result['rows'].append(dict(id=card['id'],stock_score=old['score']['learner'],new=r))
                _write(out,result)
                print('games',i,card['id'],old['score']['learner'],r['score']['learner'],sum(v['issued'] for v in r['school']),flush=True)
        finally:
            actor.close(); cam.close(); a.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('fit','held','games','audit-training','plan-routine','routine'))
    p.add_argument('--n',type=int,default=12)
    p.add_argument('--readout',type=Path,default=SCHOOL)
    a=p.parse_args()
    if a.mode=='fit':fit(a.readout)
    elif a.mode=='held':test(a.readout,a.n)
    elif a.mode=='routine':test(a.readout,a.n,CONTROL_OUT,routine=True)
    elif a.mode=='plan-routine':plan_routine()
    elif a.mode=='audit-training':audit_training(a.n)
    else:games(a.readout)
