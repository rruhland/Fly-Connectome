"""Physically test deep wide-face geometry even when its impulse prediction is poor."""

import argparse
from pathlib import Path

import numpy as np

from . import tt64_deep_receive as deep
from .tt4_transfer_trial import _hash, _read, _write

PLAN=deep.ROOT/'limits.protocol.json'
OUT=deep.ROOT/'limits.json.gz'
WIDTH=.058
LEADS=(.018,.048,.078)


def plan():
    raw=_read(deep.OUT)
    if not raw['complete']:raise ValueError('Finish the deep feasibility scan first')
    cases=[]
    for i,r in enumerate(raw['rows']):
        fits=[z for y in r['scan'] if -1.843<y['depth']<-1.647 for z in y['best'].values() if z['radial_m']<=WIDTH
              and .013<z['axial_m']<.041 and z['normal'][1]>.09]
        # Low plausible pose at each sign even if unfavorable predicted impulse.
        fits.sort(key=lambda v:(-v['normal'][1],v['radial_m']))
        cases.append(dict(index=i,release=r['case']['id'],fits=fits))
    _write(PLAN,dict(source_sha256=_hash(deep.OUT),docket_sha256=_hash(deep.PLAN),cases=cases,leads=LEADS,
        faces_per_case=2,reach_m=WIDTH,scope='Physically issue one standby and up to three directional speed-ceiling schedules per sign at best CAD-safe fitted plane behind the end of the table. Authorize on actual contact and ordinary legal landing/answer only. Training oracle timing; not an RGB actor. Never treat near contact or predicted stroke as reward.'))
    print('deep negative controls',[(q['release'],len(q['fits'])) for q in cases],flush=True)


def command(a,q,dq,time,lead,axes):
    dt=a.control_dt; stride=a._joint_increments[axes]*(dt/a.model.opt.timestep)
    lo,hi=a._joint_low[axes],a._joint_high[axes]
    old=a.data.ctrl[axes].copy(); cmd=[]
    for j in range(39):
        s=np.clip((j+1)*dt-(time-a.data.time)+lead,-.093,.059)
        aim=np.r_[q+s*dq,old[-1]]
        u=np.clip((np.clip(aim,lo,hi)-old)/stride,-.8,.8)
        old=np.clip(old+stride*u,lo,hi); cmd.append(u.tolist())
    return cmd


def candidate(r,fit,work,home,driven):
    q=np.array(fit['joint']); ik=work.ik
    center,normal,J,N=ik.pose(q,jacobian=True); normal*=fit['face']; N*=fit['face']
    sampled=np.array(r['used']['track']); line=sampled[np.argmin(abs(sampled[:,2]-fit['depth']))]
    v=line[4:7]; n=np.array(fit['normal']); w0,w1,vlim=deep.speed_family(q,n,J,N,ik.low,ik.high,v,20.)
    dq=w0 if driven else np.zeros(5)
    approach=q-.093*dq
    if not work.guard.clear(q):return None,'impact pose CAD blocked'
    middle=deep.route(work.guard,home,approach,ik.low,ik.high)
    if middle is None:return None,'preparatory path CAD blocked'
    if not work.guard.route(approach,q,steps=15):return None,'physical swing crosses self/fixture'
    # Deliberately generous normal reflection only. This is explanatory,
    # while actual impact and direction are measured independently below.
    ceiling=float(v[1]+1.8*(vlim-v@normal)*normal[1])
    return dict(q=q,dq=dq,approach=approach,waypoint=middle,face=fit['face'],time=float(r['table_separation']['time']+line[0]),
                predicted_forward_ceiling=ceiling,depth=fit['depth'],normal=n.tolist()),None


def physical(a,guard,row,card,r,option,arch):
    t0=r['fork']; cut=r['receiving']; axis=a._arm_ids[1][2]; leg=r['case']['leg']
    prep,prefixmatch,_=deep.long.physically_prepare(a,row,card,t0,cut,option,arch,brisk_flip=True)
    if not prefixmatch:return [],False
    state=deep.State(a); out=[]
    for lead in LEADS if np.any(option['dq']) else (.0,):
        state.restore(a); controls=command(a,option['q'],option['dq'],option['time'],lead,axis)
        c=deep.prior.received(a,guard,card,row,leg,prep,dict(tape=controls,lead=lead),True)
        impacts=[m for m in c['native_contacts'] if m['side']==1]; m=impacts[0] if impacts else None
        departure=next((e['velocity'] for e in c['events'] if m is not None and e['time']>m['time_s']
                         and e['kind']=='separation' and e.get('collider')=='learner_racket'),None)
        c.update(impact=m,departure=departure,lead=lead)
        if c['answered'] and not c['clearance']['penetration_samples'] and m is not None and m['radial_m']<=deep.REAL_RADIUS and abs(m['face_alignment'])>=.95:
            c['full_cad']=deep.full_verify(r['ready'],card,c,guard)
            if not c['full_cad']['penetration_samples']:c['wide_witness']=True
        out.append(c)
        if c.get('wide_witness'):break
    return out,True


def run():
    p=_read(PLAN); diag=_read(deep.OUT)
    if p['source_sha256']!=_hash(deep.OUT) or p['docket_sha256']!=_hash(deep.PLAN):raise ValueError('Changed deep geometry docket')
    result=dict(protocol_sha256=_hash(PLAN),rows=[],complete=False)
    with deep.enabled():
        for item in p['cases']:
            r=diag['rows'][item['index']]; s=r['case']; d=_read(Path(s['source'])); card,row=d['cards'][s['index']],d['rows'][s['index']]
            out=dict(case=s,excluded=[],attempts=[])
            with deep.mpc.centered(d['ready']):
                a=deep.mpc.arena(width=64,height=48); cad=deep.mpc.arena(width=64,height=48,detail='cad')
                guard=deep.CourtClearance(cad.model,articulated=True)
                try:
                    deep.long.replay_prefix(a,row,card,r['fork']); home=a.data.ctrl[a._arm_ids[1][2]][:5].copy()
                    work=deep.long.Workspace(a,r['ready']); used=set()
                    arch=[e for e in row['events'] if e['time']<=r['receiving']*a.control_dt+1e-8]
                    for fit in item['fits']:
                        face=fit['face']
                        if face in used:continue
                        still,reason=candidate(r,fit,work,home,False)
                        if still is None:
                            out['excluded'].append(dict(face=face,depth=fit['depth'],reason=reason));continue
                        active,why=candidate(r,fit,work,home,True)
                        used.add(face)
                        options=[('still',still),('normal-speed-ceiling',active)]
                        for k,goal in options:
                            if goal is None:
                                out['excluded'].append(dict(face=face,depth=fit['depth'],reason=why));continue
                            reps,match=physical(a,guard,row,card,r,goal,arch)
                            out['attempts'].append(dict(fit=fit,kind=k,recipe=deep.long.plain(goal),matched_prefix=match,physical=reps))
                            print(card['id'],face,k,match,[(l['legal'],None if l['impact'] is None else round(l['impact']['radial_m']*1000),l['departure']) for l in reps],flush=True)
                        if any(z.get('wide_witness') for q in out['attempts'] for z in q['physical']):break
                finally:
                    cad.close();a.close()
            result['rows'].append(out); result['complete']=len(result['rows'])==len(p['cases']); _write(OUT,result)
            print('case complete',card['id'],len(out['attempts']),out['excluded'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('plan','run')); a=p.parse_args()
    plan() if a.mode=='plan' else run()
