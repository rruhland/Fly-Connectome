"""Test whether real and RGB-observable success have intersecting motor support."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from . import tt57_motor_school as school
from . import tt57_motor_school_mature as mature
from . import tt57_motor_school_transfer as pupil
from .tt4_transfer_trial import _hash, _read, _write

ROOT=school.ROOT
PLAN=ROOT/'joint.protocol.json'
OUT=ROOT/'joint-teacher.json.gz'
R=96
N=6
OFFSETS=[(np.array([s*.0018,s*-.0023,s*.0028]),np.array([s*.023,s*.043,s*.073]),np.array([16.+17.*s,0.,0.]))
         for s in (-1.,1.)]


def protocol():
    doc=_read(mature.OUT); rows=[]; ids=set()
    for j,r in enumerate(doc['rows']):
        i=next((i for i,v in enumerate(r.get('arms',[])) if v['answered'] and v.get('issued_relative') and v['native_legal']),None)
        if i is None or r['case']['id'] in ids:continue
        ids.add(r['case']['id']); rows.append(dict(index=j,arm=i,id=r['case']['id'],lane=r['case']['lane']))
        if len(rows)>=N:break
    if len(rows)!=N:raise ValueError('Need sufficient distinct original physical witnesses')
    _write(PLAN,dict(rows=rows,practice_sha256=_hash(mature.OUT),limits=dict(rounds=2,count=R,physical_branches=2),
        scope='Earliest six physically answered teacher releases with exact/native agreement, chosen without reference to RGB recognition. Same two candidate populations scored in exact and independently fitted optically observed twins. Challenge is recognition and command robustness on these practice cases, explicitly not held or whole-game transfer.'))
    print('joint roster',[(r['id'],r['lane']) for r in rows],flush=True)


def score(engine,clinic,pop,ref):
    u,y,f,ex,exact=school.native(engine,pop,ref)
    a=clinic.twin
    p=a.ball.copy(); w=a.velocity.copy(); j=a._ball_dof
    spin=a.data.qvel[j+3:j+6].copy()
    losses=[]; legal=[]
    try:
        for dp,dv,dw in [(0,0,0),*OFFSETS]:
            a.ball[:]=p+dp; a.velocity[:]=w+dv; a.data.qvel[j+3:j+6]=spin+dw
            _,_,_,rank,ok=school.native(clinic.predict,pop,ref)
            losses.append(rank); legal.append(ok)
    finally:
        a.ball[:]=p; a.velocity[:]=w; a.data.qvel[j+3:j+6]=spin
    stress=legal[1].astype(int)+legal[2].astype(int)
    observed=legal[0]
    qualify=exact&observed&(stress>=1)
    quality=ex+.65*losses[0]+.30*(losses[1]+losses[2])-14.*qualify-.45*stress
    return dict(u=u,exact=exact,exact_loss=ex,observed=observed,stress=stress,qualified=qualify,quality=quality)


def selection(q):
    return dict(exact=int(q['exact'].sum()),optical=int(q['observed'].sum()),
                overlap=int(np.sum(q['exact']&q['observed'])),stress_qualified=int(np.sum(q['qualified'])))


def candidates(earned,loss,rng):
    pool=[earned,loss]
    while len(pool)<R:
        p=earned+rng.uniform(-.36,.48)*(earned-loss)
        delta=rng.normal(0,.40,size=5)
        p[:,:5]+=delta[None,:]*(earned-loss)[:,:5]
        p=school.warp(p,rng.uniform(.90,1.13),rng.uniform(-1.45,1.45))
        pool.append(p+school.mutation(rng,rng.uniform(.12,.56)))
    return np.asarray(pool)


def refine(pop,scores,rng):
    index=list(dict.fromkeys([0,*np.argsort(scores['quality'])[:11],*np.argsort(scores['exact_loss'])[:11]]))
    best=[pop[i] for i in index]
    pool=best.copy()
    while len(pool)<R:
        i,k=rng.integers(len(best),size=2); mix=rng.uniform(0.02,.35)
        p=(1.-mix)*best[i]+mix*best[k] if len(pool)%4==0 else best[i]
        p=school.warp(p,rng.uniform(.94,1.07),rng.uniform(-.85,.85))
        pool.append(p+school.mutation(rng,rng.uniform(.10,.39)))
    return np.asarray(pool)


def work(a,cam,rig,guard,doc,row,arm):
    spec=row['case']; card=spec['card']; tick=spec['tick']
    old=doc['rows'][spec['i']]; native=school.Rollout(a,1)
    stride=a._joint_increments[native.axes]*(a.control_dt/a.model.opt.timestep)
    xeye,x,q,ref,_,features,hist=school.observed_prefix(a,cam,rig,card,old,tick,native.axes)
    saved=school.State(a)
    clinic=pupil.OpticalClinic(body=True)
    try:
        pupil.warm_clinic(clinic,rig,a.control_dt,hist,xeye,tick,q,ref)
        if not clinic.sync() or features is None:raise ValueError('Lost causal RGB cue on fixed teacher witness')
        pv_error=np.r_[clinic.twin.ball-a.ball,clinic.twin.velocity-a.velocity]
        goal=np.asarray(arm['issued_relative'])
        f=np.asarray(row['stock']['controls'][:school.H])
        f=school.atlas.reference_path(ref,f,stride,a._joint_low[native.axes],a._joint_high[native.axes])[1:]
        f=school.mpc.tail(f,school.H)-ref
        rng=np.random.default_rng([57288,card['seed'],tick])
        p0=candidates(goal,f,rng); s0=score(native,clinic,p0,ref)
        p1=refine(p0,s0,rng); s1=score(native,clinic,p1,ref)
        pop=np.concatenate((p0,p1)); s={k:np.concatenate((s0[k],s1[k])) for k in s0}
        best=np.flatnonzero(s['exact']); joint=np.flatnonzero(s['qualified'])
        response={}; executed={}
        for label,rank,key in (('exact_only',best,'exact_loss'),('joint',joint,'quality')):
            if not len(rank):continue
            i=int(rank[np.argmin(s[key][rank])]); rates=s['u'][i,:school.H]
            if i not in executed:
                tape=np.vstack((rates,np.zeros((school.T+94-school.H,6))))
                r=school.physical(a,saved,card,old,spec,tape,guard)
                r.update(evaluator_proposal=i,exact=bool(s['exact'][i]),optical=bool(s['observed'][i]),
                         votes=int(s['stress'][i]),novel_linf_rad=float(np.max(abs(pop[i]-goal))))
                if r['landed'] and r['answered'] and not r['clearance']['penetration_samples']:
                    r['issued_relative']=(school.atlas.reference_path(ref,rates,stride,a._joint_low[native.axes],a._joint_high[native.axes])[1:]-ref).tolist()
                executed[i]=r
            response[label]=executed[i]
        return dict(case=spec,key=dict(features=features.tolist(),x=x.tolist(),q=q.tolist(),ref=ref.tolist()),
                    evaluator_optic_pv_error=pv_error.tolist(),original=dict(exact=bool(s0['exact'][0]),
                    optical=bool(s0['observed'][0]),votes=int(s0['stress'][0])),
                    batches=[selection(s0),selection(s1)],responses=response)
    finally:
        clinic.twin.close()


def run():
    design=_read(PLAN); doc=_read(mature.OUT)
    if design['practice_sha256']!=_hash(mature.OUT):raise ValueError('Changed paired-teacher practice source')
    output=dict(protocol_sha256=_hash(PLAN),code_sha256=_hash(Path(__file__)),rows=[],complete=False)
    if OUT.exists():
        old=_read(OUT)
        if (old['protocol_sha256'],old['code_sha256'])!=(output['protocol_sha256'],output['code_sha256']):
            raise ValueError('Cannot resume modified dual teacher')
        output=old
    cache={}
    with school.enabled():
        cad=school.mpc.arena(width=64,height=48,detail='cad'); guard=school.CourtClearance(cad.model,articulated=True)
        try:
            for q in design['rows'][len(output['rows']):]:
                start=process_time(); r=doc['rows'][q['index']]; spec=r['case']; src=spec['source']
                if spec['id']!=q['id']:raise ValueError('Changed training case order')
                if src not in cache:cache[src]=_read(Path(src))
                with school.mpc.centered(spec['ready']):
                    a=school.mpc.arena(); cam,rig,_,_,_=school.mpc.resources(a)
                    try:
                        proof=work(a,cam,rig,guard,cache[src],r,r['arms'][q['arm']]); proof['cpu_s']=process_time()-start
                    finally:
                        cam.close(); a.close()
                output['rows'].append(proof); output['complete']=len(output['rows'])==len(design['rows'])
                _write(OUT,output)
                print(q['id'],q['lane'],proof['original'],proof['batches'],
                      [(l,v['landed'],v['answered'],bool(v.get('issued_relative'))) for l,v in proof['responses'].items()],flush=True)
        finally:
            cad.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('plan','run')); a=p.parse_args()
    protocol() if a.mode=='plan' else run()
