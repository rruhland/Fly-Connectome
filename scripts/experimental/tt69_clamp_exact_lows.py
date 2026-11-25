"""Readout-only intervention on existing TT68 naturally attained low terminal bodies."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt69_clamp_school as school
from . import tt69_clamp_policy as p
from . import tt69_clamp_repair as repair
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt37_optical_clinic import OpticalClinic
from .tt40_long_low import same_trace
from .tt44_paired_credit import State, prefix

ROOT=school.ROOT
PLAN=ROOT/'exact-low.protocol.json'
OUT=ROOT/'exact-low.json.gz'
KNOWN=ROOT/'known-optical.json'
SOURCE=school.registered.ROOT/'registered.json.gz'
mpc=p.mpc


def plans():
    d=_read(SOURCE); known=_read(p.MODEL)
    trained={r['id'] for r in known['rows']}
    old=_read(p.fixture.PLAN); reserved={r['id'] for k in ('development','held') for r in old[k]}
    low=[]; routine=[]
    for i,(c,r) in enumerate(zip(d['cards'],d['rows'])):
        if c['id'] in trained|reserved:continue
        for o in r['placements']['delivered']:
            if not o['near_bounce'] or not o['receive_line']:continue
            h=p.projected_height(o['receive_line']); x=o['receive_line']['xyz'][0]
            shots=[q for q in r['searches'] if q.get('bounced') and -.97<q['p'][1]<-.48 and
                       o['near_bounce']['step']<=q['tick']<(o['near_contact'] or r['events'][-1])['step']-4]
            if not shots or shots[0].get('robust',0)>0:continue
            if o['leg']>0 and not o['near_landed'] and .93<=h<1.0 and abs(x)<.24:save=low
            elif o['near_landed'] and 1.025<h<1.14 and abs(x)<.23:save=routine
            else:continue
            if c['id'] not in {v['id'] for v in save}:
                save.append(dict(id=c['id'],i=i,leg=o['leg'],tick=shots[0]['tick'],stock=bool(o['near_landed']),
                                 height=h,x=x,prior_physically_scored=o['leg']))
    if len(low)<2 or len(routine)<3:raise ValueError('Insufficient untouched-source TT68 natural lows and scored controls')
    _write(PLAN,dict(tt68=_hash(SOURCE),fixture=_hash(p.fixture.GEOMETRY),model=_hash(p.MODEL),cases=low[:6]+routine[:4],
                     role='Reserved TT68 outcomes never train this actor. On first eligible ordinary optical search after the actual bounce, replay all original issued prefix moves and actual previous rally. Include truly sub-1 m ordinary terminal lows after at least one prior earned return plus fixed robustless but actually scored controls. No teacher state, opponent target, or future clocks in the independent camera-and-encoder learner. A no-action move preserves exact stock commands. This diagnostic is not a new full game.'))
    print('exact low docket',[(r['id'],r['leg'],round(r['height'],3),r['stock']) for r in low[:6]+routine[:4]],flush=True)


def get_rates(a,source,ref,stride,lower,upper):
    u=[]; pos=ref.copy()
    for aim in source:
        delta=np.clip((np.clip(aim,lower,upper)-pos)/stride,-.8,.8); pos=np.clip(pos+stride*delta,lower,upper)
        u.append(delta.tolist())
    return u


def run():
    spec=_read(PLAN); doc=_read(SOURCE); f=_read(p.MODEL)
    if spec['tt68']!=_hash(SOURCE) or spec['fixture']!=_hash(p.fixture.GEOMETRY) or spec['model']!=_hash(p.MODEL):
        raise ValueError('Changed exact-held real sequence')
    result=dict(protocol=_hash(PLAN),rows=[],complete=False)
    if OUT.exists():
        old=_read(OUT)
        if old['protocol']!=result['protocol']:raise ValueError('Changed exact low qualification')
        result=old
    rank=p.season.fresh.shot.Coverage(_read(p.season.fresh.RISK)); ready=doc['ready']
    ids={r['id'] for r in spec['cases']}
    with np.load(mpc.BANK) as n: raw=p.fixture.BodyRelativeRaw(mpc.DeepField(n,'local'),school.DX)
    with p.fixture.slid(school.DX),mpc.centered(ready),school.enabled():
        a=mpc.arena(); cad=mpc.arena(width=64,height=48,detail='cad'); camera,rig,stride,lo,hi=mpc.resources(a)
        axes=a._arm_ids[1][2]; guard=school.physical.GripClearance(cad.model,stock=True)
        with patch.object(mpc,'Fragments',lambda b,e:p.course.Carried(b,e,practice=[p.course.PRACTICE],held=ids)):
            actor=repair.Late(mpc.BANK,raw,variant='elastic',budget=28,coverage=rank,school_field=p.Field(f),policy='registered')
        try:
            for z in spec['cases'][len(result['rows']):]:
                c,row=doc['cards'][z['i']],doc['rows'][z['i']]; tick=z['tick']; leg=z['leg']; actions=row['actions']; shot=row['shot']
                obs=prefix(a,c,row,tick,camera,axes,window=52); root=State(a)
                actor.begin(); eye=mpc.Vision(rig,a.control_dt,raw=True); eye.eye=mpc.LearnedOptics(eye.eye,p.course.OPTIC)
                for t,im,q,ref in obs:
                    mpc.observed(eye,t,im,q); actor.engine.observe(t,eye.eye,q,ref.copy())
                q=a.joints.copy(); ref=a.data.ctrl[axes].copy(); x=mpc.observed(eye,tick,camera.capture() if tick%2==0 else None,q)
                actor.engine.observe(tick,eye.eye,q,ref.copy())
                if x is not None and actor.engine.bounced:
                    actor.try_late(tick,x,q,ref)
                baseline=school.trial(a,root,c,actions[:tick],leg,actions[tick:])
                o=row['placements']['delivered'][leg]
                if (baseline['landed']!=z['stock'] or not baseline['offer'] or not same_trace(baseline['offer'],o)
                    or digest_json(baseline['events'])!=digest_json(row['events'][:len(baseline['events'])])):
                    raise ValueError('Exact TT68 reference did not reproduce under the original rules')
                mode=baseline
                if actor.moves:
                    controls=get_rates(a,actor.best[:actor.lease],ref,stride,lo,hi)
                    mode=school.trial(a,root,c,actions[:tick],leg,controls+[school.ZERO]*152,guard)
                    if mode['landed']:
                        mode['full_cad']=school.verify(a,c,shot,mode,guard)
                match=bool(mode['offer'] and all(same_trace(mode['offer'][k],o[k]) for k in ('far_contact','near_bounce')))
                result['rows'].append(dict(case=z,stock=baseline,learned=mode,same_offer=match,camera_model=actor.place_edits,
                    visible=None if x is None else x.tolist(),observed_body=q.tolist(),observed_ref=ref.tolist(),issued=bool(actor.moves)))
                result['complete']=len(result['rows'])==len(spec['cases']);_write(OUT,result)
                print('exact',c['id'],leg,z['stock'],'new',mode['landed'],mode['answered'],'match',match,'optical',actor.place_edits,flush=True)
        finally:actor.close();camera.close();cad.close();a.close()


def known():
    """Isolate camera model error on actual already answered interior commands."""
    cases=[]; ids=set()
    for r in p.physical_paid():
        c=r['card']['id']
        if c not in ids and r['contact']['radial_m']<=.045:
            cases.append(r); ids.add(c)
        if len(cases)>=6:break
    ready=_read(p.QUAL)['ready']; result=dict(model=_hash(p.MODEL),rows=[],scope='Practice-only recognition of known exact physical commands. Exact data appear only in the reported diagnostic, not in the optical engine.')
    with p.fixture.slid(school.DX),mpc.centered(ready),school.enabled():
        a=mpc.arena(); cam,rig,stride,lo,hi=mpc.resources(a); axes=a._arm_ids[1][2]
        engine=OpticalClinic(body=True); oracle=Rollout(a,1)
        try:
            for r in cases:
                c=r['card']; leg=r['leg']; o=r['placements']['delivered'][leg]; u=r['controls']
                tick=min(o['near_bounce']['step']+6,o['near_contact']['step']-6)
                old=dict(r,actions=u); w=prefix(a,c,old,tick,cam,axes,window=48)
                engine.previous=None; engine.bounced=False
                eye=mpc.Vision(rig,a.control_dt,raw=True); eye.eye=mpc.LearnedOptics(eye.eye,p.course.OPTIC)
                for t,im,q,ref in w:
                    mpc.observed(eye,t,im,q); engine.observe(t,eye.eye,q,ref.copy())
                q=a.joints.copy(); ref=a.data.ctrl[axes].copy(); x=mpc.observed(eye,tick,cam.capture() if tick%2==0 else None,q)
                engine.observe(tick,eye.eye,q,ref.copy())
                if x is None or not engine.sync():
                    result['rows'].append(dict(id=c['id'],leg=leg,recognized=False,available=False));continue
                g=[]; pos=ref.copy()
                for ctrl in (u[tick:tick+80]+[school.ZERO]*80)[:80]:
                    pos=np.clip(pos+stride*np.asarray(ctrl),lo,hi);g.append(pos)
                g=np.array([g]); a_err=np.r_[engine.twin.ball-a.ball,engine.twin.velocity-a.velocity]
                measurements={}
                for label,model in (('exact',oracle),('rgb',engine.predict)):
                    feed=np.zeros((1,9)); _tape,y=model.trajectories(g,mpc.AIM,.38,feedback=feed)
                    measurements[label]=dict(legal=bool(np.isfinite(y[0,0])),contact=bool(feed[0,1]>.5),
                                             departure=feed[0,2:5].tolist(),radius=float(feed[0,8]))
                result['rows'].append(dict(id=c['id'],leg=leg,available=True,tick=tick,evaluator_error=a_err.tolist(),
                                           actual_radius=r['contact']['radial_m'],recognition=measurements))
                _write(KNOWN,result)
                print('known',c['id'],leg,measurements,flush=True)
        finally:engine.twin.close();cam.close();a.close()


if __name__=='__main__':
    aa=argparse.ArgumentParser(description=__doc__);aa.add_argument('mode',choices=('plan','run','known')); use=aa.parse_args().mode
    {'plan':plans,'run':run,'known':known}[use]()
