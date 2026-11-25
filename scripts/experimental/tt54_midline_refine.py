"""Earn a forecourt family on nearby actual low balls from the acquired body."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt37_opponent as opponent
from . import tt54_midline as audit
from . import tt54_midline_course as school
from . import tt51_low_reverse_curriculum as optical
from .tt4_transfer_trial import _hash, _read
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BladeIK
from .tt37_opponent_audit import CourtClearance, guarded, contact_observer
from .tt40_low_feasibility import TapeProbe
from .tt40_long_low import same_trace
from .tt44_paired_credit import State

PLAN=school.ROOT/'refine-protocol.json'
OUT=school.ROOT/'refine.json.gz'


def freeze():
    old=_read(school.RESULT)
    if not old['complete']:
        raise ValueError('Original camera paired controls must finish first')
    p=_read(school.PLAN); c=school.teacher()[0]['card']
    rng=np.random.default_rng(541107)
    held=[]
    while len(held)<4:
        seed=int(rng.integers(1,2**30))
        goal=school.STD(dict(c,reply_seed=seed),p['leg'],'random',None)[0]
        if abs(goal[0]-p['seed_target'][0])<.045 and abs(goal[2]-p['seed_target'][2])<.027:
            held.append(dict(seed=seed,target=list(goal)))
    rows=[dict(index=r['index'],target=r['target'],stock_offer=r['variants']['stock']['offer'])
          for r in old['rows'] if r['stage']=='graded' and r['matched']]
    optical.save(PLAN,dict(paired_course_sha256=_hash(school.RESULT),phase_doc_sha256=_hash(school.PLAN),
        ordinary_nearby_independent=held,rows=rows,batch=120,rounds=4,physical_candidates=2,
        role='Apply a bounded true-state motor coach only during practice from the exact acquired body to the four failed small-neighbor opportunities. Same physical balls and targets as the old failed paired scaffold. Prefer centered physically legal strokes if available, audit any real scored and answered action, and only then produce ordinary RGB/encoder templates. Do not use this coach on the held roster.'))


def calibrated_cap(a,teach,c,plan,check,candidate,cut,old):
    axes=a._arm_ids[1][2];t0=float(a.data.time);events=[];rates=[]
    guarded(a,check)
    try:
        with contact_observer(a,events):
            for i in range(140):
                u=candidate['suffix'][i] if i<len(candidate['suffix']) else [0.]*len(axes)
                rates.append(u)
                _,_,end,capped,_=a.exchange(u,mpc.FAR_COMMAND)
                if end or capped or a.rival_returns>=plan['leg']+2:
                    break
        pp=a.point_placements();leg=plan['leg']
        got=pp['delivered'][leg] if len(pp['delivered'])>leg else None
        same=bool(got and all(same_trace(got[k],old[k]) for k in ('far_contact','near_bounce')))
        clearance=a.clearance_result()
        if clearance['samples']!=round((a.data.time-t0)/a.model.opt.timestep):
            raise ValueError('Incomplete acquired-state strict CAD sampling')
        if not same:
            raise ValueError('Physical training clone changed its matched delivered opportunity')
        own=bool(got['near_landed']);answered=own and len(pp['delivered'])>leg+1
        return dict(landed=own,answered=answered,matched=same,clearance=clearance,rate_suffix=rates,
                    placements=pp,events=a.events,contacts=events,shot=teach['shot'],
                    actions=teach['actions'][:cut]+rates,coach={k:v for k,v in candidate.items() if k!='suffix'})
    finally:
        a.sample_clearance(False)


def camera_packet(a,c,witness,im,rig,stride,lo,hi,cut,name):
    a.launch(witness['shot'],card=c);a.partner.reset(seed=c['seed'])
    eye=mpc.Vision(rig,a.control_dt,raw=True);eye.eye=mpc.LearnedOptics(eye.eye,school.course.OPTIC)
    log=[];axes=a._arm_ids[1][2]
    for t,u in enumerate(witness['actions']):
        q=a.joints.copy();ref=a.data.ctrl[axes].copy()
        x=mpc.observed(eye,t,im.capture() if t%2==0 and t>=cut-42 else None,q)
        if t>=cut-35:
            log.append(dict(tick=t,x=None if x is None else x.tolist(),q=q.tolist(),ref=ref.tolist(),
                next_ref=np.clip(ref+stride*np.asarray(u),lo,hi).tolist(),
                bounce=bool(eye.eye.latest and eye.eye.latest.bounced)))
        a.exchange(u,mpc.FAR_COMMAND)
    if digest_json(a.events)!=digest_json(witness['events']):
        raise ValueError('Observation capture changed the newly rewarded direct stroke')
    record=optical.extract(log,BladeIK(a),name,cut-3)
    if record is not None:
        try:
            record['p']=school.cue_prototype(log,record)
        except ValueError:
            record=None
    return record,log


def run(out):
    conf=_read(PLAN); p=_read(school.PLAN); row,teach,ready,old=school.teacher();c=row['card'];leg=p['leg'];cut=p['cut']
    if conf['paired_course_sha256']!=_hash(school.RESULT) or conf['phase_doc_sha256']!=_hash(school.PLAN):
        raise ValueError('Changed physical coaching roster')
    result=_read(out) if out.exists() else dict(plan_sha256=_hash(PLAN),rows=[],templates=[],complete=False)
    if result['plan_sha256']!=_hash(PLAN):
        raise ValueError('Different scored forecourt family')
    with mpc.centered(ready):
        a=mpc.arena();cad=mpc.arena(detail='cad');clear=CourtClearance(cad.model,articulated=True)
        im,rig,stride,lo,hi=mpc.resources(a)
        coach=TapeProbe(a,steering=True)
        try:
            for i,s in enumerate(conf['rows'][len(result['rows']):],start=len(result['rows'])):
                goal=s['target']
                def request(card,j,protocol,rng):
                    if j==leg:
                        return tuple(goal),'tt54_declared_local_practice'
                    return school.STD(card,j,protocol,rng)
                with patch.object(opponent,'draw',request):
                    a.launch(teach['shot'],card=c);a.partner.reset(seed=c['seed'])
                    for u in teach['actions'][:cut]:
                        _,_,end,cap,_=a.exchange(u,mpc.FAR_COMMAND)
                        if end or cap:
                            raise ValueError('Original shared camera handoff no longer reached')
                    state=State(a)
                    try:
                        a.clearance_margin=.0006
                        choices,stats=coach.search([540673,c['seed'],i],conf['batch'],conf['rounds'],teach,cut,leg)
                    finally:
                        a.clearance_margin=0.
                    choices.sort(key=lambda g:(g['feedback'][8]>.064,g['cost']+8*g['feedback'][8]))
                    note=dict(index=s['index'],goal=goal,coach=stats,attempts=[],rewarded=False)
                    for candidate in choices[:conf['physical_candidates']]:
                        state.restore(a);a.clearance_margin=0.
                        tryout=calibrated_cap(a,teach,c,p,clear,candidate,cut,s['stock_offer'])
                        note['attempts'].append(tryout)
                        if not tryout['answered'] or tryout['clearance']['penetration_samples']:
                            continue
                        tryout['full_cad']=audit.full_verify(ready,c,tryout,clear)
                        if tryout['full_cad']['penetration_samples']:
                            continue
                        sample,log=camera_packet(a,c,tryout,im,rig,stride,lo,hi,cut,f'forecourt-learned-{s["index"]}')
                        tryout['camera_log']=log
                        if sample is not None:
                            result['templates'].append(optical.plain(sample));note['rewarded']=True
                        break
                result['rows'].append(note)
                optical.save(out,result)
                print(s['index'],[(z['legal'],z['contacts']) for z in stats['rounds']],
                      [(z['landed'],z['answered'],z.get('full_cad',{}).get('penetration_samples'),z['coach']['feedback'][8])
                       for z in note['attempts']],note['rewarded'],flush=True)
        finally:
            im.close();a.close();cad.close()
    result['complete']=True
    optical.save(out,result)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('plan','run'));a.add_argument('--out',type=Path,default=OUT)
    p=a.parse_args();freeze() if p.mode=='plan' else run(p.out)
