"""Qualify dual-taught paths on fresh physical offers at equal online search cost."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt57_motor_school as school
from . import tt57_motor_school_joint_teacher as teacher
from . import tt57_motor_school_transfer as pupil
from .tt4_transfer_trial import _hash, _read, _write

ROOT=school.ROOT
PLAN=ROOT/'joint-transfer.protocol.json'
OUT=ROOT/'joint-transfer.json.gz'
MODEL=ROOT/'joint-readout.npz'
SOURCES=[school.season.ROOT/f'{stage}.json.gz' for stage in ('before','after24','after48')]
BUDGET=32
mpc=school.mpc


def plan():
    old=_read(school.PLAN)
    tried={r['late_offer_sha256'] for r in [*old['held'],*_read(pupil.CONTROL_PLAN)['rows']]}
    ids={r['id'] for r in old['held']}; reserved=set(); misses=[]; routine=[]
    for src in SOURCES:
        d=_read(src)
        for r in school.course_rows(d,ids,'held'):
            tag=r['late_offer_sha256']; old_row=d['rows'][r['i']]
            if tag in tried or tag in reserved:continue
            s=next(n for n in old_row['searches'] if n['tick']==r['tick'])
            if s.get('robust',0)>0:continue
            misses.append(dict(r,source=str(src),stock=False)); reserved.add(tag)
        for i,(c,r) in enumerate(zip(d['cards'],d['rows'])):
            if c['id'] not in ids:continue
            for o in r['placements']['delivered'][:4]:
                if not o['near_landed'] or not all(o.get(k) for k in ('near_contact','near_bounce','far_contact','receive_line')):
                    continue
                z=school.projected_height(o['receive_line']); x=o['receive_line']['xyz'][0]; tag=school.digest_json(o['near_bounce'])
                if not 1.02<=z<=1.07 or tag in tried|reserved:continue
                t=[n for n in r['searches'] if n.get('p') and -.99<n['p'][1]<-.51
                   and o['far_contact']['step']+3<n['tick']<o['near_contact']['step']-9]
                if not t or t[0].get('robust',0)>0:continue
                lane='left' if x<=-.10 else 'right' if x>=.10 else 'middle'
                routine.append(dict(id=c['id'],i=i,leg=o['leg'],tick=t[0]['tick'],height=z,x=x,lane=lane,
                       source=str(src),late_offer_sha256=tag,stream=c['reply_seed'],stock=True))
                reserved.add(tag)
    rng=np.random.default_rng(571043)
    bad=school.balanced(misses,count=6,per_id=2,rng=rng); good=school.balanced(routine,count=4,per_id=1,rng=rng)
    if len(bad)!=6 or len(good)!=4:raise ValueError('Insufficient distinct fresh vulnerable offers')
    q=dict(sources={str(s):_hash(s) for s in SOURCES},main=_hash(school.PLAN),controls=_hash(pupil.CONTROL_PLAN),rows=bad+good,
           oversample_by_lane=dict(misses=dict(Counter(r['lane'] for r in bad)),routine=dict(Counter(r['lane'] for r in good))),
           search_trials=BUDGET,
           scope='Fresh exact delivers and transparent normal positives disjoint from every originally sealed TT57 outcome and all training releases. Same 32 bounded optical probes and stress sorting in each mode. Only the few training paths replaced by independently qualified dual-teacher actions can differ.')
    _write(PLAN,q)
    print('fresh qualification',[(r['id'],r['lane'],r['stock'],round(r['height'],3)) for r in q['rows']],flush=True)


def fit():
    receipt=_read(teacher.OUT)
    if not receipt['complete'] or receipt['protocol_sha256']!=_hash(teacher.PLAN):raise ValueError('Missing joint practice evidence')
    with np.load(pupil.SCHOOL) as z:
        base={k:z[k].copy() for k in z.files}
    n=0
    for r in receipt['rows']:
        outcome=r['responses'].get('joint',{}); path=outcome.get('issued_relative')
        if path is None:continue
        key=r['key']; allow=base['ids']==r['case']['id']
        matches=np.flatnonzero(allow&np.all(abs(base['keys']-key['features'])<.02,axis=1))
        if len(matches)!=1:raise ValueError('Teacher key no longer matches the old library')
        base['paths'][matches[0]]=path; n+=1
    if n<3:raise ValueError('Not enough optically admissible teaching contexts for transfer')
    base.update(source=_hash(teacher.OUT),replacement_count=n,matched_base=_hash(pupil.SCHOOL))
    np.savez_compressed(MODEL,**base)
    print('qualified path replacements',n,flush=True)


def population(readout,x,key,ref,tick):
    paths,metadata=readout(x,key,ref)
    if not paths:return paths,metadata
    n=len(paths); rng=np.random.default_rng([571044,tick]); paths=list(paths); metadata=list(metadata)
    while len(paths)<BUDGET:
        j=int(rng.integers(n)); k=int(rng.integers(n)); weight=rng.uniform(.03,.27)
        u=paths[j] if len(paths)%3 else (1.-weight)*paths[j]+weight*paths[k]
        p=school.warp(u,rng.uniform(.91,1.08),rng.uniform(-1.63,1.63))
        paths.append(p+school.mutation(rng,rng.uniform(.12,.46)))
        metadata.append(dict(metadata[j],optical_population=True))
    return paths,metadata


def test():
    p=_read(PLAN)
    if p['main']!=_hash(school.PLAN) or any(_hash(Path(n))!=v for n,v in p['sources'].items()):
        raise ValueError('Fresh ordinary docket changed')
    with np.load(MODEL) as d:
        if str(d['matched_base'])!=_hash(pupil.SCHOOL):raise ValueError('Mismatched equal-data training condition')
    models={label:pupil.Readout(f) for label,f in (('exact_teacher',pupil.SCHOOL),('joint_teacher',MODEL))}
    res=dict(protocol_sha256=_hash(PLAN),base_sha256=_hash(pupil.SCHOOL),augmented_sha256=_hash(MODEL),rows=[],complete=False)
    if OUT.exists():
        old=_read(OUT)
        if any(old[k]!=res[k] for k in ('protocol_sha256','base_sha256','augmented_sha256')):
            raise ValueError('Changed persisted control')
        res=old
    cache={}
    with school.enabled():
        cad=mpc.arena(width=64,height=48,detail='cad'); guard=school.CourtClearance(cad.model,articulated=True)
        try:
            for spec in p['rows'][len(res['rows']):]:
                src=spec['source']; i=spec['i']; t=spec['tick']
                if src not in cache:cache[src]=_read(Path(src))
                d=cache[src]; card,row=d['cards'][i],d['rows'][i]
                with mpc.centered(d['ready']):
                    a=mpc.arena(); cam,rig,*_=mpc.resources(a); clinic=pupil.OpticalClinic(body=True)
                    try:
                        eye,x,q,ref,_,key,hist=school.observed_prefix(a,cam,rig,card,row,t,a._arm_ids[1][2]); saved=school.State(a)
                        baseline=school.physical(a,saved,card,row,spec,row['actions'][t:]); n=baseline['stock_prefix_events']
                        if baseline['landed']!=spec['stock'] or school.digest_json(baseline['events'])!=school.digest_json(row['events'][n:n+len(baseline['events'])]):
                            raise ValueError('Original full-law baseline changed')
                        visible=bool(x is not None and key is not None and -1.03<x[1]<-.47 and x[4]<-.9)
                        if visible:pupil.warm_clinic(clinic,rig,a.control_dt,hist,eye,t,q,ref)
                        arms={}
                        for label,bank in models.items():
                            paths,meta=population(bank,x,key,ref,t) if visible else ([],[])
                            movement,evidence=pupil.choose(clinic,paths,meta,ref)
                            if movement is None:
                                answer=baseline
                            else:
                                tape=np.vstack((movement,np.zeros((school.T+94-school.H,6))))
                                answer=school.physical(a,saved,card,row,spec,tape)
                                if answer['landed']:
                                    guarded=school.physical(a,saved,card,row,spec,tape,guard)
                                    if answer['event_sha256']!=guarded['event_sha256']:raise ValueError('Changed physical result under audit')
                                    answer.update(clearance=guarded['clearance'])
                            arms[label]=dict(answer,proposal=evidence,issued=movement is not None)
                        res['rows'].append(dict(case=dict(spec,card=card,ready=d['ready']),stock=baseline,arms=arms,
                                  input=None if not visible else dict(x=x.tolist(),key=key.tolist(),q=q.tolist(),ref=ref.tolist())))
                        res['complete']=len(res['rows'])==len(p['rows']); _write(OUT,res)
                        print(spec['id'],spec['lane'],spec['stock'],[(k,v['proposal'],v['landed'],v['answered']) for k,v in arms.items()],flush=True)
                    finally:
                        clinic.twin.close(); cam.close(); a.close()
        finally:
            cad.close()


def audit():
    from .tt57_motor_school_report import replay_arm

    training,tested=_read(teacher.OUT),_read(OUT)
    if not training['complete'] or not tested['complete']:
        raise ValueError('Cannot audit an incomplete experiment')
    stats=dict(training_tape=_hash(teacher.OUT),qualification_tape=_hash(OUT),
          training=dict(cases=len(training['rows']),cpu_s=sum(r['cpu_s'] for r in training['rows']),
              native_candidates=sum(len(r['batches'])*teacher.R for r in training['rows']),
              original_authorized=sum(r['original']['optical'] and r['original']['votes']>=1 for r in training['rows']),ranking={}),
          fresh=dict(cases=len(tested['rows']),stock_misses=sum(not r['case']['stock'] for r in tested['rows']),
                     stock_hits=sum(r['case']['stock'] for r in tested['rows']),matched={}))
    for mode in ('exact_only','joint'):
        arms=[r['responses'].get(mode,{}) for r in training['rows']]
        stats['training']['ranking'][mode]=dict(exact_models_scored=sum(v.get('exact',False) for v in arms),
               optically_authorized=sum(v.get('optical',False) and v.get('votes',0)>=1 for v in arms),
               actual_scored_answered_clear=sum(bool(v.get('issued_relative')) for v in arms))
    for mode in ('exact_teacher','joint_teacher'):
        stats['fresh']['matched'][mode]=dict(rescues=sum(r['arms'][mode]['landed'] for r in tested['rows'] if not r['case']['stock']),
               protected_hits=sum(r['arms'][mode]['landed'] for r in tested['rows'] if r['case']['stock']),
               issued=sum(r['arms'][mode]['issued'] for r in tested['rows']))
    first=next((r for r in training['rows'] if r['responses'].get('joint',{}).get('issued_relative')),None)
    if first is not None:
        proof=replay_arm(first['case'],first['responses']['joint'])
        if proof['clearance']['penetration_samples']:
            raise ValueError('Whole exact CAD path penetrated')
        stats['first_joint_full_cad']=dict(id=first['case']['id'],**proof)
    _write(ROOT/'joint-summary.json',stats)
    print(stats,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__); a.add_argument('mode',choices=('plan','fit','run','audit')); m=a.parse_args().mode
    {'plan':plan,'fit':fit,'run':test,'audit':audit}[m]()
