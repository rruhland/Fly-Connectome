"""Report calibration rejection, actual held gains, and an independent known pattern."""

from collections import Counter

import numpy as np
from scipy.special import ndtr

from . import tt81_conditional_preparation as train
from . import tt81_preparation_play as play
from . import tt68_clamp_transfer as old
from .tt4_transfer_trial import _hash, _read, _write


def grade(row):
    l=row['placements']['delivered']; scored=bool(l and l[0]['near_landed'])
    return scored,bool(scored and len(l)>1 and l[1]['near_bounce'])


def movement(reference,candidate):
    bounced=reference['placements']['delivered'][0]['near_bounce']['step']
    m=min(bounced,len(reference['controls']),len(candidate['controls']))
    a=np.asarray(reference['controls'][:m]); b=np.asarray(candidate['controls'][:m]); diff=b-a
    return dict(differing_prebounce_controls=int(np.any(diff!=0.,axis=1).sum()),
                peak_prebounce_signed_rate_change=float(np.max(np.abs(diff))),
                issued_prep_ticks=candidate['prep_ticks'])


def contact_changes(plan,held):
    choices=[]
    for r in held['rows']:
        a=r['arms']['registered']; b=r['arms']['prepare']; c=r['case']
        if not a['matched'] or not b['matched']:continue
        a0=a['placements']['delivered'][0]['near_contact']; b0=b['placements']['delivered'][0]['near_contact']
        if bool(a0)==bool(b0):continue
        choices.append((c,b,b0,'prepare') if b0 else (c,a,a0,'registered'))
    out=[]
    s=train.school
    if not choices:return out
    with s.fixture.slid(s.DX),s.mpc.centered(plan['ready']),s.enabled():
        a=s.mpc.arena(width=64,height=48); cad=s.mpc.arena(width=64,height=48,detail='cad')
        guard=s.physical.GripClearance(cad.model,stock=True)
        try:
            for c,r,impact,arm in choices:
                measured=[]
                with s.contact.physical_contacts(a,measured):
                    clear=s.verify(a,c['card'],r['shot'],r,guard)
                x=min((x for x in measured if x['side']==1),key=lambda x:abs(x['time_s']-impact['time']))
                if abs(x['time_s']-impact['time'])>1e-7:raise ValueError('Actual contrast contact time changed')
                separation=next((e for e in r['events'] if e.get('collider')=='learner_racket' and e['time']>=impact['time']),None)
                out.append(dict(id=c['card']['id'],stratum=c['grade'],arm_making_contact=arm,contact=x,
                    separation_velocity=None if separation is None else separation['velocity'],
                    incoming=impact['incoming'],first_scored=grade(r)[0],strict_cad=clear))
        finally:cad.close();a.close()
    return out


def cv(model,rows):
    h=np.array([d['true_h'] for d in rows]); x=np.array([d['features'] for d in rows]); ids=np.array([d['id'] for d in rows])
    f=(x-np.asarray(model['center']))/np.asarray(model['scale']); raw=x[:,1]; adjusted=raw.copy()
    for release in np.unique(ids):
        mask=ids==release; w=train.ridge(f[~mask],h[~mask]-raw[~mask]); adjusted[mask]+=np.c_[np.ones(mask.sum()),f[mask]]@w
    out={}
    for label,depth in (('raw',raw),('corrected',adjusted)):
        p=ndtr((train.LOW_THRESHOLD-depth)/model['spread']); want=h<train.LOW_THRESHOLD
        out[label]=dict(leave_release_out_rmse_m=float(np.sqrt(np.mean((depth-h)**2))),
                        brier=float(np.mean((p-want)**2)),admitted_true_low=int(np.sum((p>=.38)&want)),
                        admitted_nonlow=int(np.sum((p>=.38)&~want)))
    return out


def known():
    proof=old.ROOT/'random-low-12-4.json';tape=old.ROOT/'random-low-12-4-tape.json.gz';data=_read(tape); p=_read(proof)
    if p['card']['id']!=10544 or p['tape_sha256']!=_hash(tape):raise ValueError('Changed existing independently verified known pattern')
    actual=data['rows'][0]; offer=actual['placements']['delivered'][4]; b=offer['near_bounce']['step']; hit=offer['near_contact']['step']
    if not (offer['near_landed'] and p['contacts'][4]['landed'] and p['clearance']['penetration_samples']==0):
        raise ValueError('Existing known point requires true scored/clean evidence')
    return dict(id=10544,role='independent benchmark diagnostic only, never learned or used in held counts',
                artifact_hash=_hash(proof),tape_hash=_hash(tape),far_tick=offer['far_contact']['step'],bounce_tick=b,actual_tick=hit,
                contact=p['contacts'][4],distance_from_registered_ready_rad=(np.asarray(p['contacts'][4]['joints_rad'])-
                      np.asarray(data['ready'])[:5]).tolist(),
                causal_searches=[r for r in actual['searches'] if b<=r['tick']<hit])


def report():
    m=_read(train.MODEL); plan=_read(train.PLAN); c=_read(train.ROWS)
    result=dict(protocol=_hash(train.PLAN),model=_hash(train.MODEL),frames_labelled=m['samples'],unique_calibration_releases=len(m['release_ids']),
                earned_motion_preparations=len(m['keys']),depth_calibration=cv(m,c['features']),known_10544=known(),
                pregrade=dict(Counter(r['grade'] for r in plan['held'])))
    if train.OUT.exists():
        d=_read(train.OUT); result['held_tape']=_hash(train.OUT)
        result['held']=dict(completed=d['complete'],n=len(d['rows']),arms={},movements=[])
        for r in d['rows']:
            a=r['arms']['registered']; b=r['arms']['prepare']; c=r['case']
            if a['matched'] and b['matched']:
                result['held']['movements'].append(dict(id=c['card']['id'],stratum=c['grade'],
                    **movement(a,b),contact=[bool(v['placements']['delivered'][0]['near_contact']) for v in (a,b)]))
        if d['complete']:result['held']['contact_changes']=contact_changes(plan,d)
        for key in ('registered','late','prepare'):
            a={}
            for typ in ('low','routine'):
                rows=[r['arms'][key] for r in d['rows'] if r['case']['grade']==typ]
                a[typ]=dict(scored=sum(grade(v)[0] for v in rows),answered=sum(grade(v)[1] for v in rows),
                            received=len(rows),identical=sum(v['matched'] for v in rows),prep=sum(v['prep_ticks']>0 for v in rows),
                            screened=sum(z['paths'] for v in rows for z in v['late']),
                            late_issued=sum(z['issued'] for v in rows for z in v['late']),
                            clearance_penetrations=sum(v.get('clearance',{}).get('penetration_samples',0) for v in rows))
            result['held']['arms'][key]=a
        if d['complete']:result['held']['promotion']=play.promotion()
    if train.GAMES.exists():
        d=_read(train.GAMES);result['whole_tape']=_hash(train.GAMES)
        result['whole']=dict(complete=d['complete'],scores={k:[r['arms'][k]['score']['learner'] for r in d['rows']]
                                                        for k in ('registered','prepare')})
    _write(train.ROOT/'report.json',result); print(result,flush=True)


if __name__=='__main__':report()
