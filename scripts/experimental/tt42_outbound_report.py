"""Audit physical payoffs and cue coverage without exposing evaluator state to the actor."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt39_opportunity_map import projected_height
from .tt40_long_low import same_trace
from .tt42_outbound_habit import OUT, STOCK


def answered(delivered, k):
    return bool(delivered[k]['near_landed'] and k+1<len(delivered) and delivered[k+1]['near_bounce'])


def outcome(delivered):
    return dict(offered=sum(bool(r['near_bounce']) for r in delivered),
                touched=sum(bool(r['near_contact']) for r in delivered),
                legal=sum(bool(r['near_landed']) for r in delivered),
                answered=sum(answered(delivered,k) for k in range(len(delivered))))


def prepare_due(d, when):
    offers=[(j,r) for j,r in enumerate(d) if r['far_contact']['time']>when+1e-8 and r['near_bounce']]
    return next(iter(offers), (None,None))


def summarize(result_path, output):
    result=_read(result_path)
    control=_read(STOCK)
    if result['cards']!=control['cards']:
        raise ValueError('Changed score assignments')
    total={arm:dict(n=0,legal=0,first=0,later=0,answered=0,offers=0,strict_legal=0,strict_answered=0,
                    clean_points=0,longest_own=0,longest_alternating=0) for arm in ('stock','habit')}
    latest=[]
    triggers=[]
    margins=[]
    gains=[0,0,0]
    for i,row in enumerate(result['rows']):
        old=control['rows'][i]
        c=control['cards'][i]
        dd,tt=old['placements']['delivered'],row['placements']['delivered']
        if digest_json(audit(row['events'],opening_side=1))!=digest_json(row['counts']):
            raise ValueError('Own score does not reproduce from physical events')
        for arm,rr,offers,ruling in (('stock',old['score'],dd,row['stock_clearance']),
                                    ('habit',row['counts'],tt,row['clearance'])):
            dst=total[arm]
            n=rr['learner']
            out=outcome(offers)
            if n!=out['legal'] or out['offered']!=rr['opponent']:
                raise ValueError('Placement and standalone referee disagree')
            clean=ruling['penetration_samples']==0
            dst['n']+=1
            dst['legal']+=n
            dst['first']+=int(n>0)
            dst['later']+=max(0,n-1)
            dst['answered']+=out['answered']
            dst['offers']+=out['offered']
            dst['clean_points']+=clean
            dst['strict_legal']+=n*clean
            dst['strict_answered']+=out['answered']*clean
            dst['longest_own']=max(dst['longest_own'],n)
            dst['longest_alternating']=max(dst['longest_alternating'],rr['returns'])
        diff=int(np.sign(row['counts']['learner']-old['score']['learner']))
        gains[1+diff]+=1
        for r in row['prep']:
            kick=r['tick']/60.
            j,off=prepare_due(tt,kick)
            pri=[e for e in row['events'] if e['kind']=='separation' and
                 e.get('collider')=='learner_racket' and e['time']<=kick+1e-8]
            pre=None if not pri else kick-pri[-1]['time']
            touch=None if off is None else bool(off['near_contact'])
            h=None if off is None or not off['receive_line'] else projected_height(off['receive_line'])
            lead=None if off is None else off['far_contact']['time']-kick
            match=(off is not None and j<len(dd) and same_trace(off['far_contact'],dd[j]['far_contact'])
                   and same_trace(off['near_bounce'],dd[j]['near_bounce']))
            stamped=dict(release_id=c['id'],tick=r['tick'],lead_to_far_s=lead,since_prior_near_separation_s=pre,
                         leg=j,height_projected_m=h,issued=r['issued'],ending=r.get('release','point_end'),
                         matched_physical_offer=match,touched=touch,habit_legal=None if off is None else bool(off['near_landed']),
                         stock_legal=bool(dd[j]['near_landed']) if match else None)
            triggers.append(stamped)
            if match and off['near_landed'] and not dd[j]['near_landed']:
                margins.append(stamped)
        latest.append(dict(id=c['id'],stock_legal=old['score']['learner'],habit_legal=row['counts']['learner'],
                           prep_count=len(row['prep']),full_CAD_clean=row['clearance']['penetration_samples']==0))
    actual=[r for r in triggers if r['lead_to_far_s'] is not None]
    labels=[r['height_projected_m'] for r in actual if r['height_projected_m'] is not None]
    score_delta=np.array([r['habit_legal']-r['stock_legal'] for r in latest])
    interval=None
    if len(score_delta):
        draws=np.random.default_rng(421927).choice(score_delta,size=(8000,len(score_delta)))
        interval=np.percentile(draws.mean(axis=1),[2.5,97.5]).tolist()
    cue=dict(calls=len(triggers),followed_by_legal_opponent_offer=len(actual),with_height_label=len(labels),
             below_099_m=sum(v<.99 for v in labels),below_095_m=sum(v<.95 for v in labels),
             no_observed_prior_inner_separation=sum(r['since_prior_near_separation_s'] is None for r in triggers),
             median_ms_before_far_hit=None if not actual else float(np.median([r['lead_to_far_s']*1000 for r in actual])))
    doc=dict(complete=result['complete'],source_sha256={f.name:_hash(f) for f in (result_path,STOCK)},
             totals=total,paired_games=dict(loss=gains[0],tie=gains[1],gain=gains[2]),triggers=triggers,
             extra_own_per_assigned_game=None if not latest else float(score_delta.mean()),
             paired_release_bootstrap_95=interval,cue=cue,same_offer_rescue=margins,rows=latest)
    _write(output,doc)
    print(total,'paired',gains,'preps',len(triggers),'same-offer rescues',len(margins),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=OUT/'outbound-habit-live.json.gz')
    p.add_argument('--output',type=Path,default=OUT/'outbound-habit-summary.json')
    a=p.parse_args()
    summarize(a.source,a.output)
