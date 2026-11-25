"""Find the physical motor ceiling on actual central low TT37 deliveries."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt40_low_feasibility as native
from . import tt46_continuous_season as season
from . import tt48_practice_transfer as transfer
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt40_long_low import same_trace

ROOT=Path('artifacts/tt54_midline')
PLAN=ROOT/'protocol.json'
RESULT=ROOT/'ceiling.json.gz'
BOUNDS=(-1.01,-1.11,-1.20,-1.29,-1.38,-1.46,-1.54,-1.62,-1.70,-1.78)
LEVEL=(("ordinary_receiving_window",0),("already_separated",1))
EXCLUDED=(14843,4322,8302)
SOURCES=(('tt37/mix-rollin',11846,0),('tt42/split-memory-48',4214,4),
         ('tt42/learning-48',4309,8),('tt42/split-memory-48',4636,4))


def line(o):
    p=np.asarray(o['xyz']); v=np.asarray(o['velocity'])
    t=-.43/(.98*v[1])
    return float(p[0]+t*v[0]),float(p[2]+t*v[2]-4.905*t*t)


def freeze():
    excluded=set(EXCLUDED)
    paths=(season.PLAN,transfer.PLAN,transfer.SOURCE)
    for p in paths:
        doc=_read(p)
        excluded.update(c['id'] for k in ('practice','assessment','cards') for c in doc.get(k,()))
    for p in (Path('artifacts/tt48_bounce_window/plan.json'),Path('artifacts/tt48_low_stage/protocol.json')):
        if p.exists():
            doc=_read(p)
            excluded.update(c['id'] for k in ('cases','release_ids') for c in doc.get(k,()) if isinstance(c,dict) and 'id' in c)
    rows=[]
    for label,ident,leg in SOURCES:
        p=Path('artifacts')/(label+'.json.gz'); doc=_read(p)
        i=next(i for i,c in enumerate(doc['cards'][:len(doc['rows'])]) if c['id']==ident)
        card,row=doc['cards'][i],doc['rows'][i]
        o=row['placements']['delivered'][leg]
        x,z=line(o['receive_line'])
        if ident in excluded or not abs(x)<.10 or z>=1.0 or o['near_landed'] or not o['near_bounce'] or card['profile']!='reach':
            raise ValueError((ident,x,z,excluded))
        if any(r['draw']!='random' for r in row['placements']['requested']):
            raise ValueError('Only the ordinary continuously random TT37 protocol may enter')
        contact=o['far_contact']['time']
        separation=next(e for e in row['events'] if e['kind']=='separation' and e.get('collider')=='rival_racket'
                        and contact<e['time']<contact+.12)
        early=int(np.ceil(separation['time']/mpc.DT+1e-7))+1
        public=next((r['tick'] for r in row.get('searches',()) if r['tick']>=early and r['p'][1]<-.49),None)
        access='first_logged_camera_search'
        if public is None:
            # Older ordinary camera receipts did not log native search. Their
            # near table separation is a separately flagged evaluator clock.
            bounce=o['near_bounce']['time']
            separated=next(e['time'] for e in row['events'] if e['kind']=='separation'
                           and e.get('collider') in ('table','table_right') and bounce<e['time']<bounce+.065)
            public=int(np.ceil(separated/mpc.DT+1e-7))+1
            access='evaluator_only_near_table_separation'
        rows.append(dict(id=ident,index=i,leg=leg,source=str(p),source_sha256=_hash(p),
                         h=z,lat=x,old_arm_touched=o['near_contact'] is not None,times=[public,early],ordinary_clock=access,
                         earlier_opponent_collision_s=separation['time']))
    _write(PLAN,dict(rows=rows,excluded=sorted(excluded),exclusion_sources={str(p):_hash(p) for p in paths},
                     depths_m=BOUNDS,batch=96,rounds=3,physical_attempts_per_stage=3,
                     role='New central-low diagnostic on historical ordinary stock failures. Cases span an opening touch, two later no-touch misses near .98-.99 m, and a severe centered low. Exact normal arm and public window first, then if required earliest matched far separation. Both signed faces and several contact depths may enter a fixed native search. True ball and body data are used by the privileged diagnostic only. Positive must match the original actual incoming hit and bounce and be natively answered on the unchanged opponent. No prospective whole games count.'))
    print(rows,flush=True)


def replay_control(a,old,c):
    a.launch(old['shot'],card=c); a.partner.reset(seed=c['seed'])
    for u in old['actions']:
        a.exchange(u,mpc.FAR_COMMAND)
    if not all(digest_json(o)==digest_json(l) for o,l in ((old['events'],a.events),(old['placements'],a.point_placements()))):
        raise ValueError('Failed received ball and ordinary body control did not exactly replay')


def replay(a,check,old,c,tape,tick,leg):
    prefix=native.replay_prefix(a,old,c,tick)
    began=float(a.data.time)
    guarded(a,check)
    hits=[]
    try:
        with contact_observer(a,hits):
            played=native.trial(a,old,c,tape,tick,leg,already=True)
        if a.clearance_result()['samples']!=round((a.data.time-began)/a.model.opt.timestep):
            raise ValueError('Not all physical suffix states were checked')
        old_offer=old['placements']['delivered'][leg]
        now=played['placements']['delivered'][leg] if leg<len(played['placements']['delivered']) else None
        same=bool(now and same_trace(old_offer['near_bounce'],now['near_bounce']))
        requests=[r['target'] for r in played['placements']['requested']]
        if any(target!=list(draw(c,j,'random',None)[0]) for j,target in enumerate(requests)):
            raise ValueError('Physical continuous opponent rule changed')
        if played['legal'] and not same:
            raise ValueError('A purported return changed the original actual incoming opportunity')
        clearance=a.clearance_result()
        return dict(played,exact_offer=same,native_contacts=hits,suffix_cad=clearance,
                    prefix_events_sha256=prefix,witness=played['answered'] and same and clearance['penetration_samples']==0,
                    shot=old['shot'],cut_tick=tick)
    finally:
        a.sample_clearance(False)


def full_verify(ready,c,winner,check):
    with mpc.centered(ready):
        a=mpc.arena(width=64,height=48)
        try:
            a.launch(winner['shot'],card=c);a.partner.reset(seed=c['seed'])
            guarded(a,check)
            for u in winner['actions']:
                if any(abs(z)>.8+1e-10 for z in u):
                    raise ValueError('Physical winner exceeded stock signed commands')
                a.exchange(u,mpc.FAR_COMMAND)
            if any(digest_json(original)!=digest_json(now) for original,now in ((winner['events'],a.events),
                                                   (winner['placements'],a.point_placements()))):
                raise ValueError('Independent CAD replay diverged')
            verify_clearance(a)
            return a.clearance_result()
        finally:
            a.close()


def evaluate(out):
    plan=_read(PLAN)
    result=_read(out) if out.exists() else dict(plan_sha256=_hash(PLAN),rows=[],complete=False)
    if result['plan_sha256']!=_hash(PLAN):
        raise ValueError('Changed fixed central low practice')
    for r in plan['rows'][len(result['rows']):]:
        path=Path(r['source']);d=_read(path)
        if _hash(path)!=r['source_sha256']:
            raise ValueError('Changed historical command provenance')
        old,c=d['rows'][r['index']],d['cards'][r['index']];ready=d['ready']
        with mpc.centered(ready):
            a=mpc.arena(width=64,height=48)
            cad=mpc.arena(width=64,height=48,detail='cad')
            check=CourtClearance(cad.model,articulated=True)
            stages=[]
            try:
                replay_control(a,old,c)
                coach=native.Probe(a)
                win=None
                for protocol,rank in LEVEL:
                    t=r['times'][rank]
                    hist=native.replay_prefix(a,old,c,t)
                    with patch.object(native,'DEPTHS',tuple(plan['depths_m'])):
                        try:
                            a.clearance_margin=.0006
                            options,stat=coach.search([541117,c['seed'],r['leg'],rank],plan['batch'],plan['rounds'])
                        finally:
                            a.clearance_margin=0.
                    item=dict(role=protocol,tick=t,stockly_issued_q=a.joints.tolist(),
                              stock_prefix_sha256=hist,model=stat,tests=[])
                    stages.append(item)
                    for candidate in options[:plan['physical_attempts_per_stage']]:
                        trial=replay(a,check,old,c,candidate,t,r['leg'])
                        if trial['witness']:
                            trial['full_cad']=full_verify(ready,c,trial,check)
                            if trial['full_cad']['penetration_samples']==0:
                                win=trial
                        item['tests'].append(trial)
                        if win:
                            break
                    print(c['id'],protocol,len(options),[(v['legal'],v['answered'],v['suffix_cad']['penetration_samples']) for v in item['tests']],flush=True)
                    if win:
                        break
            finally:
                a.close();cad.close()
        result['rows'].append(dict(case=r,card=c,baseline_score=old['score']['learner'],
                                   baseline_leg=old['placements']['delivered'][r['leg']],stages=stages,
                                   answered_cad_clean=bool(win)))
        result['complete']=len(result['rows'])==len(plan['rows'])
        _write(out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','run'))
    p.add_argument('--out',type=Path,default=RESULT)
    a=p.parse_args()
    freeze() if a.mode=='plan' else evaluate(a.out)
