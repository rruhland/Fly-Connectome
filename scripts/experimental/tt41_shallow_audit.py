"""Separate CAD validation of practice and a fixed-roster physical comparison."""

import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import PROFILES,arena
from .tt37_opponent_audit import CourtClearance,contact_observer,guarded,verify_clearance
from .tt41_shallow import OUT,launch,touch_band


def native_practice(cal_path,paths,out):
    cal=_read(cal_path); checked=[]
    for p in paths:
        source=_read(p)
        hub=int(source.get('hub_index',0))
        ready=np.r_[cal['hubs'][hub]['q'],cal['home'][-1]]
        demo=next((r for r in source['rows'] if r['shallow'] and r['answered']),None)
        if demo is None:continue
        with centered(ready):
            a=arena(); cad=arena(detail='cad')
            try:
                launch(a,demo['shot'])
                guarded(a,CourtClearance(cad.model,articulated=True))
                contacts=[]
                with contact_observer(a,contacts):
                    for u in demo['actions']:a.exchange(u,FAR_COMMAND)
                if digest_json(a.events)!=digest_json(demo['events']):
                    raise ValueError('Practice CAD replay changed the physical history')
                verify_clearance(a)
                score=audit(a.events,opening_side=-1)
                if score['learner']<1 or score['opponent']<1:
                    raise ValueError('The demonstrated shallow physical return and reply no longer exist')
                checked.append(dict(source=p.name,source_sha256=_hash(p),id=demo['shot']['id'],hub=hub,
                    signed_planned_face=demo['candidate']['sign'],contact=demo['actual_contact']['location'],
                    solver_contacts=contacts,clearance=a.clearance_result(),exact_events=True))
            finally:
                a.close();cad.close()
    _write(out,dict(checked=checked,method='Replay original bounded commands, CAD checks only after selection'))


def compare(old,new,out):
    s,v=_read(old),_read(new)
    if s['cards']!=v['cards'] or len(s['rows'])!=len(s['cards']) or len(v['rows'])!=len(s['cards']):
        raise ValueError('Need a completed unchanged measured roster')
    def grade(row):
        kinds=Counter(o['kind'] for o in row['offers'])
        legal=Counter(o['kind'] for o in row['offers'] if o['actual_legal'])
        issued=[]
        for r in row['searches']:
            if r.get('mode')!='shallow_workspace' or r.get('kind')!='issued':continue
            t=r['tick']/60
            h=next((p for p in row['events'] if p['kind']=='contact' and p.get('side')==1 and t<p['time']<=t+r['contact_eta']+.11),None)
            scored=None if h is None else next((p for p in row['score']['shots'] if p['side']==1 and abs(p['hit']-h['time'])<.012),None)
            answered=bool(scored and any(p['side']==-1 and p['hit']>scored['landing'] for p in row['score']['shots']))
            issued.append(dict(tick=r['tick'],history=r['historical_release'],legal=scored is not None,answered=answered,
                               early=bool(h and scored and touch_band(h)),contact=None if h is None else h['location']))
        return dict(own=row['score']['learner'],returns=row['score']['returns'],incoming=dict(kinds),legal_incoming=dict(legal),
                    early=sum(o['actual_early'] for o in row['offers']),issued=issued,cpu=row['runtime_cpu_s'],
                    recovered=row['planner'].get('outbound_hub_ticks',0))
    pairs=[]
    for c,r,p in zip(s['cards'],s['rows'],v['rows'],strict=True):
        a,b=r['placements']['requested'],p['placements']['requested']
        reply=list(zip(a,b))
        if any(x['target']!=y['target'] for x,y in reply):
            raise ValueError('Independent opponent placement draws changed')
        sa,sb=r['placements']['delivered'],p['placements']['delivered']
        # The physical many-shot sequence may diverge after the learner changes
        # its first actual outgoing ball. Check the common first offer separately.
        first=None if not sa or not sb else all(digest_json(sa[0][key])==digest_json(sb[0][key])
                                               for key in ('target','far_contact','near_bounce'))
        pairs.append(dict(id=c['id'],profile=c['profile'],first_offer_equal=first,
                         shared_opponent_draws=len(reply),stock=grade(r),learned=grade(p)))
    _write(out,dict(stock_sha256=_hash(old),learned_sha256=_hash(new),pairs=pairs))


def body_replay(saved,calibration,out):
    """Read attained encoder branches on fixed physical command tapes."""
    d=_read(saved); cal=_read(calibration)
    if len(d['rows'])!=len(d['cards']):raise ValueError('The full causal run is required')
    hubs={i:np.asarray(cal['hubs'][i]['q']) for i in (0,2)}
    record=[]
    with centered(d['ready']),patch.dict(PROFILES,{'shallow41':d['shallow_profile']}):
        a=arena(); cad=arena(detail='cad')
        try:
            for r,c in zip(d['rows'],d['cards'],strict=True):
                a.launch(r['shot'],card=c); a.partner.reset(seed=c['seed'])
                guarded(a,CourtClearance(cad.model,articulated=True))
                trace=[]
                for u in r['actions']:
                    trace.append(a.joints[:5].copy())
                    a.exchange(u,FAR_COMMAND)
                if digest_json(a.events)!=digest_json(r['events']):
                    raise ValueError('Actuator-only replay changed the actual point')
                verify_clearance(a)
                far=next((e for e in a.events if e['kind']=='contact' and e.get('side')==-1),None)
                checks=[]
                considered=[(s['tick'],s['mode']) for s in r['searches'] if 'mode' in s]
                if far is not None:
                    considered.append((round(far['time']*60),'physical_far_contact'))
                for tick,kind in sorted(considered):
                    if tick>=len(trace):continue
                    q=trace[tick]
                    speed=np.abs(q-trace[max(0,tick-1)])*60
                    diff={str(i):float(np.max(abs(q-h))) for i,h in hubs.items()}
                    checks.append(dict(tick=tick,kind=kind,q=q.tolist(),speed=float(speed.max()),hub_difference=diff))
                record.append(dict(id=c['id'],checks=checks,native_clearance=a.clearance_result(),event_sha256=digest_json(a.events)))
        finally:
            a.close(); cad.close()
    _write(out,dict(source_sha256=_hash(saved),atlas_sha256=_hash(calibration),rows=record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('practice','compare','body'));p.add_argument('out',type=Path)
    p.add_argument('sources',type=Path,nargs='+');p.add_argument('--atlas',type=Path,default=OUT/'shallow-atlas.json.gz')
    a=p.parse_args()
    if a.mode=='practice':native_practice(a.atlas,a.sources,a.out)
    elif a.mode=='compare':compare(*a.sources,a.out)
    else:body_replay(a.sources[0],a.atlas,a.out)
