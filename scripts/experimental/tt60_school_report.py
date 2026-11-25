"""Independently rescore the historical and earned data underlying the TT60 report."""

from pathlib import Path
from collections import Counter

import numpy as np

from . import tt57_motor_school_mature as maturity
from . import tt60_school_bottleneck as study
from . import tt60_school_optical_gate as gate
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit


OUT = study.ROOT / 'physical-check.json'


def verify():
    history, factor, trained, votes = map(_read,(study.HISTORY,study.FACTOR,maturity.OUT,gate.OUT))
    if not all(v['complete'] for v in (factor,trained,votes)):
        raise ValueError('Incomplete physical or recognition docket')
    checks = {study.ORDINARY:history['sha256'], maturity.OUT:factor['source_sha256'],
              Path(study.__file__):factor['operator'], Path(gate.__file__):votes['operator'],
              study.FACTOR:votes['calibration'], gate.MODEL:votes['model'],
              gate.OUT:_read(gate.REPORT)['gate_sha256']}
    if any(_hash(p)!=h for p,h in checks.items()):
        raise ValueError('Changed physical or optical provenance')
    saved = _read(study.ORDINARY)
    counters = Counter(); heights = []; abs_x = []; endings=Counter(); widths=[]
    for expected,c,r in zip(history['rows'],saved['cards'],saved['rows'],strict=True):
        m = audit(r['events'],opening_side=1)
        returned = m['learner']
        if (c['id'],returned)!=(expected['id'],expected['score']) or returned!=sum(
            bool(v.get('near_landed')) for v in r['placements']['delivered']):
            raise ValueError('Historical score does not match its native referee events')
        if expected['class_']!='receive_no_contact':
            continue
        o=r['placements']['delivered'][-1]; f=o['receive_line']
        endings[expected['terminal']]+=1
        counters['no_contact']+=1
        if f is None:
            continue
        counters['has_receive_line']+=1
        z=float(study.school.projected_height(f)); x=float(f['xyz'][0])
        heights.append(z); abs_x.append(abs(x))
        prior=[s for s in r['searches'] if s['tick']>o['far_contact']['step']+3]
        counters['has_search']+=bool(prior)
        counters['no_legal_forecast']+=not any(s.get('legal_trials',0)>0 for s in prior)
        counters['no_robust_forecast']+=not any(s.get('robust',0)>0 for s in prior)
        counters['below_1m']+=z<1.
        counters['at_or_above_1_03m']+=z>=1.03
        counters['center_abs_x_below_10cm']+=abs(x)<.10
        counters['abs_x_above_30cm']+=abs(x)>.30
        widths.append(dict(id=c['id'], height_m=z, x_m=x, score=returned, looks=len(prior)))
    radius = []; win_sets = []; net_rejections = []
    for r, t, g in zip(factor['rows'],trained['rows'],votes['rows'],strict=True):
        if (r['id'],r['tick'])!=(t['case']['id'],t['case']['tick']) or (r['id'],r['tick'])!=(g['id'],g['tick']):
            raise ValueError('Changed application context')
        if g.get('new') is not None and g['id'] in g['new']['donor_releases']:
            raise ValueError('The same measured release contributed to its own estimate')
        alternatives=[t['stock']]+[a for a in t['arms'] if a.get('controls')]
        for i,(a,b) in enumerate(zip(alternatives,r['tapes'],strict=True)):
            if a['event_sha256']!=b['event_sha256'] or bool(a['landed'])!=bool(b['landed']):
                raise ValueError('Unmatched physically issued movement')
            winner = any(e['kind']=='legal_return' and e.get('side')==1 for e in a['events'])
            if winner!=b['landed']:
                raise ValueError('Physically scored label disagrees with native referee suffix')
            if not winner:
                continue
            if not a['matched_far_hit'] or not a['matched_near_bounce'] or a['clearance']['penetration_samples']:
                raise ValueError('A learned physical winner changed the incoming or violated clearance')
            radius.append(a['actual']['radius'])
            status=dict(id=r['id'],tick=r['tick'],answer=a['answered'],
                        net=any(e['kind']=='net' for e in a['events']),
                        exact_legal=r['arms']['exact_native'][i]['legal'],radius_m=radius[-1])
            if not status['exact_legal'] and status['net']:
                net_rejections.append(status)
            win_sets.append(status)
    result=dict(checked_hashes={str(p):v for p,v in checks.items()}, histories=len(history['rows']),
                no_contact=dict(counts=dict(counters),terminal_reason=dict(endings),
                                heights_m=np.quantile(heights,[0,.5,1]).tolist(),max_abs_x_m=max(abs_x),rows=widths),
                earned=dict(rows=win_sets,net_native_rejections=net_rejections,
                            observed_radius_m=np.quantile(radius,[0,.5,1]).tolist(),
                            observed_radius_74_to_76mm=sum(.074<=r<=.076 for r in radius)))
    _write(OUT,result)
    print('verified',result['histories'],len(win_sets),len(net_rejections),dict(counters),flush=True)


if __name__=='__main__':
    verify()
