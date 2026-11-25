"""Separate native screening, exact practice yield, and optical transfer."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt57_motor_school as school
from . import tt57_motor_school_transfer as transfer
from .tt4_transfer_trial import _hash, _read, _write
from .tt48_contact_patch_report import replay

OUT=school.ROOT/'report.json'


def by_lane(cases):
    return dict(Counter(r['case']['lane'] for r in cases))


def result_count(practice):
    cases=practice['rows']
    tested=[a for r in cases for a in r.get('arms',[])]
    successful=[(r,a) for r in cases for a in r.get('arms',[]) if a.get('issued_relative')]
    unique=list({r['case']['late_offer_sha256']:r for r,a in successful}.values())
    answered=list({r['case']['late_offer_sha256']:r for r,a in successful if a['answered']}.values())
    feedback=Counter(
        ('native_legal' if v['native_legal'] else 'native_miss',
         'actual_scored' if v['landed'] else 'actual_missed') for v in tested)
    native=[v for r in cases for v in r.get('native',[])]
    radii=[(r['stock']['actual']['radius'],a['actual']['radius']) for r,a in successful]
    return dict(cases=len(cases),distinct_releases=len({r['case']['id'] for r in cases}),
                cpu_s=sum(r['cpu_s'] for r in cases),complete=practice['complete'],
                physical_stock_scored=sum(r['stock']['landed'] for r in cases),
                native_trials=sum(v['trials'] for v in native),native_legal=sum(v['legal'] for v in native),
                native_legal_inside_62mm=sum(v['legal_inside_62mm'] for v in native),
                native_found_cases=sum(any(a['legal'] for a in r.get('native',[])) for r in cases),
                shortlist_physical_arms=len(tested),
                shortlist_prediction_counts={f'{a}_{b}':n for (a,b),n in feedback.items()},
                shortlist_same_delivery=sum(not a.get('unmatched') for a in tested),
                physical_scored=len(unique), physical_scored_by_lane=by_lane(unique),
                physical_scored_releases=len({r['case']['id'] for r in unique}),
                scored_answered=len(answered),scored_answered_by_lane=by_lane(answered),
                scored_answered_releases=len({r['case']['id'] for r in answered}),
                self_side_score_faults=sum(a['landed'] and a.get('clearance',{}).get('penetration_samples',0)>0 for a in tested),
                observed_median_radius_mm=None if not radii else np.round(1000*np.median(radii,axis=0),3).tolist(),
                dq_unchanged_prefix='Original control and events reconstructed from launch for every fork')


def replay_arm(case,arm):
    original=_read(Path(case['source']))['rows'][case['i']]
    tape=dict(shot=original['shot'], actions=original['actions'][:case['tick']]+arm['controls'],
              events=original['events'][:arm['stock_prefix_events']]+arm['events'],score=arm['score'])
    if school.digest_json(tape['events'])!=arm['event_sha256']:
        raise ValueError('Incorrect matched branch event reconstruction')
    with school.enabled():
        return replay(case['ready'],case['card'],tape)


def audit_first_by_lane(practice):
    done={}
    for r in practice['rows']:
        case=r['case']; lane=case['lane']
        if lane in done:
            continue
        arm=next((a for a in r.get('arms',[]) if a['answered'] and a.get('issued_relative')),None)
        if arm is not None:
            done[lane]=dict(id=case['id'],height=case['height'],stock=replay_arm(case,r['stock']),new=replay_arm(case,arm))
    return done


def run(output, verify):
    plan, practice=map(_read,(school.PLAN,school.OUT))
    if practice['protocol_sha256']!=_hash(school.PLAN):
        raise ValueError('Wrong practice docket')
    report=dict(protocol=_hash(school.PLAN),practice_tape=_hash(school.OUT),
                practice=result_count(practice),pilot=result_count(dict(practice,rows=practice['rows'][:8],complete=len(practice['rows'])>=8)))
    if transfer.maturity.OUT.exists():
        mature=_read(transfer.maturity.OUT)
        report['matured_practice_tape']=_hash(transfer.maturity.OUT)
        report['matured_practice']=result_count(mature)
        report['matured_practice']['extensions']=mature['extensions']
        practice=mature
    for label,path in (('held',transfer.HELD),('routine',transfer.CONTROL_OUT)):
        if not path.exists():continue
        held=_read(path); rows=held['rows']; report[f'{label}_tape']=_hash(path)
        roster=plan['held'] if label=='held' else _read(transfer.CONTROL_PLAN)['rows']
        if roster[:len(rows)]!=[{k:v for k,v in r['case'].items() if k not in ('ready','card')} for r in rows]:
            raise ValueError('Changed held comparison rows')
        report[label]=dict(cases=len(rows),complete=held['complete'],visible=sum(r['optical'] for r in rows),
                            stock_preserved_before_readout=sum(r['stock_robust']>0 for r in rows),
                            stock=sum(r['stock']['landed'] for r in rows),modes={})
        for k in ('school','atlas'):
            a=[r['arms'][k] for r in rows]
            report[label]['modes'][k]=dict(supported=sum(v['decision']['attempts']>0 for v in a),
                 nominal_legal=sum(v['decision']['legal']>0 for v in a),issued=sum(v['issued'] for v in a),
                 actual_scored=sum(v['landed'] for v in a),actual_answered=sum(v['answered'] for v in a),
                 verified_clear=sum(v.get('clearance',{}).get('penetration_samples')==0 for v in a),
                 wrong_offer=sum(v.get('unmatched',False) for v in a))
        if verify and label=='held':
            first=next((r for r in rows if r['arms']['school']['landed'] and r['arms']['school'].get('clearance',{}).get('penetration_samples')==0),None)
            if first:
                report['first_held_rescue_full_cad']=dict(id=first['case']['id'],stock=replay_arm(first['case'],first['stock']),
                                                         new=replay_arm(first['case'],first['arms']['school']))
    if transfer.DIAG.exists():
        d=_read(transfer.DIAG)['rows']
        report['optical_witnesses']=dict(eligible=len(d),native_true=sum(v['exact_native_legal'] for v in d),
              optical_nominal=sum(v['observed_native']['legal']>0 for v in d),optical_issued=sum(v['would_issue'] for v in d))
    if transfer.GAMES.exists():
        a=_read(transfer.GAMES); report['games_tape']=_hash(transfer.GAMES)
        report['games']=dict(stock=[r['stock_score'] for r in a['rows']],school=[r['new']['score']['learner'] for r in a['rows']],
                             issued=sum(v['issued'] for r in a['rows'] for v in r['new']['school']),
                             exact_offer_changes=[dict(o,id=r['id']) for r in a['rows'] for o in r['new']['matched'] if o['stock']!=o['learned']])
    if verify:
        report['first_answered_full_cad_by_lane']=audit_first_by_lane(practice)
    _write(output,report)
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--verify',action='store_true')
    a=p.parse_args(); run(a.out,a.verify)
