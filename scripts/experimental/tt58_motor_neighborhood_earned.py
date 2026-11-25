"""Test whether rewarding ordinary motor practice populates a transferable local bank."""
import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt58_motor_neighborhood as near

plan = near.plan
clinic = near.clinic
school = near.school
course = near.course
mpc = near.mpc
_read, _write, _hash = near._read, near._write, near._hash
PROTOCOL = plan.ROOT/'earned-protocol.json'
FIT = plan.ROOT/'earned-fit.json.gz'
HOLD = plan.ROOT/'earned-held.json.gz'


def freeze():
    old = _read(near.PRACTICE)
    attempted = _read(near.HELD)
    if not old['complete'] or not attempted['complete'] or attempted['rows']:
        raise ValueError('Do not refit after any held motor outcomes or while the first exercise is running.')
    _write(PROTOCOL,dict(
        base_sha256=_hash(plan.PLAN),scout_sha256=_hash(plan.OUT),
        anchors_sha256=_hash(near.ANCHORS),practice_sha256=_hash(near.PRACTICE),
        closed_first_hold_sha256=_hash(near.HELD),carry_sha256=_hash(near.ACTOR_CARRY),
        controllers=['stock','fixed_seed_bank','practice_earned_bank'],
        acquired_from='All actual legally returned and answered stock practice redraws, plus any already strict-CAD acquired interventions. No new paid motor trials during harvest. Exact whole-tape CAD witness required for ordinary successes. Store RGB flight, current encoders, and actually issued references.',
        entry_onset='Use available causal stereo row nearest the matched anchor elapsed-to-credit for extraction only. Inference waits at the learned stereo y; no application of known hit or target time.',
        comparison='Exactly the same optical nearest-neighbor selector, pose shift, robust stock protection and bounds. The augmented bank only differs in its physically rewarded practice rows. Held verification scored only once. No redraws reused from practice; each retains its original prior stroke.',
        matching='Stock must exactly reproduce the physical scout fingerprint. A treatment candidate differing only within the established native event tolerance is tied to this exact baseline and independently replayed before credit.',
        minimum_distinct_source_neighborhoods=2))


def design():
    d=_read(PROTOCOL)
    files={'base':plan.PLAN,'scout':plan.OUT,'anchors':near.ANCHORS,'practice':near.PRACTICE,
           'closed_first_hold':near.HELD,'carry':near.ACTOR_CARRY}
    if any(d[f'{k}_sha256']!=_hash(v) for k,v in files.items()):
        raise ValueError('A frozen practice, hold or carry artifact changed')
    return d,_read(plan.PLAN),_read(plan.OUT)


def open_case(case):
    p=Path(case['source'])
    doc=_read(p)
    a,b=doc['cards'][case['i']],doc['rows'][case['i']]
    if a!=case['card']:
        raise ValueError('Changed physical source')
    return a,b,np.asarray(case['ready'])


def fit():
    d,original,_=design()
    if FIT.exists():
        raise ValueError('Keep the fixed earned bank')
    practice=_read(near.PRACTICE)
    anchors=_read(near.ANCHORS)['entries']
    by={i['id']:i for i in original['source']}
    rows=[]; acquired=deepcopy(practice['entries'])
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,'local')
    for key in sorted(by):
        item=by[key]; case=item['case']; cut=item['anchor']
        c,old,home=open_case(case)
        win=[r for r in practice['rows'] if r['id']==key and r['arms'].get('stock',{}).get('answered')]
        if not win:
            continue
        with mpc.centered(home),school.enabled(cameras=False):
            a=mpc.arena(width=64,height=48)
            cad=mpc.arena(detail='cad',width=64,height=48)
            actor=near.Receiver(raw)
            clear=clinic.CourtClearance(cad.model,articulated=True)
            try:
                ref=next(r for r in anchors if r['origin']['id']==key)
                for width in win:
                    tape=width['arms']['stock']
                    near.check_offer(tape,width['scout'])
                    proof=clinic.proof(a,home,c,width['seed'],case,cut,old,width['gate'],tape,clear)
                    touch=tape['offer']['near_contact']
                    at=(touch['time']-ref['observed_delay'])/mpc.DT
                    possible=[r for r in tape['log'] if r['x'] is not None and r['x'][4]<-.95
                              and -.94<r['x'][1]<-.52]
                    note=dict(id=key,seed=width['seed'],strict_cad=proof,accepted=False)
                    if possible and not proof['penetration_samples']:
                        nearest=min(possible,key=lambda r:abs(r['tick']-at))
                        entry=near.demonstration(actor,tape,nearest['tick'],touch,
                                                 dict(id=key,seed=width['seed'],stage='ordinary_practice'))
                        if entry is not None:
                            note['accepted']=True
                            acquired.append(entry)
                    rows.append(note)
                    print('earned', key,width['seed'],note['accepted'],proof['samples'],proof['penetration_samples'],flush=True)
            finally:
                actor.close()
                cad.close()
                a.close()
    _write(FIT,dict(protocol_sha256=_hash(PROTOCOL),rows=rows,entries=acquired,complete=True))


def held():
    d,schedule,scout=design()
    fit=_read(FIT)
    if not fit['complete'] or fit['protocol_sha256']!=_hash(PROTOCOL):
        raise ValueError('Do not open fresh held shots before the physically earned bank is fixed')
    got=dict(protocol_sha256=_hash(PROTOCOL),fit_sha256=_hash(FIT),rows=[],complete=False)
    out=_read(HOLD) if HOLD.exists() else got
    if any(out[k]!=got[k] for k in ('protocol_sha256','fit_sha256')):
        raise ValueError('Resume requires the same frozen bank')
    if out['complete']:
        return
    augmented=fit['entries']
    anchors=_read(near.ANCHORS)['entries']
    if len({r['origin']['id'] for r in augmented}) < d['minimum_distinct_source_neighborhoods']:
        out.update(complete=True,status='Not enough CAD-clean newly rewarded practice neighborhoods. Holdout remains unopened.')
        _write(HOLD,out);return
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,'local')
    for item, pool in zip(schedule['source'],scout['rows'],strict=True):
        selected=[r for r in pool['assigned']['held'] if not any(w['seed']==r['seed'] for w in out['rows'])]
        if not selected:
            continue
        case=item['case']; cut=item['anchor']; leg=case['leg']
        c,old,home=open_case(case)
        with mpc.centered(home),school.enabled():
            a=mpc.arena()
            cad=mpc.arena(detail='cad',width=64,height=48)
            clear=clinic.CourtClearance(cad.model,articulated=True)
            res=mpc.resources(a); camera,rig,*_=res
            with patch.object(mpc,'Fragments',lambda file,engine:near.Carried(file,engine,practice=[course.PRACTICE],held=set(plan.ANCHORS))):
                actor=near.Receiver(raw)
            course.resume(actor,near.ACTOR_CARRY.with_suffix('').with_suffix('.npz'),_read(near.ACTOR_CARRY))
            try:
                for w in selected:
                    r=dict(id=item['id'],seed=w['seed'],scout=w,arms={})
                    g=near.start(a,camera,rig,actor,old,c,leg,cut,w['seed'])
                    if g is None:
                        r['status']='No simultaneous RGB gate'
                    else:
                        for name,bank in [('stock',None),('fixed_seed_bank',anchors),('practice_earned_bank',anchors+augmented)]:
                            z=near.trial(a,res,actor,g,w,c,case,home,'bank',bank,eval_only=True)
                            if z.get('unmatched') and clinic.same(z['offer'],r['arms']['stock']['offer']):
                                z.update(unmatched=False, matched_to_stock_within_tolerance=True,
                                         landed=z['actual_landed'],answered=z['actual_answered'])
                            if z['answered'] and z['landed'] and z['selected'] is not None:
                                z['strict_cad']=clinic.proof(a,home,c,w['seed'],case,cut,old,g['time'],z,clear)
                            r['arms'][name]=z
                    out['rows'].append(r);_write(HOLD,out)
                    print('held',item['id'],w['seed'],[(k,v['landed'],v['answered'],
                           v['selected']['owner'] if v['selected'] else None) for k,v in r['arms'].items()],flush=True)
            finally:
                actor.close()
                camera.close()
                cad.close()
                a.close()
    out['complete']=True
    _write(HOLD,out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','fit','held'))
    {'plan':freeze,'fit':fit,'held':held}[p.parse_args().mode]()
