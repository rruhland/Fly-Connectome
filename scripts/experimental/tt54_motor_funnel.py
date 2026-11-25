"""Audit physically missing deliveries for loss through the ordinary motor-search funnel."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _read, _write, _hash
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt37_opponent import draw
from .tt48_rim_recovery import STOCK
from .tt44_paired_credit import State
from .tt54_executable_seeds import ParallelDiagnostic, Recentered

ROOT=Path('artifacts/tt54_motor_funnel')
PLAN=ROOT/'protocol.json'
RECENTER_PLAN=ROOT/'recenter-protocol.json'
RECEIPT=ROOT/'funnels.json.gz'
STREAM=Path('artifacts/tt48_transfer/live-plain.json.gz')


def protocol():
    doc=_read(STOCK)
    selected=[]
    for i,(card,row) in enumerate(zip(doc['cards'],doc['rows'],strict=True)):
        if card['id'] in {r['id'] for r in selected}:
            continue
        scopes=[]
        for o in row['placements']['delivered'][-2:]:
            if o['near_bounce'] is None or o['receive_line'] is None:
                continue
            start=o['far_contact']['step']
            stop=o['near_contact']['step'] if o['near_contact'] else len(row['actions'])
            due=[s for s in row['searches'] if start<s['tick']<stop and s['p'][1]<-0.73]
            if not due:
                continue
            scopes.append(dict(leg=o['leg'],success=o['near_landed'], ticks=[s['tick'] for s in due[-5:]],
                               h=projected_height(o['receive_line']), real_contact=o['near_contact'] is not None))
        if not any(not v['success'] for v in scopes):
            continue
        selected.append(dict(id=card['id'],index=i,scopes=scopes))
        if len(selected)==8:
            break
    _write(PLAN,dict(rows=selected,source=str(STOCK),source_sha256=_hash(STOCK),
                    carry_sha256=_hash(season.CARRY),rule='First eight distinct life-cycle finishes in fixed official order with a bounced missed final offer and a public late stock search. Each adjacent preceding valid delivery, if present, is a positive control. Diagnostic replay plus evaluator-only hypothetical rollouts. No new rally scores count.'))
    print([(r['id'],[(s['leg'],s['success'],s['ticks']) for s in r['scopes']]) for r in selected],flush=True)


def fingerprint(goal):
    return np.asarray(goal).tobytes()


def official_rates(source=STREAM):
    doc=_read(source)
    results=[]
    for c,r in zip(doc['cards'],doc['rows'],strict=True):
        controls=np.array(r['actions'])
        for i,offer in enumerate(r['placements']['delivered']):
            if not offer['near_bounce'] or not offer['receive_line']:
                continue
            stop=offer['near_contact']['step'] if offer['near_contact'] else len(controls)
            samples=[s for s in r['searches'] if offer['far_contact']['step']<s['tick']<stop]
            late=[s for s in samples if s['p'][1]<-.73]
            last=late[-1]['tick'] if late else None
            landed=offer['near_landed']
            selected=[z for z in late if z['chosen']]
            lo=offer['near_bounce']['step']
            if stop>lo and min(lo,stop-12)>=0:
                u=controls[max(lo,stop-12):stop,:5]
                d=dict(rate_sat_fraction=(abs(u)>=.8-1e-8).mean(axis=0).tolist(),
                       rate_90th_abs=np.quantile(abs(u),.9,axis=0).tolist(),
                       total_reverse_axis=((u[1:]*u[:-1]<-.005)).sum(axis=0).tolist())
            else:
                d={}
            results.append(dict(id=c['id'],leg=i,landed=landed,late_search_count=len(late),
                                touched=bool(offer['near_contact']),h=projected_height(offer['receive_line']),
                                any_nominal=any(z['legal_trials'] for z in late),any_issued=bool(selected),
                                any_robust=any(z.get('robust',0)>0 for z in selected),
                                nominal_separation_m=min((z['best_separation_m'] for z in late),default=None),
                                max_rate=max(abs(controls[max(lo,stop-12):stop]).ravel(),default=0.),
                                pre_stop_gap=stop-last if last is not None else None,
                                coded_guard=bool(max(r['motor_audit']['max_measured_rate_rad_s'])>5.454154),
                                searches=[(z['tick'],z['chosen'],z.get('robust',0),z['legal_trials']) for z in late],**d))
    _write(ROOT/'actual-rates.json',dict(source=str(source),source_sha256=_hash(source),
           points=len(doc['rows']),offer_rows=results,role='Actual signed command ranges in final up-to-12 issued ticks since near bounce before contact or unmatched actual endpoint. Search annotations are the public controller receipts and not actual trajectories.'))
    for outcome in (True,False):
        rr=[s for s in results if s['landed']==outcome]
        late=[s for s in rr if s['late_search_count']]
        print(outcome,len(rr),len(late),'touched',sum(s['touched'] for s in rr),'any legal',sum(s['any_nominal'] for s in rr),
              'any robust',sum(s['any_robust'] for s in rr),'median last-search gap',np.median([s['pre_stop_gap'] for s in late]),
              'sat',np.mean([s['rate_sat_fraction'] for s in rr if 'rate_sat_fraction' in s],axis=0),flush=True)


def physical_goals(goals, ref, stride, lower, upper):
    goals=np.asarray(goals)
    out=np.empty_like(goals)
    for t in range(goals.shape[1]):
        target=np.clip(goals[:,t],lower,upper)
        u=np.clip((target-ref)/stride,-.8,.8)
        ref=np.clip(ref+u*stride,lower,upper)
        out[:,t]=ref
    return out


class Audit(season.Player):
    def __init__(self,*args,stride,lower,upper,**kwargs):
        super().__init__(*args,**kwargs)
        self.stride,self.lower,self.upper=stride,lower,upper
        self.watch={}
        self.witness=[]
        self.record=None

    def begin(self):
        super().begin()
        self.witness=[]
        self.record=None

    def trial(self,goals,*,long=False,traces=False):
        r=super().trial(goals,long=long,traces=traces)
        if self.record is not None and not self.in_rank and not traces:
            self.record.append(dict(goals=np.array(goals),long=long, legal=r[1].copy(),loss=r[0].copy(),feedback=r[-1].copy()))
        return r

    def search(self,tick,x,q,ref,base):
        last=len(self.notes)
        watch=self.watch.get(tick)
        self.record=[] if watch is not None else None
        try:
            super().search(tick,x,q,ref,base)
        finally:
            calls,self.record=self.record,None
        if not calls:
            return
        if len(self.notes)!=last+1:
            raise ValueError('Stock audit failed to witness a search')
        # Add no alternative to the player. Reuse the identical isolated optical
        # solver state after stock restores the uncertainty cases.
        saved=self.diag.copy()
        try:
            actual={fingerprint(g):bool(ok) for r in calls if r['long'] for g,ok in zip(r['goals'],r['legal'],strict=True)}
            groups=[]
            for c in calls:
                if c['long']:
                    continue
                tested=np.array([fingerprint(g) in actual for g in c['goals']])
                worth=c['goals'][~tested]
                legal=np.zeros(0,dtype=bool)
                robust=[]
                skipped_scores=[]
                if len(worth):
                    loss,legal,_,y,f=mpc.MPC.trial(self,worth,long=True)
                    if legal.any():
                        agreement,_=self.uncertainty(worth[legal],loss[legal])
                        robust=agreement.tolist()
                        skipped_scores=y[legal].tolist()
                actual_refs=physical_goals(c['goals'],ref,self.stride,self.lower,self.upper)
                assigned=np.clip(c['goals'],self.lower,self.upper)
                p=np.concatenate((np.broadcast_to(ref,(len(actual_refs),1,6)),actual_refs),axis=1)
                hit=np.abs(np.diff(p,axis=1)/self.stride)>.8-1e-8
                groups.append(dict(n=len(c['goals']),tested=int(tested.sum()),short_contact=int(np.sum(c['feedback'][:,1]>0)),
                                   skipped_landing=int(legal.sum()),skipped_robust=robust,skipped_scores=skipped_scores,
                                   reachable=np.max(np.abs(actual_refs-assigned),axis=(1,2)).tolist(),
                                   initial_slew_fraction=hit[:,:5,:5].mean(axis=(0,1)).tolist(),
                                   distinct_first_five_issued=len({fingerprint(np.round(a[:5],5)) for a in actual_refs}),
                                   actual_close_m=float(c['feedback'][:,0].min())))
            self.witness.append(dict(tick=tick, offer=watch, stock=self.notes[-1], optics=x.tolist(), groups=groups,
                                     q_minus_estimate=(q-self.engine.body_state.qpos[self.engine.positions]).tolist()))
        finally:
            self.diag=saved


class Augmented(ParallelDiagnostic, Audit, Recentered):
    pass


def recenter_plan():
    old=_read(PLAN)
    _write(RECENTER_PLAN,dict(funnel_sha256=_hash(PLAN),ids=[r['id'] for r in old['rows'][4:]],
        rule='On the four preassigned last native funnel cases, repeat each settled zero-legal failed search with the same prior, random seed state, observation, budget and diagnostics. The only change is centering motor proposals on the actually issuable paths rather than their unreachable desired references. The stock actor still makes every live decision. At most one physically graded replacement per case, the earliest recentered proposal with both native uncertainty votes, with an otherwise exact stock control. No candidate beyond that is tried. The four IDs are not a fresh actor qualification.'))


def physical_witness(a,c,old,watched,stride,lo,hi):
    eligible=[w for w in watched if w.get('recenter_only',{}).get('robust_winner') is not None
              and w['recenter_only']['note']['robust']==2]
    if not eligible:
        return None
    selected=min(eligible,key=lambda w:w['tick'])
    cut=selected['tick']; leg=selected['offer']['leg']
    a.launch(old['shot'],card=c);a.partner.reset(seed=c['seed'])
    for u in old['actions'][:cut]:
        a.exchange(u,mpc.FAR_COMMAND)
    prefix=[e for e in old['events'] if e['time']<=a.data.time+1e-8]
    if digest_json(prefix)!=digest_json(a.events):
        raise ValueError('Physical pilot did not reproduce the exact archived prefix')
    same=State(a)
    axes=a._arm_ids[1][2]
    want=mpc.tail(np.array(selected['recenter_only']['robust_winner']),130)
    for u in old['actions'][cut:]:
        a.exchange(u,mpc.FAR_COMMAND)
    if digest_json(a.events)!=digest_json(old['events']):
        raise ValueError('Physical pilot stock control did not reproduce')
    same.restore(a)
    a.sample_clearance(True)
    actions=[]
    try:
        for pose in want:
            ref=a.data.ctrl[axes].copy()
            u=np.clip((np.clip(pose,lo,hi)-ref)/stride,-.8,.8)
            actions.append(u.tolist())
            _,_,end,capped,result=a.exchange(u,mpc.FAR_COMMAND)
            if end or capped or any(e['kind']=='legal_return' and e.get('side')==1 for e in result['events']):
                break
        trace=a.point_placements(); true=old['placements']['delivered'][leg]
        got=trace['delivered'][leg]
        if not all(digest_json(true[k])==digest_json(got[k]) for k in ('far_contact','near_bounce')):
            raise ValueError('Physical witness changed the same incoming offer')
        if any(q['target']!=list(draw(c,k,'random',None)[0]) for k,q in enumerate(trace['requested'])):
            raise ValueError('Physical witness changed the standard continuous draw')
        return dict(tick=cut,landed=got['near_landed'],touched=bool(got['near_contact']),native=selected['recenter_only']['note'],
                    actual=got,verification=dict(prefix_sha256=digest_json(prefix),stock_landed=false_landing(true),
                                                 events_sha256=digest_json(a.events)),clearance=a.clearance_result(),rates=actions)
    finally:
        a.sample_clearance(False)


def false_landing(offer):
    if offer['near_landed']:
        raise ValueError('Expected a known failed stock control')
    return False


def run(out,with_recenter=False):
    plan=_read(PLAN)
    if plan['source_sha256']!=_hash(STOCK) or plan['carry_sha256']!=_hash(season.CARRY):
        raise ValueError('Changed audit sources')
    d=_read(STOCK); ready=d['ready']; states=_read(season.CARRY)
    output=_read(out) if out.exists() else dict(plan_sha256=_hash(PLAN),rows=[],complete=False)
    if output['plan_sha256']!=_hash(PLAN):
        raise ValueError('Changed audit roster')
    alternate=_read(RECENTER_PLAN) if with_recenter else None
    if alternate and alternate['funnel_sha256']!=_hash(PLAN):
        raise ValueError('Changed projection comparison roster')
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,'local')
    with mpc.centered(ready):
        a=mpc.arena(); camera,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,'Fragments',lambda b,e:Carried(b,e,practice=[course.PRACTICE],held={c['id'] for c in d['cards']})):
            cls=Augmented if alternate else Audit
            actor=cls(mpc.BANK,raw,variant=d['variant'],budget=d['budget'],coverage=fresh.shot.Coverage(_read(fresh.RISK)),stride=stride,lower=lo,upper=hi)
        try:
            course.resume(actor,season.CARRY.with_suffix('').with_suffix('.npz'),states)
            for r in plan['rows'][len(output['rows']):]:
                i=r['index']; c,old=d['cards'][i],d['rows'][i]
                actor.watch={t:s for s in r['scopes'] for t in s['ticks']}
                actor.audit_alt=bool(alternate and r['id'] in alternate['ids'])
                if actor.priors.digest()!=old['memory']['before']:
                    raise ValueError('Different actual carried command bank')
                got=mpc.play(a,camera,rig,stride,lo,hi,ready,c,old['shot'],raw,actor,course.OPTIC,'frozen',d['cap'])
                for label in ('actions','events','searches','placements'):
                    if digest_json(got[label])!=digest_json(old[label]):
                        raise ValueError(f'Stock audit diverged on {label} in {r["id"]}')
                if set(actor.watch)!={v['tick'] for v in actor.witness}:
                    raise ValueError('Some scheduled published search did not replay')
                physical=physical_witness(a,c,old,actor.witness,stride,lo,hi) if actor.audit_alt else None
                output['rows'].append(dict(id=r['id'],n_own=old['score']['learner'],witness=actor.witness,
                                           tape_sha256=digest_json(got['actions']),
                                           recenter_audited=actor.audit_alt,recenter_physical=physical))
                output['complete']=len(output['rows'])==len(plan['rows'])
                _write(out,output)
                print(r['id'],old['score']['learner'],len(actor.witness),sum(g['skipped_landing'] for w in actor.witness for g in w['groups']),
                      'recenter',actor.audit_alt,[w['tick'] for w in actor.witness if w.get('recenter_only',{}).get('robust_winner')],
                      None if physical is None else (physical['landed'],physical['touched']),flush=True)
        finally:
            actor.close();camera.close();a.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','run','archive','recenter-plan'))
    p.add_argument('--out',type=Path,default=RECEIPT)
    p.add_argument('--with-recenter',action='store_true')
    a=p.parse_args()
    if a.mode=='plan':
        protocol()
    elif a.mode=='archive':
        official_rates()
    elif a.mode=='recenter-plan':
        recenter_plan()
    else:
        run(a.out,a.with_recenter)
