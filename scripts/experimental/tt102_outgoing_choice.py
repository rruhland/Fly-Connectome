"""Rank equally robust real strokes by the next reception predicted from the far rebound."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt80_prospective_course as course80
from . import tt94_optical_rebound as sense
from . import tt94_broad_rebound as teach
from . import tt82_partner_model as partner
from . import tt55_paired_preference as pair_eval
from . import tt55_paired_afterstate as pairs
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json
from .tt54_training_speedups import enabled

ROOT=Path('artifacts/tt102_outgoing_rebound')
PLAN=ROOT/'protocol.json'
MODEL=ROOT/'teacher.npz'
ACHIEVED=ROOT/'achieved-pairs.json'
SEE=ROOT/'calibration.json.gz'
GAME=ROOT/'games.json.gz'
CONTROL=course80.ROOT/'frozen.json.gz'
BASE=course80.season
COURSE=course80.course
mpc=COURSE.mpc
# Include different portions of an existing independent ordered roster, with
# no rule looking at return count, landing, future goals or player outcome.
CAL=(1,7)
TEST=(4,9,14,19,24,29)
MIN_GAP=.035
DELTA=.88
DT=mpc.DT
OFF=np.array([.0018,-.0023,.0028])
DV=np.array([.023,.043,.073])


def rebound(events,start,stop):
    b=next((e for e in events if e.get('kind')=='bounce' and e.get('side')==-1
            and start<e['time']<stop),None)
    s=next((e for e in events if e.get('collider')=='table' and e['velocity'][1]>0.
            and start<e['time']<stop),None)
    if b is None or s is None or b['time']>s['time']:
        return None
    pre=sense.estimate([(0,np.r_[b['ball'],b['incoming']])],False)
    post=sense.estimate([(0,np.r_[s['ball'],s['velocity']])],True)
    return None if pre is None or post is None else np.r_[pre,post]


def fit():
    ROOT.mkdir(exist_ok=True,parents=True)
    control=_read(CONTROL)
    pre=_read(pairs.PLAN)
    former=_read(sense.ROOT/'broad.json')
    exclude=set(former['reserved']) | {r['id'] for r in pre['held']} | {c['id'] for c in control['cards']}
    with np.load(partner.MODEL) as m:
        receiver=[m[k] for k in ('mu','sigma','mean','weight')]
    strokes={}
    for name,hash0 in _read(teach.atlas.ROOT/'fit.json')['sources'].items():
        path=Path(name)
        if _hash(path)!=hash0:
            raise ValueError('The scored outgoing teacher changed')
        d=_read(path)
        for c,r in zip(d['cards'],d['rows'],strict=True):
            if c['id'] in exclude or c.get('profile')!='reach' or c.get('opening')!='random':
                continue
            a=r['placements']['delivered']
            for x,y in zip(a,a[1:]):
                near=x.get('near_contact'); far=y.get('far_contact'); line=y.get('receive_line')
                if not (x['near_landed'] and near and far and line):
                    continue
                f=rebound(r['events'],near['time'],far['time'])
                if f is None:
                    continue
                lab=sense.response.projected(np.r_[line['xyz'],line['velocity']])
                strokes.setdefault((c['id'],*np.round(f,6),*np.round(lab,6)),(f,lab,c['id']))
    x,y,ids=map(np.asarray,zip(*strokes.values(),strict=True))
    q=np.clip(sense.response.predict(receiver,sense.response.recv_feature(y)),0.,1.)
    ys=np.c_[y[:,0],y[:,0]<.99,q]
    mu=np.median(x,axis=0); scale=np.maximum(.045,x.std(0))
    np.savez_compressed(MODEL,mu=mu,scale=scale,z=(x-mu)/scale,labels=ys,ids=ids)
    r=_read(course80.PLAN)
    if control['cards']!=r['practice'][:len(control['rows'])] or control['mode']!='frozen':
        raise ValueError('No exact baseline on the original ordinary stream')
    p=dict(schema='tt102-before-after-outgoing-1',fit=_hash(MODEL),
           reference=_hash(CONTROL),tt80=_hash(course80.PLAN),receiver=_hash(partner.MODEL),
           teacher=_hash(sense.ROOT/'broad.json'),pairs=_hash(pairs.PLAN),count=len(x),training_ids=sorted(set(ids.tolist())),
           excluded=sorted(exclude),calibration=list(CAL),assessment=list(TEST),
           calibration_ids=[control['cards'][i]['id'] for i in CAL],
           assessment_ids=[control['cards'][i]['id'] for i in TEST],
           minimum_projected_gain=MIN_GAP,maximum_ordinary_cost=DELTA,
           role='Only rank already legal ordinary options without reducing existing robustness. Two reconstructed '
                'far-table arcs use public twin trajectories and standard optical perturbations. Previous '
                'throw, current position, future opponent draw and point referee not exposed to ranker. '
                'Use observed actual rebound to assess plan-to-real correspondence after issuing a stroke. '
                'Frozen all-shot and release-different optical calibration points are separate. No resets within games.')
    _write(PLAN,p)
    print('teacher',len(x),len(set(ids)),flush=True)


class Teacher:
    def __init__(self):
        p=_read(PLAN)
        if p['fit']!=_hash(MODEL):
            raise ValueError('Changed independent rebound teacher')
        with np.load(MODEL) as a:
            self.mu,self.scale,self.y=a['mu'],a['scale'],a['labels']
            self.tree=cKDTree(a['z'])

    def predict(self,x):
        f=np.asarray(x)
        d,j=self.tree.query((f-self.mu)/self.scale,k=48)
        w=np.exp(-2*(d/max(.15,d[-1]))**2)
        y=(w@self.y[j])/w.sum()
        return np.r_[y,d[-1]]


def verify():
    p=_read(PLAN)
    if p['pairs']!=_hash(pairs.PLAN):
        raise ValueError('Independent paired roster changed')
    roster=_read(pairs.PLAN)['held']
    truth,complete=pair_eval.collect('held')
    if not complete:
        raise ValueError('Complete independently common-draw physical cases before checking the signal')
    docs={}; out=[]; teacher=Teacher()
    for i,row in enumerate(roster):
        value=[]
        for arm in row['arms']:
            file=arm['path']
            if file not in docs:
                docs[file]=_read(Path(file))
            d=docs[file]['rows'][arm['row']]
            if docs[file]['cards'][arm['row']]['id']!=row['id']:
                raise ValueError('Archived actual departing stroke changed')
            start=arm['depart']['time']
            end=next((e['time'] for e in d['events'] if e.get('kind')=='contact' and e.get('side')==-1
                      and e['time']>start),np.inf)
            f=rebound(d['events'],start,end)
            value.append(None if f is None else dict(estimate=teacher.predict(f).tolist(),observed=f.tolist()))
        out.append(dict(id=row['id'],arms=value,physical=truth[i]['contrasts']['all']))
    selected=[r for r in out if all(x is not None for x in r['arms'])]
    deltas=np.asarray([np.diff([a['estimate'] for a in r['arms']],axis=0)[0] for r in selected])
    h=[x['physical']['height_m'] for x in selected]
    q=[x['physical']['physical_next_legal'] for x in selected]
    report=dict(protocol=_hash(PLAN),n=len(out),usable=len(selected),rows=out)
    if len(selected)>=3:
        report.update(height=pair_eval.signs(deltas[:,0],h),success=pair_eval.signs(deltas[:,2],q),
                      benefit=pair_eval.robust_average(.5*np.sign(deltas[:,2])*q))
    _write(ACHIEVED,report)
    print({k:v for k,v in report.items() if k!='rows'},flush=True)


def from_trace(trace,land_time):
    """Extract the falling and rising arcs of a simulated legal far-table bounce."""
    if not np.isfinite(land_time):
        return None
    ts=np.arange(len(trace))*DT
    v=trace[:,:6]
    valid=np.isfinite(v).all(axis=1) & (v[:,4]>.85) & (v[:,1]>.05)
    before=valid & (ts<land_time-.001) & (ts>=land_time-.105) & (v[:,5]<-.60)
    after=valid & (ts>land_time+.004) & (ts<=land_time+.13) & (v[:,5]>.22)
    if not (before.any() and after.any()):
        return None
    pre=sense.estimate([(k,v[k]) for k in np.flatnonzero(before)[-3:]],False)
    post=sense.estimate([(k,v[k]) for k in np.flatnonzero(after)[:3]],True)
    if pre is None or post is None or max(np.linalg.norm(pre[:2]-post[:2]),abs(pre[1]-post[1]))>.12:
        return None
    return np.r_[pre,post]


class Player(BASE.Player):
    def __init__(self,*a,teacher,bias,choose=False,**kw):
        super().__init__(*a,**kw)
        self.teacher=teacher
        self.bias=np.array(bias)
        self.choose=choose

    def begin(self):
        super().begin()
        self.after={}
        self.perturbed=None
        self.current=None
        self.pending=None
        self.exposures=[]
        self.decisions=[]
        self.eye=sense.Rebound()
        self.inbound=False
        self.stamp=None
        self.sampled_window=None

    def search(self,tick,x,q,reference,base):
        self.after={}
        self.sampled_window=None
        v=super().search(tick,x,q,reference,base)
        last=self.notes[-1] if self.notes and self.notes[-1]['tick']==tick else {}
        if self.sampled_window and last.get('chosen'):
            self.pending=self.sampled_window
        return v

    def trial(self,goals,*,long=False,traces=False):
        need=long and (self.perturbed is not None or not self.in_rank)
        r=super().trial(goals,long=long,traces=traces or need)
        if need:
            buf=r[2]
            store=self.after if self.perturbed is None else self.perturbed
            for i,g in enumerate(goals):
                if r[1][i]:
                    f=from_trace(buf[i],r[3][i,2])
                    if f is not None:
                        store[np.asarray(g).tobytes()]=f
        return r if traces else (r[0],r[1],None,r[3],r[4])

    def robust(self, goals, rank, nominal):
        a=self.engine.twin
        p,v=a.ball.copy(),a.velocity.copy()
        spin=a.data.qvel[a._ball_dof+3:a._ball_dof+6]
        before=spin.copy()
        maps=[self.after.copy()]
        sweeps=[]
        self.in_rank=True
        try:
            for s in (-1.,1.):
                a.ball[:]=p+s*OFF
                a.velocity[:]=v+s*DV
                spin[:]=before+[16.+17.*s,0.,0.]
                self.perturbed={}
                y,good,*_=self.trial(goals,long=True)
                sweeps.append((y,good))
                maps.append(self.perturbed)
        finally:
            a.ball[:]=p; a.velocity[:]=v; spin[:]=before
            self.perturbed=None
            self.in_rank=False
        agree=np.sum([r[1] for r in sweeps],axis=0)
        margins=np.mean([np.where(g,y,2.75+.12*np.minimum(y,10.)) for y,g in sweeps],axis=0)
        quality=.43*nominal+.57*margins
        keys=[np.asarray(g).tobytes() for g in goals]
        x=np.asarray([self.metrics[g] for g in keys])
        risk,support=self.coverage.risk(x)
        incumbent,stock=BASE.fresh.shot.choice(agree,quality,risk,support)
        obs=[[d.get(k) for d in maps] for k in keys]
        value=np.full((len(keys),3,4),np.nan)
        for i,f in enumerate(obs):
            for j,y in enumerate(f):
                if y is not None:
                    value[i,j]=self.teacher.predict(y+self.bias)
        best=incumbent
        scores=value[:,:,2]-.11*value[:,:,1]
        nums=[]
        for i in range(len(keys)):
            valid=(value[incumbent,:,3]<=2.2)&(value[i,:,3]<=2.2)
            paired=scores[i,valid]-scores[incumbent,valid]
            gain=np.mean(paired).item() if len(paired) else None
            # Exact legality risk is primary. Require paired, same-context
            # improvements across both frozen optical alternatives when available.
            good=(i!=incumbent and len(paired)>=2 and agree[i]>=max(1,agree[incumbent])
                  and quality[i]<=quality[incumbent]+DELTA and gain>=MIN_GAP
                  and np.min(paired)>-.020)
            nums.append((gain,int(valid.sum()),bool(good)))
        ready=[i for i,s in enumerate(nums) if s[2]]
        suggested=max(ready,key=lambda i: nums[i][0]-.055*max(0,quality[i]-quality[incumbent])) if ready else incumbent
        if self.choose:
            best=suggested
        if incumbent!=stock:
            self.reranks.append(dict(tick=self.decision_tick,prior=int(rank[stock]),selected=int(rank[incumbent]),
                original=x[stock].tolist(),replacement=x[incumbent].tolist(),quality=quality[[stock,incumbent]].tolist(),
                risks=risk[[stock,incumbent]].tolist(),robustness=agree[[stock,incumbent]].tolist()))
        n=lambda y: np.where(np.isfinite(y),y,None).tolist()
        note=dict(tick=self.decision_tick,ordinary=int(incumbent),suggested=int(suggested),chosen=int(best),
                  agreed=agree.tolist(),quality=quality.tolist(),metrics=n(value),gains=nums)
        self.decisions.append(note)
        if obs[best][0] is not None:
            self.sampled_window=dict(tick=self.decision_tick,feature=obs[best][0].tolist(),
                                     predicted=n(value[best,0]),replaced=best!=incumbent)
        return int(rank[best]),int(agree[best]),float(quality[best])

    def step(self,tick,eye,x,q,ref,base):
        if x is not None and x[4]<-.95:
            self.inbound=True
        if self.inbound and x is not None and x[4]>.9 and x[1]<.25:
            self.eye=sense.Rebound()
            self.inbound=False
            self.stamp=None
            start=self.pending
            self.pending=None
            self.current=dict(**start,turn=tick) if start and tick-start['tick']<=27 else None
        if x is not None and x[4]>.9 and eye.latest is not None and self.stamp!=eye.latest.time:
            self.stamp=eye.latest.time
            f=self.eye.update(tick,x)
            if f is not None:
                data=dict(bounce_tick=tick,actual=f.tolist(),observed_prediction=self.teacher.predict(f).tolist())
                if self.current is not None:
                    data.update(plan=self.current)
                self.exposures.append(data)
                self.current=None
        return super().step(tick,eye,x,q,ref,base)


def calibrate(records,teacher):
    pairs=[x for d in records for x in d['observed_arcs'] if x.get('plan')]
    if len(pairs)<5:
        return np.zeros(10),len(pairs)
    d=np.asarray([np.asarray(x['actual'])-np.asarray(x['plan']['feature']) for x in pairs])
    k0=np.clip(np.median(d,axis=0),-.40*teacher.scale,.40*teacher.scale)
    return .70*k0,len(pairs)


def latest(exposures,events,teacher):
    """After the complete point only. This evaluator is never called by Player."""
    a=[(i,e) for i,e in enumerate(events) if e.get('kind')=='contact' and e.get('side')==1]
    f=[e for e in events if e.get('kind')=='contact' and e.get('side')==-1]
    for r in exposures:
        i=max((j for j,e in a if e['step']<r['bounce_tick']),default=None)
        if i is None:
            continue
        e=events[i]
        k=next((g for g in f if g['time']>e['time']),None)
        if k is None:
            continue
        w=rebound(events,e['time'],k['time'])
        if w is None:continue
        r['physical_arc']=w.tolist()
        r['physical_feature_prediction']=teacher.predict(w).tolist()
    return exposures


def same_prefix(a,b):
    diff=next((i for i,(u,v) in enumerate(zip(a['actions'],b['actions'])) if u!=v),None)
    if diff is None:
        return dict(tick=None,unchanged=digest_json(a['events'])==digest_json(b['events']),matched_offer=None)
    A,B=[r['placements']['delivered'] for r in (a,b)]
    k=next((i for i,(x,y) in enumerate(zip(A,B)) if max((x.get('near_contact') or {}).get('step',1e9),
                                                      (y.get('near_contact') or {}).get('step',1e9))>diff),None)
    if k is None:
        return dict(tick=diff,unchanged=False,matched_offer=None)
    aa,bb=A[k],B[k]
    same=all(digest_json(aa[e])==digest_json(bb[e]) for e in ('far_contact','near_bounce'))
    nxt=[]
    for m in (A,B):
        z=m[k+1] if len(m)>k+1 else {}
        line=z.get('receive_line')
        nxt.append(dict(leg=k+1,measured=bool(line),landed=z.get('near_landed'),
                        geometry=None if not line else sense.response.projected(np.r_[line['xyz'],line['velocity']]).tolist()))
    return dict(tick=diff,unchanged=False,matched_offer=same,leg=k,
                first_landed=[aa['near_landed'],bb['near_landed']],next=nxt)


def run(stage):
    p=_read(PLAN)
    if p['reference']!=_hash(CONTROL) or p['tt80']!=_hash(course80.PLAN):
        raise ValueError('The ordinary native point tape changed')
    base=_read(CONTROL)
    teacher=Teacher()
    if stage=='calibration':
        bias=np.zeros(10); count=0; schedule=p['calibration']; dest=SEE
    else:
        prior=_read(SEE)
        if not prior['complete']:
            raise ValueError('Finish disjoint native optical calibration first')
        bias,count=calibrate(prior['rows'],teacher)
        schedule=p['assessment']; dest=GAME
    contract=dict(protocol=_hash(PLAN),bias=bias.tolist(),bias_pairs=count,mode=stage)
    receipt=_read(dest) if dest.exists() else dict(**contract,rows=[],complete=False)
    if any(receipt.get(k)!=v for k,v in contract.items()):
        raise ValueError('Changed prospective policy, calibration, or roster')
    if receipt['complete']:
        return
    rank=BASE.fresh.shot.Coverage(_read(BASE.fresh.RISK))
    ready=base['ready']
    ids={c['id'] for c in base['cards']}
    if ids.intersection(p['training_ids']):
        raise ValueError('An evaluation release trained the rebound estimator')
    with np.load(mpc.BANK) as f:raw=mpc.DeepField(f,'local')
    with enabled(),mpc.centered(ready):
        a=mpc.arena()
        cam,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,'Fragments',lambda b,e:COURSE.Carried(b,e,practice=[COURSE.PRACTICE],held=ids)):
            m=Player(mpc.BANK,raw,variant=base['variant'],budget=base['budget'],coverage=rank,
                     teacher=teacher,bias=bias,choose=stage!='calibration')
        try:
            carry=_read(BASE.CARRY)
            for j in schedule[len(receipt['rows']):]:
                c,r0=base['cards'][j],base['rows'][j]
                COURSE.resume(m,BASE.CARRY.with_suffix('').with_suffix('.npz'),carry)
                m.serial+=j
                r=mpc.play(a,cam,rig,stride,lo,hi,ready,c,r0['shot'],raw,m,COURSE.OPTIC,'frozen',base['cap'])
                cmp=same_prefix(r0,r)
                extra=[d for d in m.decisions if d['ordinary']!=d['chosen']]
                if r['memory']!=r0['memory']:
                    raise ValueError('Frozen native carry changed')
                if not extra and not cmp['unchanged']:
                    raise ValueError('Trace instrumentation alone changed the ordinary physical tape')
                if extra and cmp['tick'] is not None and cmp['tick']<extra[0]['tick']:
                    raise ValueError('Actions changed before the first new policy decision')
                for k,u in enumerate(r['placements']['requested']):
                    if list(COURSE.draw(c,k,'random',None)[0])!=u['target']:
                        raise ValueError('Native opponent no longer used the same unseen goal stream')
                r.update(id=c['id'],index=j,baseline=r0['score']['learner'],comparison=cmp,
                         observed_arcs=latest(m.exposures,r['events'],teacher),choices=m.decisions,
                         intervention_count=len(extra),terminal=r['events'][-1]['kind'])
                receipt['rows'].append(r)
                receipt['complete']=len(receipt['rows'])==len(schedule)
                _write(dest,receipt)
                print('TT102',stage,c['id'],r['baseline'],r['score']['learner'],len(m.decisions),len(extra),
                      len(m.exposures),sum(bool(t.get('plan')) for t in m.exposures),flush=True)
        finally:
            m.close();cam.close();a.close()


def report():
    t=Teacher()
    original=_read(CONTROL)
    out=dict(protocol=_hash(PLAN))
    for file in (SEE,GAME):
        if not file.exists():
            continue
        d=_read(file)
        rows=d['rows']
        compared=[r['comparison'] for r in rows if r['comparison']['tick'] is not None]
        same=[z for z in compared if z['matched_offer']]
        changed=[r for r in rows if not r['comparison']['unchanged']]
        windows=[c for r in rows for c in r['choices']]
        observations=[o for r in rows for o in r['observed_arcs']]
        matched=[o for o in observations if o.get('plan')]
        s=dict(complete=d['complete'],points=len(rows),
               scores=[[r['id'],r['baseline'],r['score']['learner']] for r in rows],
               stock=sum(r['baseline'] for r in rows),experiment=sum(r['score']['learner'] for r in rows),
               physical_changed_points=len(changed),first_same_incoming=len(same),
               same_first_both_legal=sum(all(z['first_landed']) for z in same),
               decisions=len(windows),warm_from_plan=sum(c['metrics'][c['chosen']][0][0] is not None for c in windows),
               headline_candidates=sum(c['suggested']!=c['ordinary'] for c in windows),
               used_motion_choices=sum(c['chosen']!=c['ordinary'] for c in windows),
               observed_rebounds=len(observations),mapped_plan_bounce=len(matched))
        draws=[]
        for game in rows:
            for o in game['observed_arcs']:
                i=next((u for u in game['placements']['delivered']
                        if u.get('far_contact') and u['far_contact']['step']>o['bounce_tick']
                        and u['far_contact']['step']-o['bounce_tick']<50),None)
                if i is None or i.get('receive_line') is None:
                    continue
                r=i['receive_line']
                h=sense.response.projected(np.r_[r['xyz'],r['velocity']])[0]
                draws.append((h,o['observed_prediction'][0],o['observed_prediction'][2],int(i['near_landed'])))
        if len(draws)>=3:
            g=np.asarray(draws)
            s['compared_actual_single_draws']=dict(n=len(g),median_height_error_mm=float(1000*np.median(abs(g[:,0]-g[:,1]))),
                corr_single_draw=float(np.corrcoef(g[:,0],g[:,1])[0,1]),next_actual_rate=float(g[:,3].mean()),
                probability_mae=float(np.mean(abs(g[:,3]-g[:,2]))))
        details=[]
        for z in changed:
            r=z['comparison']
            if not r['matched_offer'] or r.get('leg') is None or not all(r.get('first_landed',[])):
                continue
            leg=r['leg']
            forecasts=[]
            for world in (original['rows'][z['index']],z):
                d0=world['placements']['delivered']
                near=d0[leg]['near_contact']['time']
                far=d0[leg+1]['far_contact']['time'] if len(d0)>leg+1 and d0[leg+1]['far_contact'] else np.inf
                f=rebound(world['events'],near,far)
                forecasts.append(t.predict(f).tolist() if f is not None else None)
            c=next(a for a in z['choices'] if a['chosen']!=a['ordinary'])
            comp=dict(id=z['id'],tick=r['tick'],completed=[z['baseline'],z['score']['learner']],
                      plan=[c['metrics'][c[key]][0] for key in ('ordinary','chosen')],
                      achieved=forecasts,first_actual_next=r['next'])
            details.append(comp)
        if details:s['matched_first_afterstates']=details
        if matched:
            a=np.array([o['actual'] for o in matched])
            b=np.array([o['plan']['feature'] for o in matched])
            pred=np.array([t.predict(o) for o in b])
            cal=np.array([t.predict(o+np.array(d['bias'])) for o in b])
            target=np.array([o['observed_prediction'] for o in matched])
            s['plan_to_camera']=dict(median_scaled_drift=float(np.median(np.linalg.norm((a-b)/t.scale,axis=1))),
                projected_height_gap_median_mm=float(1000*np.median(abs(pred[:,0]-target[:,0]))),
                calibrated_height_gap_median_mm=float(1000*np.median(abs(cal[:,0]-target[:,0]))),
                median_plan_support=float(np.median(pred[:,3])),median_real_support=float(np.median(target[:,3])))
        out[d['mode']]=s
    _write(ROOT/'summary.json',out)
    print(out,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('fit','pairs','calibration','games','report'))
    t=cli.parse_args().stage
    if t=='fit':fit()
    elif t=='pairs':verify()
    elif t=='report':report()
    else:run(t)
