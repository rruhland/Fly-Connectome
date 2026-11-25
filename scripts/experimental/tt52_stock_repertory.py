"""Keep rewarded RGB/body strokes as options in the existing stock receiving search."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt51_partial_finish as pf
from . import tt51_partial_live as old
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw

mpc=pf.mpc
root=Path("artifacts/tt52_stock_repertory")
plan_path=root/"plan.json"
low_artifact=Path("artifacts/tt51_low_reverse/course.json.gz")
risk_model=pf.prior.VISION_ROOT/"risk-model.npz"
MAX_OPTIONS=6


def live_support(actor,tick):
    """A recently supported ordinary or learned stroke already has priority."""
    live=actor.best is not None and 0<=tick-actor.tick<actor.lease
    if not live:return 0
    if actor.last_extra==actor.tick:return 2
    return max([int(n.get('robust',0)) for n in actor.notes[-3:] if n['tick']==actor.tick and n.get('chosen')] or [0])


def supported_choice(actor,tick):
    return live_support(actor,tick)>0


def fresh_bounce(track,tick,bounced):
    return bool(bounced and track and 0 <= tick*mpc.DT-track[-1][0] < 2.4*mpc.DT)


class Joined:
    """Receives the same camera-path and encoder context for both teaching sources."""

    def __init__(self, normal, low):
        self.normal=normal
        self.low=low

    def propose(self,x,path,body):
        if not len(path):return []
        out=[]
        focus=path[abs(path[:,2]+1.10)<.043]
        if self.low is not None and len(focus):
            d=focus[np.argmin(abs(focus[:,2]+1.10))]
            forecast=np.r_[d[1],d[3:7]]
            # Select from optical motion and current encoder support. A far
            # opponent draw, archived handoff time and native contact are never used.
            _,height=pf.prior.deep_coordinates(forecast)
            if pf.prior.bounded_forecast(forecast) and height<1.014:
                q,_,_=body
                eligible=[]
                for name,g in self.low.proposals(forecast,path,x):
                    move=np.max(abs(np.asarray(g[0,:5])-q))
                    if move<=.54: eligible.append((float(move),name,g))
                for _,name,g in sorted(eligible,key=lambda z:z[0])[:MAX_OPTIONS]:
                    out.append((f"low/{name}",g))
        if self.normal is not None:
            goals,names=self.normal.proposals(x,path,body)
            out.extend((f"ordinary/{name}",g) for name,g in zip(names,goals,strict=True))
        return out[:MAX_OPTIONS]


def low_field(path,natural=None):
    # The source artifact contains held/online outcomes, but only the two
    # initial RGB/FK teaching rows are allowed into this frozen motor field.
    from .tt51_low_reverse_curriculum import RGBTaught
    d=_read(path)
    rows=d['initial_templates']
    if len(rows)!=2:raise ValueError("Expected two initial paid optical low templates")
    rows=list(rows)
    if natural is not None:
        selected,_=natural_rows(natural)
        rows.extend(dict(r,name=r.get('name',f'natural-{j}')) for j,r in enumerate(selected))
    return RGBTaught(rows)


def natural_rows(path):
    n=_read(path)
    selected=n if isinstance(n,list) else n.get('templates',n.get('rows'))
    if not selected:raise ValueError('Need physically earned natural waveform rows')
    rows=[];ids=set()
    for item in selected:
        t=item.get('template',item)
        if not {'p','relative','refs','depth'}<=set(t):raise ValueError('Invalid RGB teaching packet')
        c=item.get('case',item)
        if 'release' in c:ids.add(int(c['release']))
        rows.append(t)
    return rows,ids


class Arm(old.Arm):
    """Reserve independent moves until the plant's normal post-bounce decision window."""

    def __init__(self,*args,repertory=None,weak=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.repertory=repertory
        self.weak=weak

    def begin(self):
        super().begin()
        self.last_extra=-999
        self.repertory_log=[]

    def search(self,tick,x,q,reference,base):
        # No change to base candidates, their scoring, native trial reuse,
        # RNG consumption, or search clock. Late options only fill an empty slot.
        super().search(tick,x,q,reference,base)
        support=live_support(self,tick)
        if self.repertory is None or not (-1.36<x[1]<-.49) or support==2:
            return
        if support==1 and (not self.weak or x[1]>-.72):
            return
        tracker=self.engine.tracker
        f=getattr(tracker,'filter',None)
        if not fresh_bounce(getattr(f,'track',None),tick,self.engine.bounced):
            return
        if not self.engine.sync():return
        a=self.engine.twin
        path=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=True,horizon=.54)
        proposals=self.repertory.propose(x,path,self.ready.feature_body())
        if not proposals:return
        names,goals=map(list,zip(*proposals))
        losses,_,traces,_,_=self.trial(goals,traces=True)
        for i in np.argsort(losses)[:2]:
            correction=self.transport(traces[i],face=True)
            if correction is not None:
                goals.append(goals[i]+.62*correction)
                names.append(names[i]+"/transport")
        incumbent=None
        if support==1:
            incumbent=len(goals)
            goals.append(mpc.tail(self.best[tick-self.tick:],mpc.HORIZON));names.append('stock-incumbent')
        ranks,legal,_,grade,feedback=self.trial(goals,long=True)
        admissible=legal & (grade[:,5]>(.018 if incumbent is None else .036))
        if incumbent is not None:
            admissible &= feedback[:,3]>2.05
            admissible[incumbent]=False
        keep=np.flatnonzero(admissible)
        keep=keep[np.argsort(ranks[keep])[:4]]
        note=dict(tick=tick,options=names,legal_count=int(len(keep)),accepted=False,
                  incoming_y=float(x[1]),freshness_ticks=(tick*mpc.DT-f.track[-1][0])/mpc.DT,
                  stock_initial_support=support)
        self.repertory_log.append(note)
        if not len(keep):return
        if incumbent is None:
            chosen,agree,quality=self.robust([goals[i] for i in keep],keep,ranks[keep])
        else:
            paired=np.r_[keep,incumbent]
            agreement,quality_=self.uncertainty([goals[i] for i in paired],ranks[paired])
            trustworthy=(agreement[:-1]==2) & ((quality_[:-1]+.20)<quality_[-1])
            note.update(stock_rechecked_support=int(agreement[-1]),stock_rechecked_score=float(quality_[-1]))
            if agreement[-1]==2 or not trustworthy.any():return
            j=int(np.argmin(np.where(trustworthy,quality_[:-1],np.inf)))
            chosen,agree,quality=int(keep[j]),int(agreement[j]),float(quality_[j])
        note.update(chosen=names[chosen],support=int(agree),score=float(quality))
        if agree!=2:return
        self.best=np.array(goals[chosen])
        self.tick=tick
        self.lease=mpc.LEASE
        self.last_extra=tick
        note['accepted']=True
        self.diag['repertory_selected']+=1


class Credit(old.Tail):
    def keep(self,rows,tag):
        # Identical ordinary carry credit. Only the repertory actor stores a
        # second copy indexed as a body-conditioned waveform.
        count=pf.prior.Carried.keep(self,rows,tag)
        a=self.owner
        if count and a.repertory is not None and a.repertory.normal is not None:
            sample=old.camera_motion(rows)
            if sample is not None:
                self.serial_skills+=1
                a.repertory.normal.moves.append((f'earned-{self.serial_skills}',sample))
                if len(a.repertory.normal.moves)>self.original_skills+8:
                    a.repertory.normal.moves.pop(self.original_skills)
        return count


def freeze():
    prior=pf.prior
    never={r['id'] for r in _read(pf.plan_path)['rows']}|{r['id'] for r in _read(old.WHOLE)['cards']}|{2741}
    for p,k in ((prior.course.PLAN,'practice'),(prior.season.PLAN,'practice'),(prior.fresh.PLAN,'assessment')):
        never.update(c['id'] for c in _read(p)[k])
    never.update(c['id'] for c in _read(prior.VISION_ROOT/'difficult-plan.json')['cases'])
    for p in sorted((prior.VISION_ROOT/'observations').glob('case-*.npz')):
        with np.load(p) as z:never.add(int(z['id']))
    src=_read(prior.course.SOURCE)
    pool=sorted({c['id']:c for c in src['practice']+src['assessment']
                 if c['id'] not in never and c['profile']=='reach' and c['opening']=='random'}.values(),key=lambda c:c['id'])
    rng=np.random.default_rng(5263801)
    cards=[dict(pool[i],seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30)))
           for i in rng.permutation(len(pool))[:4]]
    if len(cards)!=4:raise ValueError('Need four wholly new measured releases')
    _write(plan_path,dict(schema='tt52-stock-repertory-1',cards=cards,holdout_count=len(pool),
                         ordinary_school_sha256=_hash(pf.school_path),fixed_risk_sha256=_hash(risk_model),
                         low_initial_sha256=pf.prior.digest_json(_read(low_artifact)['initial_templates']),
                         preserved_failure=12068,role='Actual new broad games compare identical risk-guided ready body. Only extra choices differ.' ))


def install(raw,scene,reference,*,enabled,frames,weak=False,natural=None):
    prior=pf.prior
    f=pf.Finish(_read(pf.school_path)) if enabled else None
    rec=Joined(f,low_field(low_artifact,natural=natural)) if enabled else None
    p=Arm(mpc.BANK,raw,mode='partial',finish=pf.Finish(),checker=scene,
          predictor=old.logged_predictor(risk_model),repertory=rec,weak=weak,variant=reference['variant'],budget=reference['budget'],
          coverage=prior.fresh.shot.Coverage(_read(prior.fresh.RISK)))
    p.tap=frames;p.priors.owner=p;p.priors.original_skills=len(f.moves) if f is not None else 0;p.priors.serial_skills=0
    return p


def run(kind,out,n,weak=False,natural=None):
    reg=_read(plan_path)
    if (reg['ordinary_school_sha256']!=_hash(pf.school_path) or reg['fixed_risk_sha256']!=_hash(risk_model) or
        reg['low_initial_sha256']!=pf.prior.digest_json(_read(low_artifact)['initial_templates'])):
        raise ValueError('Changed early risk or physically earned templates')
    prior=pf.prior;base=_read(prior.ARCHIVE);boot=base['ready']
    old_result=_read(old.OUT)
    if kind=='exact':
        cards=[c for c in old_result['cards'] if c['id']==reg['preserved_failure']]
        cap=5
    else:
        cards=reg['cards'][:n];cap=240
    if natural is not None and natural_rows(natural)[1]&{c['id'] for c in cards}:
        raise ValueError('The natural waveform archive includes a held release')
    doc=dict(schema='tt52-ordinary-search-repertory-native-1',kind=kind,cards=cards,cap=cap,
             plan_sha256=_hash(plan_path),source_sha256=_hash(Path(__file__)),weak_incumbent_comparison=weak,
             natural_waveforms_sha256=None if natural is None else _hash(natural),rows=[])
    if out.exists():
        prev=_read(out)
        if any(prev.get(k)!=v for k,v in doc.items() if k!='rows'):raise ValueError('Changed physical run')
        doc=prev
    learned={**base,'rows':[None]*len(_read(prior.season.CARRY)['rows'])+base['rows']}
    shots=mpc.measured_launches(Path('/tmp/tt25-rallies.json'))
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    with mpc.centered(boot):
        a=mpc.arena();cad=mpc.arena(detail='cad')
        cam,rig,stride,lo,hi=mpc.resources(a);tap=old.Tap(cam)
        checker=prior.Reach(cad,boot,boot)
        try:
            for c in cards[len(doc['rows']):]:
                row={'id':c['id'],'variants':{}}
                for enabled in (False,True):
                    with patch.object(mpc,'Fragments',lambda f,e:Credit(f,e,practice=[prior.course.PRACTICE],held={d['id'] for d in cards})):
                        p=install(raw,checker,base,enabled=enabled,frames=tap,weak=weak,natural=natural)
                    try:
                        prior.course.resume(p,prior.BANK,learned)
                        r=mpc.play(a,tap,rig,stride,lo,hi,boot,c,shots[c['id']],raw,p,prior.course.OPTIC,'live',cap)
                        if any(u['target']!=list(draw(c,j,'random',None)[0]) for j,u in enumerate(r['placements']['requested'])):
                            raise ValueError('Unchanged broad TT37 targets required')
                        row['variants']['stock-options' if enabled else 'prepared-stock']=r|dict(
                            repertory=p.repertory_log,optical_updates=p.priors.serial_skills,
                            readiness_ticks=p.native_readiness_ticks,risk_forecasts=p.ready.predictor.estimates)
                    finally:p.close()
                a0,b0=row['variants'].values()
                if a0['memory']['before']!=b0['memory']['before']:raise ValueError('Changed opening motor bank')
                if kind=='exact':
                    initial=next(u for u in old_result['rows'] if u['id']==c['id'])['runs']['partial-finish']
                    first=next(s['tick'] for s in initial['followthrough'] if s.get('model_support',0)>0)
                    expected=initial['placements']['delivered'][1]
                    row['preserves_prior_commands_through_tick']=first-1
                    if any(v['actions'][:first]!=initial['actions'][:first] or not
                           all(prior.same_trace(v['placements']['delivered'][1][k],expected[k]) for k in ('far_contact','near_bounce'))
                           for v in (a0,b0)):
                        raise ValueError('Changed physical intervention before tick 158')
                print(c['id'],[(m,v['score']['learner'],len(v['repertory']),sum(t['accepted'] for t in v['repertory']))
                      for m,v in row['variants'].items()],flush=True)
                doc['rows'].append(row);_write(out,doc)
        finally:a.close();cam.close();cad.close()
    doc['complete']=True;_write(out,doc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','exact','live'))
    p.add_argument('--out',type=Path)
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--weak',action='store_true',help='also permit a checked comparison against a marginal stock stroke')
    p.add_argument('--natural',type=Path,help='optional independently earned prior natural return waveforms')
    args=p.parse_args();root.mkdir(parents=True,exist_ok=True)
    if args.mode=='plan':freeze()
    else:run(args.mode,args.out or root/(args.mode+'.json.gz'),args.limit,weak=args.weak,natural=args.natural)
