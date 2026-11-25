"""Late optical authorization of same-body learned movement on new physical offers."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt69_clamp_policy as p
from . import tt69_clamp_school as school
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_opportunity_map import projected_height
from .tt40_long_low import same_trace

mpc=p.mpc
ROOT=school.ROOT
PLAN=ROOT/'late.protocol.json'
OUT=ROOT/'late-first.json.gz'
WIDTH=30


def plan():
    roster=_read(p.QUAL); src=roster['portable_full_game']; rng=np.random.default_rng(691070)
    ready=roster['ready']; shots=mpc.measured_launches(school.FEEDS); scout=[]; selected=[]
    # A scout has no motor learner. Its ordinary real receive-line grade
    # stratifies independent uniform draws before any controller plays them.
    with p.fixture.slid(school.DX),mpc.centered(ready),school.enabled():
        a=mpc.arena(width=64,height=48)
        try:
            for c in src:
                seen=set()
                for seed in rng.integers(1,2**30,12):
                    v=dict(c,seed=int(rng.integers(1,2**30)),reply_seed=int(seed))
                    root,prefix=school.launch(a,v,shots[v['id']],[],0,ready)
                    r=None if root is None else school.trial(a,root,v,prefix,0,[school.ZERO]*104)
                    o=None if r is None else r['offer']; line=None if o is None else o['receive_line']
                    h=None if not line else projected_height(line)
                    x=None if not line else line['xyz'][0]
                    mark=None if x is None else 'low' if .935<h<1.026 and -.31<x<.145 else 'routine' if 1.033<h<1.16 and abs(x)<.32 else None
                    sc=dict(card=v,height=h,x=x,grade=mark)
                    scout.append(sc)
                    if mark is not None and mark not in seen:
                        seen.add(mark);selected.append(sc)
                    if len(seen)>=2:break
        finally:a.close()
    _write(PLAN,dict(qualification=_hash(p.QUAL),school=_hash(p.MODEL),ready=ready,cases=selected,scouts=scout,
                     role='Six untouched measured releases from the reserved TT69 portable pool. Unmodified continuous random law. Select the first naturally delivered low and ordinary height evidence per release with neutral motor (at most 12 scouts), then freeze both paired actors. The policy does not see the referee grade or scout targets. Keep all selected physical outcomes. No changes in early preparation or certified standard moves.'))
    print('late docket',len(scout),[(q['card']['id'],q['grade'],round(q['height'],3)) for q in selected],flush=True)


class Late(p.Receiver):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw); self.added=False

    def begin(self):
        super().begin(); self.tried=-100; self.place_edits=[]
        self.in_rank=False; self.metrics={}; self.decision_tick=-99

    def search(self,tick,x,q,ref,base):
        super().search(tick,x,q,ref,base)
        if (not self.added or not self.engine.bounced or not self.notes or self.notes[-1]['tick']!=tick or
            self.notes[-1].get('robust',0)>0):
            return
        self.try_late(tick,x,q,ref)

    def try_late(self,tick,x,q,ref):
        if tick-self.tried<18 or not -.97<x[1]<-.48 or not self.engine.sync():return
        path=p.p_arc(self.engine.twin.partner.flight,self.engine.tracker,tick,x)
        key=p.point_at(path,p.AXIS)
        if key is None or not .929<key[3]<1.083 or not -.32<key[1]<.20:return
        options=[]; reachable=0; related=0
        mark=p.marker(key)
        for k in self.school_field.rows:
            sample=p.point_at(path,k['depth'])
            if sample is None or not .065<sample[0]<.36:continue
            dist=float(np.linalg.norm((mark-k['key'])/(1.10*p.SCALE)))
            if dist>1.95:continue
            related+=1
            if np.max(abs(k['contact_goal'][:5]-ref[:5]))>4.16*(sample[0]+.027):continue
            reachable+=1
            trans=self.adaptation(k,sample)
            time=np.arange(1,mpc.HORIZON+1)*mpc.DT
            present=np.r_[0,path[:,0]]; depth=np.r_[-x[1],-path[:,2]]
            for drift in (-.023,.0,.023):
                yy=np.interp(time+drift,present,depth)
                shape=np.array([np.interp(yy,k['y'],col) for col in k['goal'].T]).T
                taper=np.clip(time/.115,0.,1.)[:,None]
                for scale in (.52,1.,1.28):
                    alt=shape.copy(); alt[:,:5]+=taper*scale*trans
                    if np.max(abs(alt[0,:5]-ref[:5]))>.38:continue
                    options.append((dist+abs(drift)*4+abs(1.-scale)*.11,k,alt))
        if not options:
            self.place_edits.append(dict(tick=tick,paths=0,legal=0,issued=False,related=related,reachable=reachable))
            return
        self.tried=tick
        options=sorted(options,key=lambda q:q[0])[:WIDTH]; go=[q[2] for q in options]
        loss,legal,_,values,_=self.trial(go,long=True)
        vv=np.flatnonzero(legal)
        note=dict(tick=tick,paths=len(go),legal=len(vv),issued=False,related=related,reachable=reachable)
        self.place_edits.append(note)
        if not len(vv):return
        vv=vv[np.argsort(loss[vv])[:6]]
        votes,risk=self.uncertainty([go[j] for j in vv],loss[vv])
        r=values[vv,12]
        admissible=((votes==2)&(r<.065))|((votes>=1)&(r<.044))
        if not admissible.any():return
        win=int(np.argmin(np.where(admissible,risk+8*r,np.inf))); j=int(vv[win])
        self.best=go[j]; self.tick=tick
        horizon=int(np.ceil(values[j,1]/mpc.DT))+4
        self.lease=min(horizon,mpc.HORIZON); self.next=tick+self.lease
        note.update(issued=True,donor=options[j][1]['id'],votes=int(votes[win]),nominal_radius=float(r[win]),duration=self.lease)
        self.moves.append(note); self.intervened_tick=tick


def run():
    q=_read(PLAN); d=_read(p.MODEL)
    if q['qualification']!=_hash(p.QUAL) or q['school']!=_hash(p.MODEL):raise ValueError('Changed independent late holdout')
    cards={r['card']['id'] for r in q['cases']}
    if cards & {r['id'] for r in d['rows']}:raise ValueError('Scope leakage to learned motor row')
    result=dict(protocol=_hash(PLAN),rows=[],complete=False); shots=mpc.measured_launches(school.FEEDS)
    if OUT.exists():
        old=_read(OUT)
        if old['protocol']!=result['protocol']:raise ValueError('Changed unmodified random docket')
        result=old
    with np.load(mpc.BANK) as f:raw=p.fixture.BodyRelativeRaw(mpc.DeepField(f,'local'),school.DX)
    rank=p.season.fresh.shot.Coverage(_read(p.season.fresh.RISK)); ready=q['ready']
    with p.fixture.slid(school.DX),mpc.centered(ready),school.enabled():
        a=mpc.arena(); cad=mpc.arena(width=64,height=48,detail='cad'); cam,rig,stride,lo,hi=mpc.resources(a)
        guard=school.physical.GripClearance(cad.model,stock=True)
        with patch.object(mpc,'Fragments',lambda b,e:p.course.Carried(b,e,practice=[p.course.PRACTICE],held=cards)):
            actor=Late(mpc.BANK,raw,variant='elastic',budget=28,coverage=rank,school_field=p.Field(d),policy='registered')
        actor.priors.home=np.asarray(ready)
        try:
            p.course.resume(actor,p.season.CARRY.with_suffix('').with_suffix('.npz'),_read(p.season.CARRY))
            for spec in q['cases'][len(result['rows']):]:
                c=spec['card']; shot=shots[c['id']]; arms={}
                for mode in ('registered','learned'):
                    actor.added=mode=='learned'
                    v=p.play_actor(a,cam,rig,stride,lo,hi,actor,raw,ready,c,shot,True)
                    v['late']=actor.place_edits
                    if any(z['issued'] for z in v['late']):v['clearance']=school.verify(a,c,shot,v,guard)
                    arms[mode]=v
                a0,b=(arms[k]['placements']['delivered'] for k in ('registered','learned'))
                matched=bool(a0 and b and all(same_trace(a0[0][k],b[0][k]) for k in ('far_contact','near_bounce')))
                rec=dict(case=spec,matched=matched,arms=arms); result['rows'].append(rec)
                result['complete']=len(result['rows'])==len(q['cases']); _write(OUT,result)
                print('late',c['id'],spec['grade'],matched,[(k,v['score']['learner'],len(v['late']),v['school']) for k,v in arms.items()],flush=True)
        finally:actor.close();cam.close();cad.close();a.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__); a.add_argument('mode',choices=('plan','run'));m=a.parse_args().mode
    plan() if m=='plan' else run()
