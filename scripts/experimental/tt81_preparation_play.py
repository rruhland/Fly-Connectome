"""Visible uncertainty, bounded preparation, and actual shifted-clamp qualification."""

import argparse
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

from . import tt81_conditional_preparation as train
from . import tt69_clamp_school as school
from . import tt69_clamp_policy as taught
from . import tt69_clamp_repair as late
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt40_long_low import same_trace

BASE=train.BASE
course=taught.course
season=taught.season
ROOT=train.ROOT


class Policy:
    def __init__(self,doc):
        self.center=np.array(doc['center']); self.scale=np.array(doc['scale']); self.w=np.array(doc['weights'])
        self.sigma=doc['spread']; self.rows=doc['keys']; self.values=np.array([r['key'] for r in self.rows])
        self.goals=np.array([r['pre'] for r in self.rows]); self.end=np.array([r['landing'] for r in self.rows]);
        self.radius=np.array([r['radius'] for r in self.rows]); self.strength=doc['strength']; self.limit=np.array(doc['actuator_limit'])

    def get(self,x,q):
        pred=float(x[1]+np.clip(np.r_[1.,np.clip((x-self.center)/self.scale,-2.25,2.25)]@self.w,-.070,.070))
        low=float(ndtr((train.LOW_THRESHOLD-pred)/self.sigma))
        if low<.38 or not -.325 < x[0] < .20:return None
        # Marginalize a compact observational neighborhood instead of
        # assuming the most likely ray is an exactly known future hit.
        covariance=np.array([[0.,0.],[-.026,-self.sigma],[.026,-self.sigma],[-.026,self.sigma],[.026,self.sigma]])
        prior=np.array([.46,.135,.135,.135,.135]); bounds=np.array([.139,.088,.65,.90,.90]); data=x[:5]
        width=np.broadcast_to(self.values[None]-data[None,None,:],(len(covariance),len(self.rows),5)).copy()
        width[:,:,:2]-=covariance[:,None,:]
        distance=np.linalg.norm(width/bounds,axis=2)
        # The ordinary approach must plausibly reach the preparatory arc
        # without jumping to a different wrist-roll branch as camera fits vary.
        distance+=.25*np.linalg.norm((self.goals[None,:,:5]-q[None,None,:5])/[.48,.70,.64,.65,1.03],axis=2)
        weight=(prior[:,None]*np.exp(-.5*(distance/.98)**2)).sum(axis=0)
        weight*=np.clip((.085-self.radius)/.064,.34,1.)
        best=int(np.argmax(weight)); best_dist=float(np.min(distance[:,best]))
        if best_dist>1.38 or weight[best]<.065:return None
        distinct=np.max(abs(self.goals[:,:5]-self.goals[best,:5]),axis=1)<.66
        weigh=weight*distinct*(weight>=.23*weight[best]); spend=weigh/max(1e-8,weigh.sum())
        pick=(spend[:,None]*self.goals).sum(axis=0); actual=np.asarray(self.rows[best]['pre'])
        ready_mix=.77*pick+.23*actual
        return ready_mix,dict(low=low,predicted=pred,distance=best_dist,central=self.rows[best]['id'],
                              supports=[self.rows[j]['id'] for j in np.flatnonzero(weigh)])


class Prepared(late.Late):
    def __init__(self,*a,precursor,**kw):
        super().__init__(*a,**kw); self.precursor=precursor; self.mode='registered'

    def begin(self):
        super().begin();self.clear_prep();self.preparation=[];self.paid_ticks=0

    def clear_prep(self):
        self.gate_prev=None;self.gate_time=-99.;self.ready_hint=None;self.last_seen=-999;self.paid_inbound=False

    def step(self,tick,eye,x,q,ref,base):
        if self.mode!='prepare':return super().step(tick,eye,x,q,ref,base)
        if x is not None and x[4]>.90:
            self.clear_prep();return super().step(tick,eye,x,q,ref,base)
        fresh=eye.latest
        live_phase=bool(fresh is not None and not fresh.bounced and x is not None and x[4]<-.90 and -.48<x[1]<.82)
        if not live_phase:
            if fresh is not None and fresh.bounced:
                self.gate_prev=None;self.ready_hint=None
            return super().step(tick,eye,x,q,ref,base)
        if fresh.time>self.gate_time+1e-8:
            old=self.gate_prev; old_t=self.gate_time; self.gate_time=fresh.time
            path=taught.p_arc(self.engine.twin.partner.flight,eye,tick,x); observation=train.cue(path,x)
            if observation is not None:
                self.gate_prev=observation
                if old is not None and fresh.time-old_t<.091 and np.linalg.norm((observation[:5]-old[:5])/taught.SCALE)<.95:
                    averaged=.63*observation+.37*old
                    found=self.precursor.get(averaged,q)
                    if found is not None:
                        self.last_seen=tick; posture,evidence=found
                        self.ready_hint=posture,evidence
                        if not self.paid_inbound:
                            self.preparation.append(dict(tick=tick,estimated=evidence,q=q.tolist(),expected=posture.tolist()))
                            self.paid_inbound=True
        if self.ready_hint is not None and tick-self.last_seen<=6:
            q0,d=self.ready_hint
            post=base.copy(); phase=float(np.clip((.82-x[1])/.35,0.,1.)*np.clip((x[1]+.54)/.15,0.,1.))
            weight=self.precursor.strength*phase*float(np.clip((d['low']-.25)/.57,0.,1.))
            shift=weight*np.clip(q0[:5]-base[:5],-self.precursor.limit,self.precursor.limit)
            if np.max(abs(shift))>.005:
                post[:5]+=shift; base=post; self.paid_ticks+=1;self.intervened_tick=tick
        return super().step(tick,eye,x,q,ref,base)


def actor_scope(q,values,ids,ready):
    raw0,rank,params=values
    with patch.object(BASE,'Fragments',lambda b,e:course.Carried(b,e,practice=[course.PRACTICE],held=ids)):
        actor=Prepared(BASE.BANK,raw0,variant='elastic',budget=28,coverage=rank,
                       school_field=taught.Field(q),policy='registered',precursor=Policy(params))
    actor.priors.home=np.asarray(ready); return actor


def first():
    q=_read(train.PLAN); params=_read(train.MODEL); materials=_read(taught.MODEL)
    if params['protocol']!=_hash(train.PLAN) or q['raw_field_sha256']!=_hash(taught.MODEL):raise ValueError('Changed same-body hold')
    if {c['card']['id'] for c in q['held']}&{r['id'] for r in params['keys']}:raise ValueError('New held release in body training')
    shots=BASE.measured_launches(school.FEEDS);ready=q['ready'];ids={s['card']['id'] for s in q['held']}
    result=dict(protocol=_hash(train.PLAN),model=_hash(train.MODEL),rows=[],complete=False)
    if train.OUT.exists():
        old=_read(train.OUT)
        if any(old[k]!=result[k] for k in ('protocol','model')):raise ValueError('Changed exact receive comparison')
        result=old
    with np.load(BASE.BANK) as z:raw=taught.fixture.BodyRelativeRaw(BASE.DeepField(z,'local'),school.DX)
    rank=season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with taught.fixture.slid(school.DX),BASE.centered(ready),school.enabled():
        a=BASE.arena(); cad=BASE.arena(width=64,height=48,detail='cad'); cam,rig,stride,lo,hi=BASE.resources(a)
        guard=school.physical.GripClearance(cad.model,stock=True)
        actor=actor_scope(materials,(raw,rank,params),ids,ready)
        try:
            course.resume(actor,season.CARRY.with_suffix('').with_suffix('.npz'),_read(season.CARRY))
            for spec in q['held'][len(result['rows']):]:
                c=spec['card']; shot=shots[c['id']]; arms={}
                for k in ('registered','late','prepare'):
                    actor.mode=k;actor.added=k!='registered'
                    r=taught.play_actor(a,cam,rig,stride,lo,hi,actor,raw,ready,c,shot,True)
                    r.update(preparation=actor.preparation,prep_ticks=actor.paid_ticks,late=actor.place_edits)
                    if r['engaged_ticks']:
                        r['clearance']=school.verify(a,c,shot,r,guard)
                    offers=r['placements']['delivered']; target=spec['actual_offer']
                    r['matched']=bool(offers and all(same_trace(offers[0][j],target[j]) for j in ('far_contact','near_bounce')))
                    arms[k]=r
                result['rows'].append(dict(case=spec,arms=arms));result['complete']=len(result['rows'])==len(q['held']);_write(train.OUT,result)
                print('new held',c['id'],spec['grade'],round(spec['h'],3),
                      [(k,v['score']['learner'],v['matched'],v['prep_ticks'],sum(o['issued'] for o in v['late']),
                        v.get('clearance',{}).get('penetration_samples')) for k,v in arms.items()],flush=True)
        finally:actor.close();cam.close();cad.close();a.close()


def promotion():
    d=_read(train.OUT)
    if not d['complete'] or d['model']!=_hash(train.MODEL) or d['protocol']!=_hash(train.PLAN):
        raise ValueError('Finish matched physical qualification before the full course')
    gains=[]; losses=[]
    for r in d['rows']:
        rr=r['arms']; baseline=rr['registered']['placements']['delivered']; candidate=rr['prepare']['placements']['delivered']
        a=bool(baseline and baseline[0]['near_landed']); b=bool(candidate and candidate[0]['near_landed'])
        if a and not b:losses.append(r['case']['card']['id'])
        if (not a and b and r['case']['grade']=='low' and rr['registered']['matched'] and rr['prepare']['matched'] and
            len(candidate)>1 and candidate[1]['near_bounce'] and rr['prepare'].get('clearance',{}).get('penetration_samples')==0):
            gains.append(r['case']['card']['id'])
    return dict(gains=gains,losses=losses,qualified=bool(gains and not losses))


def games():
    q=_read(train.PLAN); params=_read(train.MODEL); materials=_read(taught.MODEL); permit=promotion()
    if not permit['qualified']:
        print('Full points not licensed',permit,flush=True);return
    shots=BASE.measured_launches(school.FEEDS); ready=np.array(q['ready']); ids={r['id'] for r in q['whole']}
    result=dict(protocol=_hash(train.PLAN),model=_hash(train.MODEL),held_sha256=_hash(train.OUT),promotion=permit,rows=[],complete=False)
    if train.GAMES.exists():
        old=_read(train.GAMES)
        if any(old[k]!=result[k] for k in ('protocol','model','held_sha256')):raise ValueError('Changed qualified full roster')
        result=old
    rank=season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with np.load(BASE.BANK) as f:raw=taught.fixture.BodyRelativeRaw(BASE.DeepField(f,'local'),school.DX)
    with taught.fixture.slid(school.DX),BASE.centered(ready),school.enabled():
        a=BASE.arena(); cad=BASE.arena(width=64,height=48,detail='cad'); cam,rig,stride,lo,hi=BASE.resources(a)
        guard=school.physical.GripClearance(cad.model,stock=True); parent=_read(season.CARRY)
        names=('registered','prepare'); actors={k:actor_scope(materials,(raw,rank,params),ids,ready) for k in names}
        saved={k:ROOT/f'{k}-whole.npz' for k in names}
        try:
            for k,actor in actors.items():
                course.resume(actor,season.CARRY.with_suffix('').with_suffix('.npz'),parent)
                if result['rows']:
                    course.resume(actor,saved[k],dict(checkpoint_sha256=result['checkpoints'][k],rows=[None]*len(parent['rows'])+[r['arms'][k] for r in result['rows']]))
                actor.mode=k;actor.added=k=='prepare'
            for c in q['whole'][len(result['rows']):]:
                arms={}; shot=shots[c['id']]
                for k,actor in actors.items():
                    r=BASE.play(a,cam,rig,stride,lo,hi,ready,c,shot,raw,actor,course.OPTIC,'live',240,20)
                    if any(v['draw']!='random' or v['target']!=list(draw(c,i,'random',None)[0]) for i,v in enumerate(r['placements']['requested'])):
                        raise ValueError('Unmodified broad native opponent law changed')
                    r.update(prep_ticks=actor.paid_ticks,preparation=actor.preparation,late=actor.place_edits)
                    if actor.paid_ticks or actor.moves:
                        r['clearance']=school.verify(a,c,shot,dict(r,controls=r['actions']),guard)
                    arms[k]=r
                result['rows'].append(dict(card=c,arms=arms));result['complete']=len(result['rows'])==len(q['whole'])
                result['checkpoints']={k:course.save(v,saved[k]) for k,v in actors.items()}; _write(train.GAMES,result)
                print('whole',c['id'],[(k,r['score']['learner'],r['prep_ticks']) for k,r in arms.items()],flush=True)
        finally:
            for v in actors.values():v.close()
            cam.close();cad.close();a.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('first','games'));a=p.parse_args()
    first() if a.mode=='first' else games()
