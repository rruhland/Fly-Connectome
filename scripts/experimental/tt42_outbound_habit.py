"""Learn a reversible direct-joint preparation from outgoing RGB and physical examples."""

import argparse
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_cue_vision import quality
from .tt37_joint_capture import observed
from .tt37_joint_deep import DeepField
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, guarded, verify_clearance
from .tt37_visual_inertia import LearnedOptics
from .tt40_long_low import same_trace, ease
from .tt40_low_feasibility import SOURCE, replay_prefix
from .tt41_anticipate import outbound_projection, physical_projection
from .tt41_deadline import RATE, bound
from .tt41_deadline_warm import reference_tape

ROOT=Path('artifacts')
OUT=ROOT/'tt42'
BANK=ROOT/'tt37/self-combined.npz'
OPTICS=ROOT/'tt37/rebound.npz'
STOCK=ROOT/'tt39/action-baseline.json.gz'
PLAN=ROOT/'tt37/memory-course.json'
PROOFS=[ROOT/'tt41_anticipate'/f'{s}-preparation.json.gz' for s in ('identified','strict')]
SHOTS=Path('/tmp/tt25-rallies.json')
SCALE=np.array([.42,.265,.235,.18,1.95,.34,1.8,2.25])
CAPS=np.array([.225,.54,.78,.24,.72,0.])
COMMON_AXES=[1,2,3]
SUPPORT=2.65
SAMPLE_TICKS=32
LEASE=58
STABLE_INBOUND=-1.15


def feature(fit):
    """Use the same prompt short-far-bounce projection as the timing assay."""
    pos,vel=fit.position,fit.velocity
    if not (-1.36<pos[1]<-.16):
        return None
    b=outbound_projection(np.r_[pos,vel],fit.bounced)
    if b is None or not (.05 < b[1] < .65):
        return None
    return np.r_[b,pos[1],vel[:2]]


class Habit:
    """Optical reversal gates a bounded motor memory, not future opponent aim."""

    def __init__(self,mem,enabled):
        self.enabled=enabled
        self.x=np.asarray(mem['features'])
        self.curves=np.asarray(mem['relative_curves'])
        self.source_keys=np.asarray(mem['groups'])
        self.last_seen=None
        self.stance=None
        self.fades=0
        self.records=[]
        self.missed=Counter()

    def inspect(self,tick,fit,dt,reference):
        now=tick*dt
        if not quality(fit,now):
            return
        p=fit.position
        v=fit.velocity
        if self.stance is not None:
            if v[1]<STABLE_INBOUND:
                # An unobserved gap does not invent a return. Demand current
                # genuine high quality exposures in the opposite direction.
                self.finish(tick,'inbound_rgb')
            return
        if v[1]<STABLE_INBOUND and p[1]<-.79:
            self.last_seen=tick
            return
        if self.last_seen is None or tick-self.last_seen>29:
            return
        x=feature(fit)
        if x is None:
            return
        self.last_seen=None
        distance=np.linalg.norm((self.x-x)/SCALE,axis=1)
        # At most one exemplar per earlier practice release. Several adjacent
        # RGB exposures from one point provide temporal coverage, not votes.
        ix=np.array([np.min(np.flatnonzero(self.source_keys==k)[np.argsort(distance[self.source_keys==k])[:1]])
                     for k in np.unique(self.source_keys)])
        ix=ix[np.argsort(distance[ix],kind='stable')]
        dist=distance[ix]
        if dist[0]>SUPPORT:
            self.missed['out_of_support']+=1
            return
        weight=np.exp(-.5*((dist-dist[0])/.77)**2)
        weight/=weight.sum()
        candidate=np.tensordot(weight,self.curves[ix],axes=1)
        # Shared pitch/elbow preparation is reversible. Pan and roll need
        # more evidence than a short flight alone because the far aim is random.
        for axis in (0,4):
            winner=weight[0]>.76 and dist[0]<1.43
            candidate[:,axis]=self.curves[ix[0],:,axis]*.54 if winner else 0.
        confidence=np.clip((SUPPORT-dist[0])/(SUPPORT-.80),.25,1.)
        residual=np.clip(.84*confidence*candidate,-CAPS,CAPS)
        self.stance=dict(tick=tick,feature=x.tolist(),max_offset=np.max(abs(residual),axis=0).tolist(),
                         initial_reference=reference.tolist(),weights=weight.tolist(),
                         distance=dist.tolist(),issued=0)
        self.residual=residual
        self.records.append(self.stance)

    def finish(self,tick,why):
        self.stance.update(end_tick=tick,release=why)
        self.stance=None
        if self.enabled:
            self.fades=3

    def command(self,tick,fit,dt,base,reference,last,stride,lo,hi):
        self.inspect(tick,fit,dt,reference)
        if self.stance is not None:
            age=tick-self.stance['tick']
            if age>=LEASE:
                self.finish(tick,'timeout')
            elif self.enabled:
                step=min(age,len(self.residual)-1)
                phase=ease((step+1)/11.)
                bias=phase*self.residual[step]
                target=np.asarray(base)+bias
                adjusted=bound(target[None,None,:],reference,last,stride,lo,hi)[0,0]
                self.stance['issued']+=1
                return np.clip((adjusted-reference)/stride,-RATE,RATE)
        target=np.clip(base,lo,hi)
        if self.fades:
            self.fades-=1
            target=bound(target[None,None,:],reference,last,stride,lo,hi)[0,0]
        return np.clip((target-reference)/stride,-RATE,RATE)


def train(bank):
    source=_read(SOURCE)
    if source['group']!='practice':
        raise ValueError('Every early motor demonstration must be development-only')
    source_index={tuple(c[k] for k in ('id','seed','reply_seed')):(c,r) for c,r in zip(source['cards'],source['rows'],strict=True)}
    result=dict(role='reversible direct-joint residuals from optical and issued reference practice only',
                hashes={str(f):_hash(f) for f in [SOURCE,*PROOFS]},ready=source['ready'],
                features=[],relative_curves=[],groups=[],receipts=[])
    seen=set()
    with centered(source['ready']):
        a=arena(protocol='random')
        camera,rig,*_=resources(a)
        try:
            for path in PROOFS:
                d=_read(path)
                if not d['complete']:
                    raise ValueError('Partial physical practice proof')
                for row in d['rows']:
                    w=row['result']
                    if (not w['outcomes']['answered'] or not w['incoming_unchanged']
                     or w['clearance']['penetration_samples']):
                        continue
                    key=tuple(row['card'][k] for k in ('id','seed','reply_seed'))
                    if key in seen:
                        raise ValueError('Duplicate practice reply')
                    seen.add(key)
                    c,old=source_index[key]
                    act=w['active']
                    physical=dict(w,shot=old['shot'])
                    refs,*_=reference_tape(a,physical,c)
                    anchor=refs[act].copy()
                    eye=Vision(rig,a.control_dt,raw=True)
                    start=act-24
                    replay_prefix(a,physical,c,start)
                    n=0
                    for tick in range(start,act+11):
                        if tick%2==0:
                            eye.update(tick,camera.capture(),a.joints)
                            fit=eye.eye.latest
                            if act<=tick and quality(fit,tick*a.control_dt):
                                x=feature(fit)
                                if x is not None:
                                    # Only the reversible first fraction of the physically
                                    # issued preparation is learned. Never store stroke or
                                    # exact reference timing from future impact.
                                    q=refs[tick+1:tick+1+SAMPLE_TICKS]-anchor
                                    result['features'].append(x.tolist())
                                    result['relative_curves'].append(q.tolist())
                                    result['groups'].append(c['id'])
                                    n+=1
                        a.exchange(physical['actions'][tick],FAR_COMMAND)
                    if not n:
                        raise ValueError(f"No fresh RGB support for clean practice {c['id']}")
                    result['receipts'].append(dict(practice_id=c['id'],leg=row['leg'],source=str(path),
                                            issued_events_sha256=w['events_sha256'],exposures=n,
                                            initial_reference=anchor.tolist()))
                    print('practice',c['id'],n,'causal starts',flush=True)
        finally:
            camera.close()
            a.close()
    result['complete']=True
    _write(bank,result)


def freeze(plan,bank,feeds):
    old=_read(STOCK)
    school=_read(BANK.with_suffix('.json'))
    course=_read(PLAN)
    mem=_read(bank)
    ids={x['id'] for x in course['replication']}
    practice={r['card']['id'] for r in school['practice']}|set(mem['groups'])
    if not mem['complete'] or old['cards']!=course['replication'] or ids&practice:
        raise ValueError('Practice and prospective measured whole points are not disjoint')
    hashes={str(p):_hash(p) for p in (BANK,OPTICS,STOCK,PLAN,bank,feeds)}
    if school['bank_sha256']!=old['source_sha256'] or school['bank_sha256']!=hashes[str(BANK)]:
        raise ValueError('Frozen baseline uses different direct instruction')
    if course['feed_sha256']!=hashes[str(feeds)] or old['candidate_source']['optical_sha256']!=hashes[str(OPTICS)]:
        raise ValueError('Frozen baseline uses different optics or feeds')
    _write(plan,dict(hashes=hashes,cards=course['replication'],practice_ids=sorted(practice),
                     ready=old['ready'],cap=old['cap'],
                     inclusion='All 24 preassigned broad live points, no per-release selection, stop at terminal/cap.',
                     role='Direct joint law with frozen causal outbound preparation only.'))
    print('frozen',len(course['replication']),'matched starts',flush=True)


def play(a,renderer,rig,stride,lo,hi,ready,c,shot,field,mem,clear,cap):
    a.launch(shot,card=c)
    a.partner.reset(seed=c['seed'])
    guarded(a,clear)
    eye=Vision(rig,a.control_dt,raw=True)
    eye.eye=LearnedOptics(eye.eye,OPTICS)
    actor=Habit(mem,True)
    home=np.asarray(ready)
    qref=home.copy()
    previous=np.zeros(6)
    inbound=False
    commands=[]
    t0=process_time()
    for tick in range(a.max_steps):
        q=a.joints
        # Same base camera and clock as the archived direct stock. The add-on
        # sees only its raw optical estimate, never a callback or a far target.
        x=observed(eye,tick,renderer.capture() if tick%2==0 else None,q)
        if x is not None and x[4]<-.90 and x[1]<-.90:
            inbound=True
        desired=home
        if x is not None and (x[4]<-.90 or (inbound and x[1]<.50)):
            goal,confidence=field.get(x,q,reference=qref)
            if goal is not None:
                desired=home+confidence*(goal-home)
        u=actor.command(tick,eye.eye.tracker.latest,a.control_dt,desired,qref,previous,stride,lo,hi)
        qref=np.clip(qref+u*stride,lo,hi)
        previous=u
        commands.append(u.tolist())
        _,_,end,capped,_=a.exchange(u,FAR_COMMAND)
        if end or capped or a.legal_returns+a.rival_returns-1>=cap:
            break
    verify_clearance(a)
    return dict(shot=shot,actions=commands,events=a.events,counts=audit(a.events,opening_side=1),
                placements=a.point_placements(),prep=actor.records,out_of_support=dict(actor.missed),
                clearance=a.clearance_result(),motor=a.motor_audit(),cpu_s=process_time()-t0)


def capacity(a,clear,c,old):
    a.launch(old['shot'],card=c)
    a.partner.reset(seed=c['seed'])
    guarded(a,clear)
    for u in old['actions']:
        a.exchange(u,FAR_COMMAND)
    verify_clearance(a)
    if digest_json(a.events)!=digest_json(old['events']):
        raise ValueError('CAD observer disturbed the archived exact control')
    return a.clearance_result()


def evaluate(plan,bank,out,feeds,limit,resume):
    plan_doc,mem,old=_read(plan),_read(bank),_read(STOCK)
    if any(_hash(Path(k))!=v for k,v in plan_doc['hashes'].items()):
        raise ValueError('Frozen inputs moved')
    shots=measured_launches(feeds)
    with np.load(BANK) as z:
        field=DeepField(z,'local')
    result=dict(schema='tt42-outgoing-preparation-v1',manifest_sha256=_hash(plan),cards=plan_doc['cards'],
                code_sha256=_hash(Path(__file__)),rows=[],complete=False)
    if resume and out.exists():
        saved=_read(out)
        if any(result[k]!=saved[k] for k in ('manifest_sha256','cards','code_sha256')):
            raise ValueError('Cannot splice different frozen policies')
        result['rows']=saved['rows']
    _write(out,result)
    with centered(plan_doc['ready']):
        a=arena(protocol='random')
        cad=arena(protocol='random',detail='cad')
        clear=CourtClearance(cad.model,articulated=True)
        renderer,rig,stride,lo,hi=resources(a)
        try:
            end=len(result['cards']) if limit is None else min(limit,len(result['cards']))
            for i in range(len(result['rows']),end):
                c=result['cards'][i]
                control=old['rows'][i]
                r=play(a,renderer,rig,stride,lo,hi,plan_doc['ready'],c,shots[c['id']],field,mem,clear,plan_doc['cap'])
                first=r['prep'][0]['tick'] if r['prep'] else len(control['actions'])
                if r['actions'][:first]!=control['actions'][:first]:
                    raise ValueError('Pure sensory preparation changed a prefix before it fired')
                if not r['prep']:
                    if digest_json(r['events'])!=digest_json(control['events']):
                        raise ValueError('Inactive habit differed from archived stock')
                    control_cad=r['clearance']
                else:
                    control_cad=capacity(a,clear,c,control)
                # Evaluator-only outcome correspondence, never a policy input.
                pairs=[]
                for v,w in zip(control['placements']['delivered'],r['placements']['delivered']):
                    same=same_trace(v.get('far_contact'),w.get('far_contact')) and same_trace(v.get('near_bounce'),w.get('near_bounce'))
                    pairs.append(dict(on_same_physical_offer=same,
                        stock_legal=bool(v.get('near_landed')),habit_legal=bool(w.get('near_landed'))))
                r.update(stock_counts=control['score'],stock_clearance=control_cad,identical_prefix_ticks=first,paired_offers=pairs)
                result['rows'].append(r)
                _write(out,result)
                print(c['id'],'returns',control['score']['learner'],r['counts']['learner'],'preps',len(r['prep']),
                      'clean',control_cad['penetration_samples']==0,r['clearance']['penetration_samples']==0,flush=True)
        finally:
            renderer.close()
            cad.close()
            a.close()
    result['complete']=len(result['rows'])==len(result['cards'])
    _write(out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('train','freeze','evaluate'))
    p.add_argument('--bank',type=Path,default=OUT/'outbound-habit-bank.json')
    p.add_argument('--plan',type=Path,default=OUT/'outbound-habit-plan.json')
    p.add_argument('--output',type=Path,default=OUT/'outbound-habit-live.json.gz')
    p.add_argument('--feeds',type=Path,default=SHOTS)
    p.add_argument('--limit',type=int)
    p.add_argument('--resume',action='store_true')
    o=p.parse_args()
    o.output.parent.mkdir(parents=True,exist_ok=True)
    if o.phase=='train':
        train(o.bank)
    elif o.phase=='freeze':
        freeze(o.plan,o.bank,o.feeds)
    else:
        evaluate(o.plan,o.bank,o.output,o.feeds,o.limit,o.resume)
