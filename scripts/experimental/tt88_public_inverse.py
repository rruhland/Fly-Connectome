"""Generate inverse-body roots from ordinary RGB and earned receiving endpoints."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import tt39_terminal_examples as examples
from . import tt83_population_search as population
from . import tt88_inverse_body as inverse
from .tt37_opponent_audit import guarded
from .tt37_optical_clinic import OpticalClinic
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json

mpc, mechanism = inverse.mpc, inverse.tt72
DATA = examples.OUT / 'empirical-endpoints.npz'
PROTOCOL = inverse.ROOT / 'public.protocol.json'
NATIVE = inverse.ROOT / 'public-matched.json.gz'
ROOTS = 6
H = population.H


class PublicRoots:
    """Public-twin adapter for TT83. It never accepts a live arena or launch."""

    def __init__(self, twin, *, exclude=(), model=inverse.MODEL):
        self.motor = inverse.Inverse(inverse.Body(twin), twin, model)
        receipt = _read(DATA.with_suffix('.json'))
        if receipt['endpoints_sha256'] != _hash(DATA):
            raise ValueError('Changed rewarded receiving endpoint receipt')
        with np.load(DATA) as f:
            kept = [i for i,r in enumerate(receipt['train']) if r['id'] not in exclude]
            self.x, self.labels = f['features'][kept], f['endpoints'][kept]
        self.index = cKDTree(self.x/examples.SCALE)

    def target_options(self, engine):
        d = engine.twin
        path = engine.nominal.flight.path(d.ball,d.velocity,d.spin,1,bounced=engine.bounced,horizon=.43)
        if not len(path):
            return []
        choices=[]
        for h in range(6,H-2):
            j=int(np.argmin(abs(path[:,0]-h*inverse.DT)))
            if abs(path[j,0]-h*inverse.DT)>.012:
                continue
            p,velocity=path[j,1:4],path[j,4:7]
            if not (-1.67<p[1]<-1.00 and .839<p[2]<1.105 and abs(p[0])<.23
                    and velocity[1]<-1.):
                continue
            distances,ids=self.index.query(np.r_[p,velocity]/examples.SCALE,8)
            for distance,i in zip(distances,ids,strict=True):
                dx,n,speed,prior=self.labels[i,:3],self.labels[i,3:6],self.labels[i,6],self.labels[i,7:]
                if distance>2.55:
                    break
                v = float(speed)*n
                center=p-dx
                # Test the same active plane on both physical faces where
                # the existing encoder posture allows it. No donor rate.
                for sign in (1.,-1.):
                    target=np.r_[center,sign*n,v]
                    travel=np.max(abs(prior-engine.previous[:5]))
                    orientation=.044 if sign<0. else 0.
                    score=float(distance+.15*travel+orientation+.18*abs(center[1]+1.36))
                    choices.append((score,h,target,int(i)))
        selected=[]
        for item in sorted(choices,key=lambda x:x[0]):
            _,h,target,i=item
            if any(abs(h-r[1])<2 and np.dot(target[3:6],r[2][3:6])>.93 for r in selected):
                continue
            selected.append(item)
            if len(selected)>=ROOTS:
                break
        return selected

    def roots(self,engine,reference):
        # Joint velocity is computed by the ordinary clinic from two encoder
        # reads, not supplied by a simulator or the actual opponent.
        state=np.r_[engine.previous[:5],engine.rate[:5],reference[:5]]
        paths, data=[], []
        for score,h,target,i in self.target_options(engine):
            local,baseline=self.motor.solve(state,target,np.zeros(2),h,rounds=2)
            for tag,rates in (('learned',local),('kinematic',baseline)):
                if np.max(abs(rates))>.800001:
                    raise ValueError('Inverse exceeded stock signed speed')
                prior=reference.copy()
                out=[]
                for rate in rates:
                    prior[:5]=np.clip(prior[:5]+self.motor.motor.stride*rate,
                                      self.motor.motor.lo,self.motor.motor.hi)
                    out.append(prior.copy())
                for _ in range(H-len(out)):
                    # Report actual smooth follow-through, then hold.
                    prior[:5]=np.clip(prior[:5]+.42*self.motor.motor.stride*rates[-1]/(1+len(out)-h),
                                      self.motor.motor.lo,self.motor.motor.hi)
                    out.append(prior.copy())
                data.append(dict(method=tag,horizon=h,receipt_row=i,target=target.tolist(),distance=score))
                paths.append(np.asarray(out)-reference)
        return paths,data


def freeze():
    bench=_read(inverse.BENCH)
    rows=[r for r in bench['rows'] if r['kind']=='held-babble']
    if not bench['complete'] or np.median([r['learned']['weighted'] for r in rows]) >= (
            .75*np.median([r['baseline']['weighted'] for r in rows])):
        raise ValueError('Motor inverse did not qualify out of sample')
    plan=dict(schema='tt88-stock-public-gate-1',cases=[9254,8831],
              model=_hash(inverse.MODEL),benchmark=_hash(inverse.BENCH),
              endpoints=_hash(DATA),endpoint_receipt=_hash(DATA.with_suffix('.json')),roots=ROOTS,
              instructions='Matched developmental receiving trials on the original stock body. '
                           'On misses use actual stock commands until ordinary RGB and encoder inference '
                           'find a robust physical legal stroke. Both fitted and kinematic inverse '
                           'have equal target and legal-gate budgets. No archived motion or privileged state is provided '
                           'to either proposal. Keep ordinary opponent and actual CAD/score. Do not open a fresh '
                           'assessment release without a repeatable new stock skill.')
    if not PROTOCOL.exists():
        _write(PROTOCOL,plan)
    if _read(PROTOCOL)!=plan:
        raise ValueError('Changed the frozen stock public inverse comparison')
    return plan


def run_case(a,clear,cam,rig,engine,proposer,c,old,anchor,actual_control,mode):
    tape=mechanism.tape
    tape.setup(a,c,old,anchor)
    guarded(a,clear)
    eye=mpc.Vision(rig,a.control_dt,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,mechanism.OPTIC)
    ref=a.data.ctrl[a._arm_ids[1][2]].copy()
    lo, hi=proposer.motor.motor.lo,proposer.motor.motor.hi
    stride=proposer.motor.motor.stride
    notes,commands,hits=[],[],[]
    active=None
    last=-100
    try:
        with tape.impact.physical_contacts(a,hits):
            for tick in range(len(actual_control)+H+140):
                q=a.joints.copy()
                x=mpc.observed(eye,tick,cam.capture() if tick%2==0 else None,q)
                engine.observe(tick,eye.eye,q,ref)
                if (active is None and len(notes)<5 and tick-last>=3 and x is not None
                        and x[4]<-1.1 and -1.12<x[1]<-.36 and engine.sync()):
                    last=tick
                    refs, meta=proposer.roots(engine,ref)
                    keep=[i for i,r in enumerate(meta) if r['method']==mode]
                    diag=dict(candidates=0)
                    if keep:
                        selected=np.asarray([refs[i] for i in keep])
                        ranked,diag=population.few.rank_paths(engine,selected,ref,[meta[i] for i in keep])
                        allowed=[ans for ans in ranked if ans[1]['legal'] and ans[1]['robust']>=1]
                        if allowed:
                            active=min(allowed,key=lambda r:r[1]['score'])
                            active=(tick,*active)
                    notes.append(dict(tick=tick,public_x=x.tolist(),gate=diag,
                                      issued=None if active is None else active[2]))
                if active is None:
                    u=actual_control[tick] if tick<len(actual_control) else mechanism.ZERO
                else:
                    i=tick-active[0]
                    u=active[1][i].tolist() if i<len(active[1]) else mechanism.ZERO
                commands.append(list(u))
                ref[:5]=np.clip(ref[:5]+stride*np.asarray(u)[:5],lo,hi)
                # Keep the ordinary calibrated gripper reference in the observation.
                ax=a._arm_ids[1][2][-1]
                ref[5]=np.clip(ref[5]+a._joint_increments[ax]*(a.control_dt/a.model.opt.timestep)*u[5],
                               a._joint_low[ax],a._joint_high[ax])
                _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND)
                if done or cap or a.rival_returns>=2:
                    break
        outcome=tape.trace_result(a,anchor,hits)
        if not outcome['same']:
            raise ValueError('RGB candidate changed the first physical offer')
        return dict(**outcome,notes=notes,commands=commands,clearance=a.clearance_result(),
                    issued=active is not None,mode=mode,terminal=a.point_result())
    finally:
        a.sample_clearance(False)


def run():
    p=freeze()
    ready,index=mechanism.receipt()
    out=_read(NATIVE) if NATIVE.exists() else dict(protocol=_hash(PROTOCOL),rows=[],complete=False)
    if out['protocol']!=_hash(PROTOCOL):
        raise ValueError('Changed public inverse trial roster')
    with mechanism.lateral.fixture.school.enabled(),mpc.centered(ready[0.]):
        a,cad=mpc.arena(),mpc.arena(width=64,height=48,detail='cad')
        clear=mechanism.qual.GripClearance(cad.model,stock=True)
        cam,rig,*_=mpc.resources(a)
        try:
            for release in p['cases'][len(out['rows']):]:
                r=index[release,0.]
                s=r['case']; doc=_read(Path(s['source']))
                card,old,anchor=mechanism.qual.get(doc,s)
                raw=r['motion']['stock_commands']
                exact=raw['placements']['delivered'][0]
                record=[]
                for mode in ('kinematic','learned'):
                    engine=OpticalClinic(body=True)
                    try:
                        choose=PublicRoots(engine.twin,exclude=p['cases'])
                        trace=run_case(a,clear,cam,rig,engine,choose,card,old,anchor,raw['commands'],mode)
                    finally:
                        engine.twin.close()
                    if not mechanism.first_ball(trace,exact):
                        raise ValueError('Public re-test changed exact incoming far impact/bounce')
                    if not trace['issued'] and digest_json(trace['events'])!=digest_json(raw['events']):
                        raise ValueError('An unissued inverse changed the archived point')
                    if mechanism.qual.received(trace,True):
                        replay=mechanism.tape.matched_motion(a,card,old,anchor,trace['commands'],clear)
                        if not mechanism.qual.received(replay,True) or digest_json(trace['events'])!=digest_json(replay['events']):
                            raise ValueError('Public clean interior did not repeat')
                        trace['repeat_sha256']=digest_json(replay['events'])
                    record.append(trace)
                    print('public trade',release,mode,trace['issued'],trace['landed'],trace['answered'],
                          [(x['tick'],x['gate']) for x in trace['notes']],flush=True)
                out['rows'].append(dict(id=release,case=s,conditions=record,native_initial_identical=True))
                out['complete']=len(out['rows'])==len(p['cases'])
                _write(NATIVE,out)
        finally:
            cam.close()
            cad.close()
            a.close()



CACHE_PLAN = inverse.ROOT / 'normal.protocol.json'
GAMES = inverse.ROOT / 'normal-games.json.gz'


class WholePoint(population.Player):
    """Retain the ordinary season actor, adding one evidence-triggered motor option."""

    def __init__(self,*args,exclude=(),**kwargs):
        super().__init__(*args,**kwargs)
        self.inverse=PublicRoots(self.engine.twin,exclude=exclude)

    def begin(self):
        super().begin()
        self.inverse_notes=[]
        self.budget_offer=2
        self.inverse_last=-100

    def step(self,tick,eye,x,q,ref,base):
        if x is not None and x[4]>1.:
            self.budget_offer=2
        return super().step(tick,eye,x,q,ref,base)

    def search(self,tick,x,q,ref,base):
        super().search(tick,x,q,ref,base)
        if not (-1.15<x[1]<-.18 and x[4]<-1.15 and abs(x[0])<.25):
            return
        if not self.notes or self.notes[-1]['tick']!=tick or tick-self.inverse_last<5:
            return
        prior=self.notes[-1]
        if prior.get('robust',0):
            self.bookkeeping['inverse-protect-robust']+=1
            return
        if prior.get('chosen') or not self.budget_offer or not self.engine.sync():
            return
        self.inverse_last=tick
        self.budget_offer-=1
        roots,meta=self.inverse.roots(self.engine,ref)
        keep=[i for i,r in enumerate(meta) if r['method']=='learned']
        row=dict(tick=tick,public_x=x.tolist(),issued=False,options=len(keep))
        self.inverse_notes.append(row)
        if not keep:
            return
        paths=np.asarray([roots[i] for i in keep])
        ranked,diag=population.few.rank_paths(self.engine,paths,ref,[meta[i] for i in keep])
        allowed=[(u,m) for u,m in ranked if m['legal'] and m['robust']]
        explored=False
        if not allowed and self.budget_offer==1:
            # A small exploratory lane on optically underserved receives only.
            from .tt59_online_corrections import low_envelope
            if .80 <= low_envelope(x) <= 1.060:
                allowed=[(u,m) for u,m in ranked if m['legal'] or m['separated'] and m['nominal']<8.25]
                explored=bool(allowed)
        row.update(gate=diag,exploratory=explored)
        if allowed:
            u,d=min(allowed,key=lambda r:r[1]['score'])
            self.best=population.school.atlas.reference_path(
                ref,u,self.population.stride,self.population.lo,self.population.hi)[1:]
            self.tick=tick
            self.lease=H
            row.update(issued=True,selected=d)
            self.budget_offer=0


def full_points():
    from unittest.mock import patch
    from . import tt83_population_eval as protocol
    from . import tt83_playable_course as playable
    existing=_read(playable.OUT)
    roster=_read(playable.PLAN)
    if not existing['complete'] or existing['plan']!=_hash(playable.PLAN):
        raise ValueError('Matched TT83 ordinary points are not yet complete')
    keys={x['card']['id'] for x in roster['games']}
    normal=dict(schema='tt88-normal-public-1', assignment_sha256=_hash(playable.PLAN),
                baseline_sha256=_hash(playable.OUT), motor_sha256=_hash(inverse.MODEL), examples_sha256=_hash(DATA),
                ids=[x['card']['id'] for x in roster['games'][:2]], carry_sha256=_hash(protocol.season.CARRY),
                policy='Reuse the forward frozen TT83 ordinary stock start roster and stock records in order. '
                       'Start from the same carried skills, no injection or reset. Preserve old robust and '
                       'selected stock moves, ask the learned inverse only on up to two public uncertain '
                       'windows per receiving leg, with a bounded low uncertain exploration lane. '
                       'After an issued candidate, return to ordinary within-point operation.')
    if not CACHE_PLAN.exists():
        _write(CACHE_PLAN,normal)
    if _read(CACHE_PLAN)!=normal:
        raise ValueError('Whole-point comparison changed')
    result=_read(GAMES) if GAMES.exists() else dict(protocol=_hash(CACHE_PLAN),rows=[],complete=False)
    if result['protocol']!=_hash(CACHE_PLAN):
        raise ValueError('Changed frozen whole-point inverse contract')
    if result['complete']:
        return
    ready=np.asarray(roster['ready'])
    prior=population.MotorPrior()
    bank=population.sensitivity.Bank(_read(population.sensitivity.MODEL))
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,'local')
    with mpc.centered(ready),protocol.school.enabled():
        a=mpc.arena()
        cam,rig,stride,lo,hi=mpc.resources(a)
        lim=(stride,lo,hi)
        actor=None
        try:
            with patch.object(population,'Player',lambda *args,**kw:WholePoint(*args,**kw,exclude=keys)):
                actor=protocol.factory(raw,lim,bank,prior,'stock',keys,reload=True)
            for i,item in enumerate(roster['games'][:2]):
                c=item['card']
                if i<len(result['rows']):
                    # The predeclared ordinary practice carry is reconstructed
                    # through the same real game when a process must resume.
                    run=protocol.live.play(a,cam,rig,lim,ready,c,item['shot'],raw,actor,item['exploratory_seed'])
                    if digest_json(run['events'])!=result['rows'][i]['learned']['events_sha256']:
                        raise ValueError('Whole-point carry failed deterministic replay')
                    continue
                baseline=existing['rows'][i]['arms']['stock']
                if existing['rows'][i]['card']!=c:
                    raise ValueError('Ordinary stock seed differs')
                row=protocol.live.play(a,cam,rig,lim,ready,c,item['shot'],raw,actor,item['exploratory_seed'])
                row['events_sha256']=digest_json(row['events'])
                row['expansion']=actor.inverse_notes
                row['protected']=actor.bookkeeping['inverse-protect-robust']
                score0=baseline['score']['learner']; score1=row['score']['learner']
                compare=protocol.games0.compare(baseline,row)
                result['rows'].append(dict(id=c['id'],baseline_score=score0,learned_score=score1,
                                           compared_offers=compare,learned=row))
                result['complete']=len(result['rows'])==2
                _write(GAMES,result)
                print('public normal',c['id'],score0,score1,len(actor.inverse_notes),row['protected'],
                      sum(n['issued'] for n in actor.inverse_notes),flush=True)
        finally:
            if actor is not None:
                actor.close()
            cam.close()
            a.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run','normal'))
    {'freeze':freeze,'run':run,'normal':full_points}[parser.parse_args().stage]()
