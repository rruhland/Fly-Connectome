"""Privileged native dynamic stroke teacher. Never a deployed visual policy."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from . import tt33_joint as joint, tt37_motor_clinic as clinic
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, Rollout, measured_launches
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt37_belief import ArrivalHistory
from .tt37_opponent import arena
from .tt37_opponent_benchmark import ARTIFACTS, PLAN, mark
from .tt37_posture import Surface, sample

OUT=ARTIFACTS/'tt38'
SOURCE=ARTIFACTS/'tt37'/'clinic-long-replication.json.gz'
AIM=np.array([0.,1.29,1.05,3.55])
DEPTHS=(-1.34,-1.43,-1.50,-1.56,-1.62,-1.68)
SIZES=np.array([.22,.33,.36,.37,.51,2.4,2.9,2.7,2.8,3.2,.039,.15,.19])


def answerable(value):
    """Rank legal table contacts by their actual post-bounce flight near the far player."""
    v=np.asarray(value)
    x,y,clear,rx,ry,rz,pace=(v[...,i] for i in (3,4,5,6,7,8,9))
    return (6.4*np.maximum(abs(x)-.32,0.)+4.0*abs(y-.68)+
            8.5*np.maximum(abs(rx)-.45,0.)+5.5*np.maximum(1.02-ry,0.)+
            2.8*abs(rz-1.07)+.85*abs(pace-3.4)+
            12.*np.maximum(.055-clear,0.)+23.*np.maximum(v[...,12]-.044,0.))


def good_enough(row):
    return row is not None and float(answerable(row)) < 3.4


def schedules(poses,rates,eta,params,dt,n):
    """Vary the phase lead and the wrist motion before a compliant physical strike."""
    t=(1+np.arange(n))*dt
    effective=t[None,:]-eta[:,None]+(.036+params[:,10])[:,None]
    span=np.clip(effective,-.15,.081)
    phase=np.clip(effective/.13,-1.,.72)
    basis=.5*(1.-np.cos(np.pi*phase))
    posture=poses+params[:,:5]
    dq=rates+params[:,5:10]
    targets=posture[:,None,:]+span[:,:,None]*dq[:,None,:]
    targets[:,:,3:5]+=basis[:,:,None]*params[:,None,11:13]
    return np.concatenate((targets,np.full((*targets.shape[:2],1),clinic.GRIP)),axis=2)


class Dynamic:
    def __init__(self,a,surface,batch=160,rounds=6,depths=DEPTHS):
        self.a=a
        self.surface=surface
        self.core=Rollout(a,1)
        self.batch=batch
        self.rounds=rounds
        self.depths=depths

    def basis(self):
        a=self.a
        h=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=a._received_bounce,horizon=.99)
        if not len(h):
            return []
        h=h[(h[:,0]>.048)&(h[:,3]>.85)&(h[:,3]<1.38)&(abs(h[:,1])<.54)]
        if not len(h):
            return []
        rows=[]
        for depth in self.depths:
            if depth<h[:,2].min()-.017 or depth>h[:,2].max()+.017:
                continue
            tgt=h[np.argmin(abs(h[:,2]-depth))]
            t,x,y,z,vx,vy,vz=tgt[:7]
            dx=-.18*x-.085*vx
            legacy=np.array([dx,1.,.29-.095*vz])
            cross=max(.29,(.73-y)/4.05)
            wanted=np.array([-x/cross,4.55,(.79-z)/cross+4.905*cross])
            measured= wanted-np.array([vx,vy,vz])
            measured/=max(.01,measured[1])
            # Keep both common legacy faces and impulse normals selected by the native outgoing loss.
            for normal in (legacy,.48*measured+.52*legacy,measured):
                normal=normal/np.linalg.norm(normal)
                q,w=self.surface.solution(x,z,normal,depth=y)
                rows.append(dict(eta=t,pose=q,velocity=.84*w,depth=y,normal=normal))
        return rows

    def search(self, *, max_receive_cost=11., tolerance=3.25):
        a=self.a
        bases=self.basis()
        if not bases:
            return None,dict(reason='no bounded contact window',time_s=float(a.data.time))
        rng=np.random.default_rng([383031,a.card['seed'],a.steps])
        scram=int(rng.integers(1,2**31))
        initial=qmc.Sobol(13,scramble=True,seed=scram).random_base2(int(np.ceil(np.log2(self.batch))))[:self.batch]
        choices=(2*initial-1)*SIZES
        options=(np.arange(self.batch)%len(bases)).astype(int)
        choices[:len(bases)]=0.
        q=np.array([r['pose'] for r in bases]); dq=np.array([r['velocity'] for r in bases]); t=np.array([r['eta'] for r in bases])
        dt=a.control_dt
        until=float(t.max()+.115)
        n=round((until+1.14)/dt)
        records=[]
        history=[]
        best=None
        for index in range(self.rounds):
            g=schedules(q[options],dq[options],t[options],choices,dt,n)
            feedback=np.zeros((len(choices),9))
            rates,vals=self.core.trajectories(g,AIM,until,feedback=feedback)
            legal=np.isfinite(vals[:,0]); quality=answerable(vals)
            dense=clinic.landing_error(feedback)
            dense=np.where(feedback[:,1]>0,dense,14.0+12.*feedback[:,0])
            scores=np.where(legal,quality-9.,dense)
            rs=np.flatnonzero(legal)
            if len(rs):
                j=int(rs[np.argmin(quality[rs])])
                if best is None or quality[j]<best['cost']:
                    # Use exactly the native rates screened, including short follow-through.
                    length=int(np.ceil((vals[j,1]+.105)/dt))
                    b=bases[options[j]]
                    best=dict(cost=float(quality[j]),rates=rates[j,:length].copy(),native=vals[j].tolist(),
                              feedback=feedback[j].tolist(),depth=float(b['depth']),eta=float(b['eta']),
                              normal=b['normal'].tolist(),params=choices[j].tolist())
            records.append(dict(round=index,trials=len(choices),contacts=int(np.sum(feedback[:,1]>0)),legal=int(legal.sum()),
                           playable=int(np.sum(legal&(quality<tolerance))),closest_m=float(feedback[:,0].min()),
                           best_cost=None if not len(rs) else float(quality[rs].min())))
            if best is not None and best['cost']<tolerance:
                break
            history.extend(zip(scores,options,choices,strict=True))
            pool=sorted(history,key=lambda r:r[0])[:24]
            scale=.67 if index<2 else .46 if index<4 else .31
            picks=rng.integers(len(pool),size=self.batch)
            options=np.array([pool[i][1] for i in picks],dtype=int)
            choices=np.array([pool[i][2] for i in picks])+rng.normal(size=(self.batch,13))*SIZES*scale
            choices=np.clip(choices,-2.65*SIZES,2.65*SIZES)
            retain=min(8,len(pool))
            options[:retain]=[k[1] for k in pool[:retain]]
            choices[:retain]=[k[2] for k in pool[:retain]]
        notes=dict(time_s=float(a.data.time),tick=a.steps,ball=a.ball.tolist(),velocity=a.velocity.tolist(),
                    rounds=records,window_count=len(bases))
        if best is None:
            return None,notes
        notes['selected']={k:v for k,v in best.items() if k!='rates'}
        return (best if best['cost']<max_receive_cost else None),notes


def setup(depth=-1.56):
    a=arena()
    try:
        ik=BladeIK(a)
        return Surface(ik,sample(ik),depth,.35)
    finally:
        a.close()


def suffix(path,output,tick,rounds,batch,*,cap=50,release=None,automatic=False,forward=False):
    d=_read(path)
    eligible=[i for i,c in enumerate(d['cards'][:len(d['rows'])]) if release is None or c['id']==release]
    i=max(eligible,key=lambda i:d['rows'][i]['score']['learner'])
    old=d['rows'][i]; c=d['cards'][i]
    surf=setup(d.get('depth',-1.56))
    source_tick=tick
    with centered(d['ready']):
        a=arena()
        try:
            if automatic:
                e=next(e for e in reversed(old['events']) if e['kind']=='separation' and e.get('collider')=='rival_racket')
                tick=int(np.ceil(e['time']/a.control_dt-1e-7))+2
            a.launch(old['shot'],card=c)
            a.partner.reset(seed=c['seed'])
            prefix=old['actions'][:tick]
            for u in prefix:
                a.exchange(u,clinic.FAR)
            old_events=[e for e in old['events'] if e['time']<=float(a.data.time)+1e-8]
            if digest_json(a.events)!=digest_json(old_events):
                raise ValueError('Action prefix is not the physical archived witness')
            core=Dynamic(a,surf,batch,rounds,depths=(-1.08,-1.15,-1.23,-1.30,-1.38,-1.48,-1.56) if forward else DEPTHS)
            plan,log=core.search(max_receive_cost=5.5)
            print(tick,log,flush=True)
            header=dict(witness_sha256=_hash(path),source=path.name,card=c,ready=d['ready'],fork_tick=tick,
                explained_auto_tick=automatic,requested_fixed_tick=source_tick,cap=cap,expanded_forward=forward,
                exactly_replayed_prefix=True,search=log,prefix_own=audit(a.events,opening_side=1)['learner'],
                variant=dict(grasp='straight',oracle=True),
                explanation='Diagnostic fork from a measured legal point, not a held whole-policy result.')
            _write(output,dict(header,rows=[],cards=[],complete=False))
            if plan is not None:
                row=continue_point(a,surf,old['shot'],c,prefix,plan['rates'],rounds,batch,cap=cap)
            else:
                row=None
            _write(output,dict(header,rows=[] if row is None else [row],cards=[] if row is None else [c],complete=True))
        finally:
            a.close()


def continue_point(a,surface,shot,card,prefix,initial_rates,rounds,batch,cap,*,control=False,initial_bank=(),
                   warm=None,progress=None):
    """Sensory clinic carries ordinary phases. Exact-state optimization is teacher-only."""
    stereo,rig,stride,lo,hi=joint.resources(a)
    joint.RECEIVE_Y=surface.depth
    old_model=_read(PUPIL)
    actor=joint.JointPolicy(old_model|{'ready':surface.ready.tolist()},stride,lo,hi)
    actor.reference=a.data.ctrl[a._arm_ids[1][2]].copy()
    eye,history=(joint.Vision(rig,a.control_dt),ArrivalHistory()) if warm is None else warm
    arrivals={}
    schedules_old=clinic.schedules
    params=clinic.generate(48)
    optimizer=Dynamic(a,surface,batch,rounds)
    actions=list(prefix)
    tape=None
    tape_start=-1
    raw=initial_rates
    raw_start=len(actions)
    plans=[]
    retry=-1
    deadline=-100.
    lastprep=-100
    prep=surface.ready
    bank=list(initial_bank)
    try:
        while not (a.terminated or a.truncated or a.legal_returns+a.rival_returns-1>=cap):
            tick=len(actions)
            now=tick*a.control_dt
            q=a.joints.copy()
            obs=eye.update(tick,stereo.capture() if tick%2==0 else None,q)
            obs=history.update(now,obs)
            if raw is not None and tick-raw_start<len(raw):
                command=raw[tick-raw_start].copy()
            else:
                if raw is not None:
                    raw=None
                    actor.reference=a.data.ctrl[a._arm_ids[1][2]].copy()
                if tape is not None and tick==retry and obs is not None and .035<obs[0]<.22 and .76<obs[2]<1.38:
                    tape=None
                    deadline=-100
                if tape is not None and tick-tape_start<len(tape):
                    desired=tape[tick-tape_start]
                else:
                    tape=None
                    desired=surface.ready
                    if obs is not None and .03<obs[0]<.80 and now>deadline:
                        if obs[0]<=.225 and .76<obs[2]<1.38:
                            eta=obs[0]; fit=clinic.geometry(surface,obs)
                            x=np.r_[obs[1:6],q[:5]]
                            remembered=clinic.recall(bank,x)
                            selected,score,notes=clinic.search(optimizer.core,fit,eta,np.vstack([remembered,params]),
                              a.control_dt,surface.ready,5,np.random.default_rng([37173,len(bank),tick]),True)
                            row=dict(tick=tick,previous_score=score,search=notes)
                            plans.append(row)
                            if not control and not good_enough(score) and arrivals.get(a.rival_contacts,0)<2:
                                arrivals[a.rival_contacts]=arrivals.get(a.rival_contacts,0)+1
                                candidate,report=optimizer.search(max_receive_cost=9.)
                                row['dynamic']=report
                                if candidate is not None and (score is None or candidate['cost']+.30<float(answerable(score))):
                                    raw=candidate['rates']; raw_start=tick
                                    row['executed_dynamic']=True
                                    deadline=now+len(raw)*a.control_dt+.13
                            if raw is None:
                                if score is not None:
                                    bank.append(dict(x=x.tolist(),change=selected.tolist()))
                                tape=schedules_old(*fit,eta,np.asarray([selected]),a.control_dt,surface.ready,
                                                    max(1,round((eta+.075)/a.control_dt)))[0]
                                desired=tape[0]; tape_start=tick; deadline=now+eta+.26
                                retry=tick+max(3,round((eta-.092)/a.control_dt)) if score is None and eta>.13 else -1
                        elif .76<obs[2]<1.38:
                            if tick-lastprep>=4:
                                p,v=clinic.geometry(surface,obs)
                                prep=np.r_[p-.075*v,clinic.GRIP]
                                lastprep=tick
                            desired=prep
                command=raw[0].copy() if raw is not None else actor.drive(desired,q,.8)
            _,_,_,_,info=a.exchange(command,clinic.FAR)
            actions.append(command.tolist())
            if progress is not None and any(e['kind']=='legal_return' and e['side']==1 for e in info['events']):
                progress(a,actions,plans)
    finally:
        stereo.close()
    row=dict(actions=actions,records=plans,events=a.events,placements=a.point_placements(),score=audit(a.events,opening_side=1),
             shot=shot,opponent_seed=card['seed'])
    mark(a,row,cap)
    print(card['id'],row['score']['learner'],row['score']['returns'],row['physical']['reason'],flush=True)
    return row


def live(feeds,output,rounds,batch,*,control=False,cap=50,count=None,start=0,seed_shift=0):
    d=_read(SOURCE)
    # Every archived release on which the old mentor returned at least four balls.
    base_cards=[card for card,r in zip(d['cards'],d['rows'],strict=True) if r['score']['learner']>=4]
    cards=[c|dict(seed=384076+9211*i+seed_shift,reply_seed=3980734+87351*i+seed_shift*13,
                 key=f'tt38-random-{i}-{c["id"]}',opening='random') for i,c in enumerate(base_cards)]
    cards=cards[start:start+count if count is not None else None]
    shots=measured_launches(feeds)
    surface=setup(d['depth'])
    result=dict(cards=cards,ready=surface.ready.tolist(),rows=[],variant=dict(grasp='straight',oracle=True),
                phase='matched full random',condition='clinic-reference' if control else 'dynamic',cap=cap,
                feed_sha256=_hash(feeds),warm_bank_count=len(d['bank']),source_sha256=_hash(SOURCE),
                search_rounds=rounds,search_batch=batch)
    previous_cpu=0.
    if output.exists():
        saved=_read(output)
        if any(saved.get(k)!=v for k,v in result.items() if k!='rows'):
            raise ValueError('Saved run does not match protocol')
        if saved.get('complete'):
            return
        result['rows']=saved['rows']
        previous_cpu=saved.get('cpu_s',0.)
    started=process_time()
    with centered(surface.ready):
        a=arena()
        try:
            a.launches=len(result['rows'])
            for c in cards[len(result['rows']):]:
                shot=shots[c['id']]
                a.launch(shot,card=c)
                a.partner.reset(seed=c['seed'])
                def interim(live,actions,plans):
                    if live.legal_returns%4==0:
                        print('checkpoint',c['id'],live.legal_returns,len(actions),flush=True)
                row=continue_point(a,surface,shot,c,[],None,rounds,batch,cap,control=control,initial_bank=d['bank'],
                                   progress=interim if cap>=100 else None)
                result['rows'].append(row)
                _write(output,dict(result,cpu_s=previous_cpu+process_time()-started,complete=False))
        finally:
            a.close()
    _write(output,dict(result,cpu_s=previous_cpu+process_time()-started,complete=True))


def extend(prefix_path,output,rounds,batch,cap):
    """Replay one capped point with exact physical prefix and recovered perception/memory."""
    source=_read(SOURCE)
    d=_read(prefix_path)
    k=max(range(len(d['rows'])),key=lambda i:d['rows'][i]['score']['learner'])
    c,capped=d['cards'][k],d['rows'][k]
    if capped['stop']!='target':
        raise ValueError('A referee-ended point cannot be extended')
    proof=_hash(prefix_path)
    r=capped
    if output.exists():
        prior=_read(output)
        if prior.get('source_sha256')!=proof or prior.get('cap')!=cap:
            raise ValueError('Saved continuation belongs to a different capped source or cap')
        if prior.get('complete'):
            return
        candidate=prior['rows'][0]
        if candidate['actions'][:len(capped['actions'])]!=capped['actions'] or candidate['stop'] not in ('other','target'):
            raise ValueError('Saved continuation does not preserve the capped prefix')
        r=candidate
    fork=d.get('fork_tick',0)
    bank=[] if fork else list(source['bank'])
    legacy={p['tick']:p for p in r['records'] if p['previous_score'] is not None and not p.get('executed_dynamic')}
    qsurface=setup(source['depth'])
    joint.RECEIVE_Y=qsurface.depth
    with centered(d['ready']):
        a=arena()
        try:
            a.launch(r['shot'],card=c)
            a.partner.reset(seed=c['seed'])
            stereo,rig,*_=joint.resources(a)
            eye=joint.Vision(rig,a.control_dt)
            history=ArrivalHistory()
            engine=Rollout(a,1)
            params=clinic.generate(48)
            try:
                for tick,u in enumerate(r['actions']):
                    if tick>=fork:
                        q=a.joints.copy()
                        obs=eye.update(tick,stereo.capture() if tick%2==0 else None,q)
                        obs=history.update(tick*a.control_dt,obs)
                        if tick in legacy:
                            if obs is None:
                                raise ValueError('No optical context at an archived accepted action')
                            x=np.r_[obs[1:6],q[:5]]
                            fit=clinic.geometry(qsurface,obs)
                            delta,score,_=clinic.search(engine,fit,obs[0],np.vstack([clinic.recall(bank,x),params]),
                                 a.control_dt,qsurface.ready,5,np.random.default_rng([37173,len(bank),tick]),True)
                            if not np.allclose(score,legacy[tick]['previous_score'],atol=1e-9,rtol=1e-8):
                                raise ValueError(f'Legacy action replay diverged at tick {tick}')
                            bank.append(dict(x=x.tolist(),change=delta.tolist()))
                    a.exchange(u,clinic.FAR)
            finally:
                stereo.close()
            if any(digest_json(x)!=digest_json(r[k]) for x,k in ((a.events,'events'),(a.point_placements(),'placements'))):
                raise ValueError('Extended point did not exactly replay the capped prefix')
            metadata=dict(cards=[c],ready=d['ready'],source=prefix_path.name,source_sha256=proof,
                          variant=dict(grasp='straight',oracle=True),cap=cap,exact_physical_prefix=True,
                          restored_optical_and_memories=True,legacy_memories_restored=len(legacy),
                          prefix_ticks=len(capped['actions']),prefix_legal=capped['score']['learner'],
                          resume_ticks=len(r['actions']))

            def checkpoint(live,actions,plans):
                at=audit(live.events,opening_side=1)
                print('extend',c['id'],at['learner'],at['returns'],len(actions),flush=True)
                row=dict(shot=r['shot'],opponent_seed=c['seed'],actions=actions,records=r['records']+plans,
                         events=live.events,placements=live.point_placements(),score=at)
                mark(live,row,cap)
                if len(actions)>len(r['actions']):
                    _write(output,dict(metadata,complete=False,rows=[row]))
            continuation=continue_point(a,qsurface,r['shot'],c,r['actions'],None,rounds,batch,cap,
                                        initial_bank=bank,warm=(eye,history),progress=checkpoint)
            continuation['records']=r['records']+continuation['records']
            _write(output,dict(metadata,complete=True,rows=[continuation]))
        finally:
            a.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('suffix','dynamic','baseline','extend'))
    p.add_argument('--witness',type=Path,default=SOURCE); p.add_argument('--out',type=Path)
    p.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    p.add_argument('--tick',type=int,default=1797)
    p.add_argument('--rounds',type=int,default=6); p.add_argument('--batch',type=int,default=160)
    p.add_argument('--cap',type=int,default=50)
    p.add_argument('--prefix',type=Path)
    p.add_argument('--start',type=int,default=0); p.add_argument('--count',type=int)
    p.add_argument('--seed-shift',type=int,default=0)
    p.add_argument('--release',type=int)
    p.add_argument('--automatic-reception',action='store_true')
    p.add_argument('--forward-window',action='store_true')
    args=p.parse_args()
    out=args.out or OUT/f'motorcoach-{args.phase}.json.gz'
    out.parent.mkdir(parents=True,exist_ok=True)
    if args.phase=='suffix':
        suffix(args.witness,out,args.tick,args.rounds,args.batch,cap=args.cap,release=args.release,
               automatic=args.automatic_reception,forward=args.forward_window)
    elif args.phase=='extend':
        if args.prefix is None:
            p.error('extend needs --prefix')
        extend(args.prefix,out,args.rounds,args.batch,args.cap)
    else:
        live(args.feeds,out,args.rounds,args.batch,control=args.phase=='baseline',cap=args.cap,
             start=args.start,count=args.count,seed_shift=args.seed_shift)
