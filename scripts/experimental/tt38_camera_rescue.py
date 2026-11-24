"""Bounded dynamic rescue in a camera/encoder-initialized, never-launched court copy."""

import numpy as np
from scipy.stats import qmc
from pathlib import Path

from .tt37_motor_clinic import geometry, landing_error
from .tt38_motorcoach import AIM, SIZES, answerable, schedules


DEPTHS=(-1.27,-1.37,-1.45,-1.50,-1.56,-1.62,-1.68)
MIN_FEEDBACK=7


def unit(x):
    v=np.asarray(x,dtype=float)
    return v/max(1e-9,np.linalg.norm(v))


def desired_velocity(position, pace):
    p=np.asarray(position)
    t=max(.29,(.66-p[1])/(pace*.87))
    net_t=max(.16,-p[1]/(pace*.92))
    z=max((.78-p[2])/t+4.905*t,(1.015-p[2])/net_t+4.905*net_t)
    return np.array([-p[0]/t,pace,z])


def basis(engine,surface,obs):
    """Alternative contact windows from the independently initialized estimate only."""
    t=engine.twin
    eta=float(obs[0]); q,dq=geometry(surface,obs)
    bases=[dict(eta=eta,depth=float(surface.depth),pose=q,velocity=.86*dq,anchor='clinic')]
    path=t.partner.flight.path(t.ball,t.velocity,t.spin,1,bounced=t._received_bounce,horizon=.45)
    path=path[(path[:,0]>=.031)&(path[:,0]<.41)&(path[:,3]>.823)&(path[:,3]<1.37)&(abs(path[:,1])<.46)]
    if not len(path):
        return bases
    # Avoid giving every base the same offset if it is really the same trajectory sample.
    use=[]
    for depth in DEPTHS:
        if not path[:,2].min()-.011 < depth < path[:,2].max()+.011:
            continue
        row=path[np.argmin(abs(path[:,2]-depth))]
        if any(abs(row[0]-old[0])<.009 for old in use):
            continue
        use.append(row)
        now,x,y,z,vx,vy,vz=row[:7]
        legacy=unit([-.18*x-.085*vx,1.,.29-.095*vz])
        impulse=unit(desired_velocity([x,y,z],4.6)-[vx,vy,vz])
        for alpha in (.15,.65):
            face=unit((1.-alpha)*legacy+alpha*impulse)
            q,dq=surface.solution(x,z,face,depth=y)
            bases.append(dict(eta=float(now),depth=float(y),pose=q,velocity=.86*dq,anchor='optical-flight'))
    return bases


def measurements(f):
    hit=f[:,1]>.5
    category=np.where(~hit,0,np.where(f[:,3]<=.85,1,2))
    loss=np.maximum(f[:,0],0.)
    for i in np.flatnonzero(hit):
        targets=[desired_velocity(f[i,5:8],p) for p in (3.7,4.4,5.15)]
        error=min(np.linalg.norm((f[i,2:5]-v)*[.88,1.,1.12]) for v in targets)
        # Ranking is within each material outcome stratum, not against a gap.
        loss[i]=.57*error+.43*min(12.,landing_error(f[i:i+1])[0])+2.*f[i,8]
    return category,loss


def stratified(history,groups=8):
    elites={}
    for category in (0,1,2):
        best=[]
        for row in sorted((x for x in history if x['kind']==category),key=lambda x:x['loss']):
            if any(row['slot']==x['slot'] and np.linalg.norm((row['params']-x['params'])/SIZES)<.40 for x in best):
                continue
            best.append(row)
            if len(best)>=groups:
                break
        if best:
            elites[category]=best
    return elites


def inverse_steps(history,elites,bases,rng):
    contacts=[r for r in history if r['kind']]
    if len(contacts)<MIN_FEEDBACK:
        return []
    proposals=[]
    for k in (1,2):
        for e in elites.get(k,[])[:3]:
            center=e['params']/SIZES
            z=bases[e['slot']]['depth']
            near=[h for h in contacts if abs(bases[h['slot']]['depth']-z)<.086]
            if len(near)<MIN_FEEDBACK:
                continue
            x=np.array([h['params']/SIZES for h in near]); v=np.array([h['feedback'][2:5] for h in near])
            d=np.linalg.norm(x-center,axis=1)+np.array([.22 if h['slot']!=e['slot'] else 0. for h in near])
            ii=np.argsort(d)[:40]; x=x[ii]; v=v[ii]
            w=np.exp(-.5*(d[ii]/max(.35,np.median(d[ii])))**2); w/=w.sum()
            dx=x-w@x; dv=v-w@v
            slope=np.linalg.solve(dx.T@(w[:,None]*dx)+.065*np.eye(len(SIZES)),dx.T@(w[:,None]*dv)).T
            predicted=w@v+slope@(center-w@x)
            for pace in (3.9,4.7,5.15):
                target=desired_velocity(e['feedback'][5:8],pace)
                correction=slope.T@np.linalg.solve(slope@slope.T+.25*np.eye(3), target-predicted)
                correction=np.clip(correction,-.95,.95)
                choice=np.clip(center+.92*correction+rng.normal(0,.07,len(SIZES)),-2.6,2.6)*SIZES
                proposals.append((e['slot'],choice))
    return proposals


def costs(engine,outcomes):
    value=getattr(engine,'motor_value',None)
    quality=answerable(outcomes)
    if value is None:
        return .62*outcomes[:,0]+.38*quality
    return .73*value.cost(outcomes)+.27*quality


class CameraRescue:
    """Search exploration uses a private stream, never a game or release seed."""

    def __init__(self, *, search_seed, batch=56, rounds=3):
        if batch<24 or rounds<1:
            raise ValueError('Rescue needs 24 trials per round and at least one round')
        self.batch=int(batch)
        self.rounds=int(rounds)
        self.rng=np.random.default_rng(search_seed)
        self.searches=0

    @staticmethod
    def _output(trace,note,tape=None,score=None,eta=None):
        return dict(tape=tape,score=score,eta=eta,trace=trace,note=note)

    def plan(self,engine,surface,obs,fallback):
        if engine is None or not hasattr(engine,'twin'):
            raise ValueError('A private camera court model is required')
        t=engine.twin
        if t.card is not None or t.launches!=0:
            raise ValueError('Camera rescue may not read a launched court or placement card')
        self.searches+=1
        private=int(self.rng.integers(1,2**31))
        rng=np.random.default_rng(private)
        obs=np.asarray(obs,dtype=float)
        note=dict(status='unresolved',ordinal=self.searches,private_stream_draw=private)
        if (obs.ndim!=1 or len(obs)<6 or not np.isfinite(obs[:6]).all() or not (.029<obs[0]<.36)
                or engine.latest is None or engine.tick*t.control_dt-engine.latest.time>.11):
            note['status']='no current sensory estimate'
            return self._output([],note)
        obs=obs[:6]
        bases=basis(engine,surface,obs)
        note['bases']=len(bases)
        b=self.batch
        draws=qmc.Sobol(13,scramble=True,seed=private).random_base2(int(np.ceil(np.log2(b))))[:b]
        offsets=(2.*draws-1.)*SIZES
        slots=np.arange(b,dtype=int)%len(bases)
        offsets[:len(bases)]=0.
        if fallback is not None and np.asarray(fallback).shape==(10,) and np.isfinite(fallback).all():
            i=min(b-1,len(bases))
            slots[i]=0
            offsets[i]=np.r_[fallback,0.,0.,0.]
        dt=t.control_dt
        q=np.array([p['pose'] for p in bases]); dq=np.array([p['velocity'] for p in bases])
        times=np.array([p['eta'] for p in bases]); cutoff=float(times.max()+.115)
        frames=round((cutoff+1.05)/dt)
        rows=[]; trace=[]
        for k in range(self.rounds):
            goals=schedules(q[slots],dq[slots],times[slots],offsets,dt,frames)
            feedback=np.zeros((len(goals),9))
            _,truth=engine.trajectories(goals,AIM,cutoff,feedback=feedback)
            group,loss=measurements(feedback)
            legal=np.flatnonzero(np.isfinite(truth[:,0]))
            counts=np.bincount(group,minlength=3).tolist()
            trace.append(dict(trials=len(goals),contacts=counts[1]+counts[2],legal=len(legal),
                gap_only=counts[0],weak_or_backward=counts[1],forward_touches=counts[2],
                closest=float(feedback[:,0].min()),stage='camera_rescue'))
            if len(legal):
                ix,stability=self.robust_pick(engine,goals,truth,legal,cutoff,rng,trace)
                n=min(frames,max(1,round((truth[ix,1]+.084)/dt)))
                base=bases[int(slots[ix])]
                note.update(status='selected',deadline_eta=float(truth[ix,1]),legal_nominal=len(legal),
                            base=base['anchor'],depth=float(base['depth']),tested_rounds=k+1,
                            stability=stability,coefficients=offsets[ix].tolist())
                return self._output(trace,note,goals[ix,:n].copy(),truth[ix].tolist(),float(truth[ix,1]))
            rows.extend(dict(kind=int(j),loss=float(l),slot=int(s),params=p.copy(),feedback=f.copy())
                        for j,l,s,p,f in zip(group,loss,slots,offsets,feedback,strict=True))
            if k+1>=self.rounds:
                break
            elite=stratified(rows)
            order=list(elite)
            parents=[]
            for j in range(b):
                candidates=elite[order[j%len(order)]]
                parents.append(candidates[rng.integers(len(candidates))])
            slots=np.array([r['slot'] for r in parents],dtype=int)
            offsets=np.array([r['params'] for r in parents])+rng.normal(size=(b,13))*SIZES*(.58 if k==0 else .40)
            offsets=np.clip(offsets,-2.6*SIZES,2.6*SIZES)
            kept=[v for p in elite.values() for v in p[:2]]
            for i,e in enumerate(kept):
                slots[i],offsets[i]=e['slot'],e['params']
            inverse=inverse_steps(rows,elite,bases,rng)[:max(0,b//4)]
            for i,(slot,choice) in enumerate(inverse,len(kept)):
                slots[i],offsets[i]=slot,choice
            trace[-1]['retained_strata']={str(key):len(val) for key,val in elite.items()}
            trace[-1]['signed_inverse_proposals']=len(inverse)
        note['tested_rounds']=self.rounds
        return self._output(trace,note)

    def robust_pick(self,engine,goals,truth,legal,cutoff,rng,trace):
        nominal=costs(engine,truth[legal]); contenders=legal[np.argsort(nominal)[:6]]
        weights=costs(engine,truth[contenders]); stable=np.ones(len(contenders),int)
        t=engine.twin
        p,v=t.ball.copy(),t.velocity.copy()
        total_contact=0; total_legal=0
        try:
            for s in (-1.,1.):
                t.ball[:]=p+s*np.array([.0018,0.,.0028])
                t.velocity[:]=v+s*np.array([.012,.035,.083])
                f=np.zeros((len(contenders),9))
                _,out=engine.trajectories(goals[contenders],AIM,cutoff,feedback=f)
                hit=np.isfinite(out[:,0]); stable+=hit
                weights[hit]+=.32*costs(engine,out[hit]); weights[~hit]+=1.38+.13*np.minimum(8.,landing_error(f[~hit]))
                total_legal+=int(hit.sum()); total_contact+=int(f[:,1].sum())
        finally:
            t.ball[:]=p; t.velocity[:]=v
        selected=int(np.argmin(weights))
        trace.append(dict(trials=2*len(contenders),contacts=total_contact,legal=total_legal,stage='camera_rescue_uncertainty',
                          robust_settings=int(stable[selected])))
        return int(contenders[selected]),int(stable[selected])


def reproduction(source, output, seed):
    """Targeted audit of the earlier 6785 optical failure, not whole-point evaluation."""
    from time import process_time
    from . import tt33_joint as joint
    from .tt4_transfer_trial import _read,_write
    from .tt19_view_full_game import digest_json
    from .tt29_endurance import audit
    from .tt31_ready import centered
    from .tt34_latency import PUPIL
    from .tt37_belief import ArrivalHistory
    from .tt37_opponent import arena
    from .tt37_optical_clinic import OpticalClinic
    from .tt37_visual_inertia import LearnedOptics
    from .tt38_motor_value import MotorValue
    from .tt38_motorcoach import setup

    old=_read(source)
    i=next(i for i,c in enumerate(old['cards']) if c['id']==6785)
    card=old['cards'][i]; row=old['rows'][i]; home=np.asarray(old['ready'])
    wanted=[r for r in row['records'] if r['score'] is None][-2:]
    if not wanted:
        raise ValueError('The archived terminal rescue opportunity has changed')
    targets={r['tick']:r for r in wanted}
    surface=setup(old['depth'])
    joint.RECEIVE_Y=surface.depth
    remedy=CameraRescue(search_seed=seed,batch=56,rounds=3)
    started=process_time()
    with centered(home):
        court=arena()
        engine=OpticalClinic(body=True)
        engine.motor_value=MotorValue(_read(Path(old['motor_value'])))
        renderer,rig,stride,lo,hi=joint.resources(court)
        actor=joint.JointPolicy(_read(PUPIL)|{'ready':home.tolist()},stride,lo,hi)
        eye=joint.Vision(rig,court.control_dt)
        eye.eye=LearnedOptics(eye.eye,old['visual'])
        memory=ArrivalHistory()
        observations=[]
        try:
            court.launch(row['shot'],card=card)
            court.partner.reset(seed=card['seed'])
            actions=[]
            for tick,command in enumerate(row['actions'][:max(targets)+1]):
                q=court.joints.copy(); stamp=tick*court.control_dt
                state=eye.update(tick,renderer.capture() if tick%2==0 else None,q)
                state=memory.update(stamp,state)
                engine.observe(tick,eye.eye,q,actor.reference.copy())
                if tick in targets:
                    earlier=[e for e in row['events'] if e['time']<=court.data.time+1e-8]
                    if digest_json(court.events)!=digest_json(earlier):
                        raise ValueError('Physical replay changed before the rescue')
                    base=targets[tick]
                    private=remedy.plan(engine,surface,state,base['action']) if engine.sync() else None
                    raw_context=None if state is None else np.r_[state[1:6],q[:5]].tolist()
                    record=dict(tick=tick,archived_context=base['context'],rgb_encoder_context=raw_context,
                                archive_legal_choices=base['good'],rescue_context_matches=bool(raw_context is not None and np.allclose(raw_context,base['context'],atol=1e-8)),
                                rescue=None if private is None else private['note'],
                                batches=[] if private is None else private['trace'],
                                private_legal=None if private is None else private['score'])
                    observations.append(record)
                    print(tick,record['rescue'],flush=True)
                    if private is not None and private['tape'] is not None:
                        # Physically attempt only the first permitted selection. The real plant receives motor rates, not state writes.
                        near_before=audit(court.events,opening_side=1)['learner']
                        suffix=[]
                        for u in private['tape']:
                            cmd=actor.drive(u,court.joints.copy(),.8)
                            _,_,done,truncated,_=court.exchange(cmd,joint.FAR_COMMAND)
                            suffix.append(cmd.tolist())
                            if done or truncated:
                                break
                        for _ in range(110):
                            a=audit(court.events,opening_side=1)
                            if court.terminated or court.truncated or (a['learner']>near_before and a['opponent']>=near_before+2):
                                break
                            idle=np.zeros(6)
                            court.exchange(idle,joint.FAR_COMMAND)
                            suffix.append(idle.tolist())
                        ended=audit(court.events,opening_side=1)
                        record['physical']=dict(own_before=near_before,own_after=ended['learner'],opponent_after=ended['opponent'],
                                                point_ended=bool(court.terminated or court.truncated),exact_prefix=len(actions),
                                                events_after= court.events[len(earlier):])
                        result=dict(card=card,shot=row['shot'],ready=home.tolist(),actions=actions+suffix,events=court.events,
                                    placements=court.point_placements(),score=ended,opponent_seed=card['seed'])
                        _write(output,dict(source_row_sha256=digest_json(row),observations=observations,search_seed=seed,
                            terminal_prefix_only=True,physical=result,cpu_s=process_time()-started))
                        return
                court.exchange(command,joint.FAR_COMMAND)
                actions.append(command)
                actor.reference=np.clip(actor.reference+stride*np.asarray(command),lo,hi)
        finally:
            renderer.close()
            engine.twin.close()
            court.close()
    _write(output,dict(source_row_sha256=digest_json(row),observations=observations,search_seed=seed,
                        terminal_prefix_only=True,physical=None,cpu_s=process_time()-started))


def summary(reference, tested):
    """Compare the predeclared paired card order. Physical state is opened only for evaluation."""
    from .tt4_transfer_trial import _read,_hash
    from .tt19_view_full_game import digest_json
    a,b=_read(reference),_read(tested)
    if a['cards']!=b['cards'] or len(a['rows'])!=len(a['cards']) or len(b['rows'])!=len(b['cards']):
        raise ValueError('Pair must have complete matched game ledgers')
    options=('ready','depth','cut','mode','economy','memory_start','camera_stride','repair','robust','body',
             'visual','refresh','variable','cap','refine','rounds','samples','guided','group','motor_value')
    if any(a.get(k)!=b.get(k) for k in options):
        raise ValueError('Default controller settings differ apart from rescue')
    if bool(a.get('forget_between_points'))!=bool(b.get('forget_between_points')):
        raise ValueError('Point-to-point practice setting differs')
    n=a['memory_start']
    if a['bank'][:n]!=b['bank'][:n] or any(ra['shot']!=rb['shot'] for ra,rb in zip(a['rows'],b['rows'],strict=True)):
        raise ValueError('Starting practice memories or releases differ')
    pairs=[]
    for c,ra,rb in zip(a['cards'],a['rows'],b['rows'],strict=True):
        x,y=ra['placements'],rb['placements']
        common=list(zip(x['requested'],y['requested']))
        if any(t['target']!=u['target'] for t,u in common):
            raise ValueError('The intervention changed an opponent random request')
        keys=('target','far_contact','near_bounce','receive_line')
        first=bool(x['delivered'] and y['delivered'] and all(
            digest_json(x['delivered'][0][k])==digest_json(y['delivered'][0][k]) for k in keys))
        rescues=[r for r in rb['records'] if r.get('camera_rescue') is not None]
        installed=[r for r in rescues if r['camera_rescue']['status']=='selected']
        effects=[]
        def play_after(row, start, end):
            hit=next((s for s in row['score']['shots'] if s['side']==1 and start<s['hit']<=end),None)
            touches=[e for e in row['events'] if e['kind']=='contact' and e.get('side')==1 and start<e['time']<=end]
            separation=None if not touches else next((e for e in row['events'] if e['kind']=='separation'
                and e.get('collider')=='learner_racket' and touches[0]['time']<e['time']<=end+.035),None)
            point=next((e for e in reversed(row['events']) if e['kind']=='point'),None)
            return dict(touch_s=None if not touches else touches[0]['time'],
                        outgoing_mps=None if separation is None else separation['velocity'],
                        legal=hit is not None,landing=None if hit is None else hit['landing_xyz'],
                        terminal=None if point is None else point['reason'],winner=None if point is None else point['winner'])
        for r in installed:
            start=r['tick']/60
            eta=r['score'][1]
            end=start+eta+.20
            baseline=play_after(ra,start,end)
            realized=play_after(rb,start,end)
            effects.append(dict(tick=r['tick'],predicted_touch_s=start+eta,predicted_landing_xy=r['score'][3:5],
                same_prior_commands=ra['actions'][:r['tick']]==rb['actions'][:r['tick']],
                stability=r['camera_rescue']['stability'],baseline=baseline,rescue=realized))
        trials=[t for r in rescues for t in r['search'] if t.get('stage','').startswith('camera_rescue')]
        near_later=lambda p:sum(e['near_landed'] for e in p['delivered'][1:])
        match_cmd=next((i for i,(u,v) in enumerate(zip(ra['actions'],rb['actions'])) if u!=v),None)
        winner=lambda row: next((e['winner'] for e in reversed(row['events']) if e['kind']=='point'),None)
        pairs.append(dict(id=c['id'],key=c['key'],first_physical_ball_exact=first,matched_request_prefix=len(common),
            baseline_own=ra['score']['learner'],rescue_own=rb['score']['learner'],baseline_total=ra['score']['returns'],
            rescue_total=rb['score']['returns'],baseline_later=near_later(x),rescue_later=near_later(y),
            baseline_winner=winner(ra),rescue_winner=winner(rb),first_command_change=match_cmd,
            rescue_calls=len(rescues),selected=len(installed),legally_realized=sum(e['rescue']['legal'] for e in effects),
            selections=effects,rescue_trials=sum(t['trials'] for t in trials),weak_trials=sum(t.get('weak_or_backward',0) for t in trials),
            contact_trials=sum(t['contacts'] for t in trials),nominal_legal_trials=sum(t['legal'] for t in trials if t['stage']=='camera_rescue'),
            baseline_cpu=ra.get('cpu_s'),rescue_cpu=rb.get('cpu_s')))
    return dict(reference_sha256=_hash(reference),test_sha256=_hash(tested),private_config=b['camera_rescue'],pairs=pairs)


def clearance_replay(paths, point_id):
    """Replay commands with a full CAD safety observer and match the native point ledger."""
    from .tt4_transfer_trial import _read,_hash
    from .tt19_view_full_game import digest_json
    from .tt29_endurance import audit
    from .tt29_partner import CARRIAGE
    from .tt31_ready import centered
    from .tt33_joint import FAR_COMMAND
    from .tt37_opponent import arena
    from .tt37_opponent_audit import CourtClearance,contact_observer,guarded,verify_clearance

    receipts=[]
    for path in paths:
        source=_read(path)
        i=next(i for i,c in enumerate(source['cards']) if c['id']==point_id)
        c,r=source['cards'][i],source['rows'][i]
        with centered(source['ready']):
            a=arena(); cad=arena(detail='cad')
            try:
                a.launch(r['shot'],card=c)
                a.partner.reset(seed=c['seed'])
                guarded(a,CourtClearance(cad.model,articulated=True))
                contacts=[]
                with contact_observer(a,contacts):
                    for u in r['actions']:
                        a.exchange(u,FAR_COMMAND)
                for k,live in (('events',a.events),('placements',a.point_placements()),('score',audit(a.events,opening_side=1))):
                    if digest_json(live)!=digest_json(r[k]):
                        raise ValueError(f'Full CAD observer changed {k}')
                verify_clearance(a)
                if len(contacts)!=sum(e['kind']=='contact' for e in a.events) or any(
                        (p['side']==-1)!=(p['player']==CARRIAGE) for p in contacts):
                    raise ValueError('Contact count or physical body disagrees with referee')
                receipts.append(dict(source=Path(path).name,source_row_sha256=digest_json(r),
                    exact_events_sha256=digest_json(a.events),point=c,own=r['score']['learner'],score=r['score']['returns'],
                    ticks=len(r['actions']),clearance=a.clearance_result(),native_contacts=len(contacts),
                    terminal=next((e for e in reversed(a.events) if e['kind']=='point'),None)))
            finally:
                a.close()
                cad.close()
    return dict(replays=receipts)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    from .tt4_transfer_trial import _write
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('diagnose','summarize','clearance'))
    p.add_argument('reference',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--tested',type=Path)
    p.add_argument('--search-seed',type=int)
    p.add_argument('--point-id',type=int)
    a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.mode=='diagnose':
        if a.search_seed is None:
            p.error('Supply one prospectively chosen private seed')
        reproduction(a.reference,a.output,a.search_seed)
    elif a.mode=='summarize':
        if a.tested is None:
            p.error('Supply the completed matched run')
        _write(a.output,summary(a.reference,a.tested))
    else:
        if a.tested is None or a.point_id is None:
            p.error('Supply the paired physical game archive and one point ID')
        _write(a.output,clearance_replay((a.reference,a.tested),a.point_id))
