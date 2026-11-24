"""Development-only late-start coach on three exact low-ball prefixes."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_fast_camera import native_ball_pixels
from .tt29_rally_physics import Rollout
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, resources
from .tt36_impact import triangulate
from .tt37_motor_clinic import landing_error
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt38_motorcoach import AIM, SIZES, answerable, schedules
from .tt40_long_low import ORDER, OUT, ease, indexed, physical_finish
from .tt40_low_feasibility import SOURCE, replay_prefix, roster

DEST = Path('artifacts/tt41')
PROOFS = ['long-low', 'long-low-flipped', 'long-low-alternate-strict']
RATE = .8
SLEW = .32


class TurnGate:
    """Three consecutive stereo positions certify inbound motion after seeing outbound flight."""
    def __init__(self):
        self.armed = False
        self.last = []

    def update(self, tick, point, dt):
        if point is None:
            self.last.clear()
            return False
        self.last.append((tick, np.asarray(point)))
        self.last = self.last[-3:]
        if len(self.last) < 3:
            return False
        t = np.array([r[0] for r in self.last])
        y = np.array([r[1][1] for r in self.last])
        if np.any(np.diff(t) != 2):
            return False
        delta = np.diff(y)
        if y[-1] > -.18 and np.all(delta > .020) and (y[-1]-y[0])/((t[-1]-t[0])*dt) > .9:
            self.armed = True
        return (self.armed and -.87 < y[-1] < 1.6 and np.all(delta < -.025)
                and (y[-1]-y[0])/((t[-1]-t[0])*dt) < -.9)


def color_pos(view, rig):
    imgs = view.capture()
    rays = [native_ball_pixels(imgs[s]) for s in ('left', 'right')]
    if any(r is None or r.clipped for r in rays):
        return None
    return triangulate([r.center for r in rays], rig)


def optical_gate(a, view, rig, old, c, begin, end):
    """Start watching after the preceding strike, then decide with frames only."""
    replay_prefix(a,old,c,begin)
    obs = TurnGate()
    frames = []
    result = None
    for tick in range(begin,end+1):
        if tick % 2 == 0:
            p = color_pos(view,rig)
            frames.append([tick, None if p is None else p.tolist()])
            if obs.update(tick,p,a.control_dt):
                result = tick
                break
        a.exchange(old['actions'][tick],FAR_COMMAND)
    return result, frames


def bound(goals, q0, last, stride, low, high):
    """Slew and magnitude bound on the direct command, not on actuator tracking error."""
    result = np.empty_like(goals)
    q = q0.copy()
    u = np.clip(last,-RATE,RATE)
    for i in range(goals.shape[1]):
        request = (goals[:,i,:]-q)/stride
        u = np.clip(np.clip(request,u-SLEW,u+SLEW),-RATE,RATE)
        # Begin braking before meeting the hard motor reference limit. At
        # most three steps to zero; budget the exact discrete stopping distance.
        for sign, edge in ((1,high),(-1,low)):
            room = sign*(edge-q)/stride
            cap = np.minimum(room,np.minimum((room+SLEW)/2,(room+3*SLEW)/3))
            u = sign*np.minimum(sign*u,np.maximum(0.,cap))
        q = np.clip(q+u*stride,low,high)
        result[:,i,:] = q
    return result


def clean_proofs(a, k, old, c, views):
    """Confirmed signed alternatives and their issued reference trajectories."""
    a.sample_clearance(False)
    axes = a._arm_ids[1][2]
    priors, recipes = [], []
    for key, doc in views.items():
        for r in doc['rows']:
            if r['id'] != c['id'] or not r['wins']:
                continue
            for b in r['all_clear_routed_possibilities']:
                priors.append(dict(b,absolute_eta=r['late_tick']*a.control_dt+b['eta']))
            for win in r['wins']:
                if not (win['legal'] and win['answered']):
                    continue
                b = win['chosen']['variant']
                priors.insert(0,dict(b,absolute_eta=r['late_tick']*a.control_dt+b['eta'],
                    params=win['chosen']['parameters'], proof=key))
                end = int(np.ceil(win['placements']['delivered'][k]['near_contact']['time']/a.control_dt))
                replay_prefix(a,old,c,r['fork_tick'])
                goals = [a.data.ctrl[axes].copy()]
                for u in win['actions'][r['fork_tick']:end+12]:
                    a.exchange(u,FAR_COMMAND)
                    goals.append(a.data.ctrl[axes].copy())
                recipes.append(dict(goals=np.array(goals),start=r['fork_tick'],contact=end,proof=key))
    if not recipes:
        raise ValueError('Missing earlier positive feasibility control')
    return priors,recipes


def build_targets(q0, priors, recipes, code, params, tick, dt, length, route):
    j = code % len(priors)
    q = np.asarray([priors[h]['q'] for h in j])
    dq = np.asarray([priors[h]['dq'] for h in j])
    etas = np.asarray([priors[h]['absolute_eta']-tick*dt for h in j])
    gs = schedules(q,dq,etas,params[:,:len(SIZES)],dt,length)
    t=(np.arange(length)+1)*dt
    # Alternate a front-loaded reach, the unsmoothed winning stroke target,
    # and a compressed earlier course spliced to its original strike phase.
    for k in range(len(gs)):
        if route[k]==1:
            front = max(.04,etas[k]-.067+params[k,-1]*.042)
            w = ease(t/front)**(.57+.40*abs(params[k,-2]))
            g = gs[k,max(0,min(length-1,int(front/dt)))]
            m = t<front
            gs[k,m,:5] = (1-w[m,None])*q0[:5] + w[m,None]*g[:5]
        if route[k]==2:
            rec = recipes[code[k] % len(recipes)]
            contact = rec['contact']
            off = np.clip(params[k,10]*28.,-2.5,2.5)
            release = max(1,contact-tick-3+off)
            index = np.arange(length)+1.
            s = np.minimum(1., index/release)**(.56+.68*abs(params[k,-2]))
            splice = contact-3-rec['start']
            origin = (.07+.18*abs(params[k,-1]))*splice
            j2 = np.where(index<release,origin+s*(splice-origin),splice+index-release)
            for axis in range(5):
                gs[k,:,axis] = np.interp(j2,np.arange(len(rec['goals'])),rec['goals'][:,axis]) + params[k,axis]
    return gs


def search(a, engine, priors, recipes, tick, last, batch, rounds, seed):
    tm=process_time()
    axes=a._arm_ids[1][2]
    dt=a.control_dt
    stride=a._joint_increments[axes]*(dt/a.model.opt.timestep)
    lo,hi=a._joint_low[axes],a._joint_high[axes]
    q0=a.data.ctrl[axes].copy()
    rng=np.random.default_rng(seed)
    stock_proof=[p for p in priors if 'proof' in p]
    # Preserve both faces and deep alternatives among well-routed source IK points.
    plausible=[p for p in priors if tick*dt < p['absolute_eta'] < tick*dt+.72]
    rank=lambda p: np.max(abs(np.array(p['q'])-q0[:5])/(RATE*stride[:5]/dt))-(p['absolute_eta']-tick*dt)
    tests=stock_proof+[p for p in sorted(plausible,key=rank) if 'proof' not in p][:22]
    if not tests:
        return [],[],[],0.
    cutoff=min(.79,max(p['absolute_eta'] for p in tests)-tick*dt+.12)
    alternatives=[dict(p,reference_budget_deficit_s=float(rank(p))) for p in tests]
    length=int(np.ceil((cutoff+.94)/dt))
    raw=qmc.Sobol(15,scramble=True,seed=int(rng.integers(1,2**31))).random_base2(
        int(np.ceil(np.log2(batch))))[:batch]
    z=(2*raw-1)*np.r_[SIZES*.67,1.,1.]
    codes=np.arange(batch)%len(tests)
    modes=(np.arange(batch)//len(tests))%3
    for i in range(min(len(tests)*3,batch)):
        z[i,:len(SIZES)]=np.asarray(tests[codes[i]].get('params',np.zeros(len(SIZES)))) if modes[i]!=2 else 0.
    seen,qual,notes=[],[],[]
    for rnd in range(rounds):
        g=build_targets(q0,tests,recipes,codes,z,tick,dt,length,modes)
        g=bound(g,q0,last,stride,lo,hi)
        f=np.zeros((batch,9))
        a.clearance_margin=.0006
        try:
            rate,value=engine.trajectories(g,AIM,cutoff,feedback=f)
        finally:
            a.clearance_margin=0.
        hit=f[:,1]>.5
        legal=np.isfinite(value[:,0])
        # After the first weak touch, optimize its departure rather than spending
        # the remaining budget on near misses with a slightly smaller gap.
        grade=np.where(legal,answerable(value)-100.,
                       np.where(hit,np.minimum(180.,landing_error(f))+2*f[:,8],200.+36*f[:,0]))
        for i in np.flatnonzero(legal):
            tape=rate[i,:max(1,int(np.ceil((value[i,1]+.15)/dt)))]
            if np.max(abs(tape))>RATE+1e-9 or np.max(abs(np.diff(np.vstack([last,tape,np.zeros(6)]),axis=0)))>SLEW+1e-9:
                continue
            quality=float(answerable(value[i]))
            if any(np.shape(b['tape'])==tape.shape and np.allclose(b['tape'],tape) for b in qual):
                continue
            qual.append(dict(tape=tape.tolist(),cost=quality,outcome=value[i].tolist(),variant=tests[codes[i]],
                            mode=int(modes[i]),params=z[i].tolist(),feedback=f[i].tolist()))
        notes.append(dict(rnd=rnd,tested=batch,touched=int(hit.sum()),legal=int(legal.sum()),
                          closest=float(f[:,0].min()),best=float(grade.min())))
        seen.extend(zip(grade,codes.copy(),modes.copy(),z.copy(),strict=True))
        if len(qual)>=6 and min(b['cost'] for b in qual)<2.3:
            break
        top=sorted(seen,key=lambda x:x[0])[:min(20,len(seen))]
        selected=rng.integers(len(top),size=batch)
        codes=np.array([top[i][1] for i in selected])
        modes=np.array([top[i][2] for i in selected])
        z=np.array([top[i][3] for i in selected])
        z+=rng.normal(size=z.shape)*np.r_[SIZES,np.ones(2)]*(.62 if rnd<2 else .36)
        z=np.clip(z,-2.7*np.r_[SIZES,np.ones(2)],2.7*np.r_[SIZES,np.ones(2)])
        z[:len(top)]=[x[3] for x in top]
        codes[:len(top)]=[x[1] for x in top]
        modes[:len(top)]=[x[2] for x in top]
        # Leave a minority of each round exploring other initial families.
        nr=batch//7
        if nr:
            codes[-nr:]=rng.integers(len(tests),size=nr)
            modes[-nr:]=rng.integers(3,size=nr)
    return sorted(qual,key=lambda p:p['cost']),notes,alternatives,process_time()-tm


def guarded_prefix(a, clear, old, c, tick):
    # Step zero invokes launch; enable full-CAD sampling for the immutable tape too.
    replay_prefix(a,old,c,0)
    guarded(a,clear)
    for u in old['actions'][:tick]:
        a.exchange(u,FAR_COMMAND)
    initial=[e for e in old['events'] if e['time']<=float(a.data.time)+1e-8]
    if digest_json(initial)!=digest_json(a.events):
        raise ValueError('Full CAD guard changed stock physics')


def physical_test(a,clear,old,c,leg,tick,choice):
    contacts=[]
    with contact_observer(a,contacts):
        guarded_prefix(a,clear,old,c,tick)
        real=physical_finish(a,old,c,leg,old['actions'][:tick],choice)
        verify_clearance(a)
    ruling=a.clearance_result()
    real['clearance']=ruling
    real['contacts']=contacts
    real['fork_tick']=tick
    replies=real['placements']['delivered']
    real['clean_answer']=bool(real['answered'] and leg+1<len(replies)
                              and replies[leg+1]['near_bounce'] and ruling['penetration_samples']==0)
    return real


def run(output,batch,rounds,cases,check,only,gates_only=False,resume=False):
    d=_read(SOURCE)
    if d.get('group')!='practice':
        raise ValueError('Only named development data may supply counterfactual prefixes')
    proofs={p:_read(OUT/(p+'.json.gz')) for p in PROOFS}
    if any(p['source_sha256']!=_hash(SOURCE) for p in proofs.values()):
        raise ValueError('Earlier proofs came from a different source')
    mapping={r['id']:r for r in roster(d)}
    doc=dict(schema='tt41-low-deadline-v1',role='offline privileged feasible timing witnesses; no runtime policy',
             ready=d['ready'],source_sha256=_hash(SOURCE),proof_hashes={p:_hash(OUT/(p+'.json.gz')) for p in proofs},
             rate_bound=RATE,rate_change_per_tick=SLEW,batch=batch,rounds=rounds,rows=[])
    if resume and output.exists():
        prior=_read(output)
        if any(doc[k]!=prior[k] for k in ('source_sha256','proof_hashes','batch','rounds')):
            raise ValueError('Resume expects the same source, proof hashes and budget')
        doc['rows']=prior['rows']
    _write(output,doc)
    with centered(d['ready']):
        a=arena(protocol='random')
        cad=arena(protocol='random',detail='cad')
        view,rig,*_=resources(a)
        clear=CourtClearance(cad.model,articulated=True)
        engine=None if gates_only else Rollout(a,1)
        axes=a._arm_ids[1][2]
        dt=a.control_dt
        try:
            for id in ORDER:
                if cases and id not in cases:
                    continue
                t=mapping[id]; k=t['leg']; c=d['cards'][t['index']]; old=d['rows'][t['index']]
                off=old['placements']['delivered'][k]
                fork,late,_=indexed(old['events'],off,dt)
                far=float(off['far_contact']['time'])
                sep=next(e['time'] for e in old['events'] if far<e['time']<far+.06 and
                         e['kind']=='separation' and e.get('collider')=='rival_racket')
                far_tick=int(np.ceil((sep+1e-7)/dt))+1
                hit_tick=int(np.ceil((far-1e-7)/dt))
                # For the literal first control tick after hit, refuse to shadow
                # an unresolved far contact with its blade parked in isolation.
                immediate=hit_tick if hit_tick*dt>sep+1e-8 else None
                early,frames=optical_gate(a,view,rig,old,c,fork,late-1)
                if early is not None and early < far_tick:
                    raise ValueError('Verified turn gate fired before completed far impact')
                print(id,'far',far_tick,'optical',early,'table',late,'actual far time',far,flush=True)
                previous=next((r for r in doc['rows'] if r['id']==id),None)
                row=dict(id=id,card=c,index=t['index'],leg=k,projected_height_m=t['height'],
                         original_legal=bool(off['near_landed']),earlier_control_fork_tick=fork,
                         previous_earliest_proof_tick=fork,near_bounce_tick=late,
                         actual_far_contact_s=far,actual_far_separation_s=sep,frames=frames,stages=[])
                if previous is None:
                    doc['rows'].append(row)
                else:
                    targets=dict(rgb=early,far_separation=far_tick,first_tick_after_hit=immediate)
                    row['stages']=[s for s in previous['stages'] if
                                   s['fork_tick'] is not None and s['fork_tick']==targets[s['name']]
                                   and s.get('shadow')]
                    doc['rows'][doc['rows'].index(previous)]=row
                if gates_only:
                    _write(output,doc)
                    continue
                priors,recipes=clean_proofs(a,k,old,c,proofs)
                ordered=[('rgb',early),('far_separation',far_tick)]
                if only=='first_tick_after_hit':
                    ordered=[('first_tick_after_hit',immediate)]
                for name,tick in ordered:
                    if only and only!=name:
                        continue
                    if any(s['name']==name for s in row['stages']):
                        continue
                    info=dict(name=name,fork_tick=tick,wins=[],physical_attempts=[])
                    row['stages'].append(info)
                    if tick is None:
                        _write(output,doc)
                        continue
                    digest=replay_prefix(a,old,c,tick)
                    q=a.data.ctrl[axes].copy()
                    u=np.asarray(old['actions'][tick-1])
                    incoming=dict(ball=a.ball.tolist(),velocity=a.velocity.tolist(),spin=a.spin.tolist())
                    found,trace,geometry,cpu=search(a,engine,priors,recipes,tick,u,batch,rounds,[410950,id,tick])
                    info.update(prefix_sha256=digest,encoder=a.joints.tolist(),initial_ctrl=q.tolist(),
                                absolute_s=tick*dt,lag_after_far_s=tick*dt-far,initial_last_command=u.tolist(),
                                shadow=trace,geometry=geometry,teacher_only_initial_flight=incoming,cpu_s=cpu,
                                private_total_legal=len(found))
                    print(id,name,trace,'private unique legal',len(found),flush=True)
                    for b in found[:check]:
                        r=physical_test(a,clear,old,c,k,tick,b)
                        info['physical_attempts'].append({s:r[s] for s in ('legal','answered','clearance','clean_answer','chosen')})
                        print(id,name,'physical',r['clean_answer'],r['answered'],r['clearance'],flush=True)
                        if r['clean_answer']:
                            info['wins'].append(r)
                            break
                    _write(output,doc)
        finally:
            view.close()
            a.close()
            cad.close()
    doc['complete']=True
    _write(output,doc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEST/'deadline-stock.json.gz')
    p.add_argument('--batch',type=int,default=192)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--check',type=int,default=5)
    p.add_argument('--cases',nargs='*',type=int)
    p.add_argument('--only',choices=('rgb','far_separation','first_tick_after_hit'))
    p.add_argument('--gates-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    run(a.output,a.batch,a.rounds,a.cases,a.check,a.only,a.gates_only,a.resume)
