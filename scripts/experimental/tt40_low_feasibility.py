"""Privileged physical feasibility of hard low shots on intact matched practice prefixes."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, Rollout
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_motor_clinic import landing_error
from .tt37_opponent import arena
from .tt38_motorcoach import AIM, SIZES, answerable, schedules
from .tt39_opportunity_map import measure, projected_height


SOURCE = Path('artifacts/tt37/mix-rollin.json.gz')
OUT = Path('artifacts/tt40')
DEPTHS = (-1.11, -1.20, -1.29, -1.38, -1.46, -1.54, -1.62, -1.70)
SCALE = np.r_[[.13, .12, .105], [.50, .50, .50]]
ZERO = np.zeros(6)


def roster(data):
    """Choose low misses deterministically, ignoring treatment outcomes and all held files."""
    low = []
    for i, (c, row) in enumerate(zip(data['cards'], data['rows'], strict=True)):
        for j, offer in enumerate(row['placements']['delivered']):
            crossing = offer.get('receive_line')
            if j==0 or crossing is None or offer['near_landed']:
                continue
            z = projected_height(crossing)
            if z < .99:
                low.append(dict(index=i, leg=j, id=c['id'], height=z, x=crossing['xyz'][0]))
    ranked = sorted(low, key=lambda c:(c['height'], c['id']))
    d = list({r['id']:r for r in reversed(ranked)}.values())
    d.sort(key=lambda c:(c['height'], c['id']))
    result = d[:3]
    rest = d[3:]
    if rest:
        left = min(rest,key=lambda p:p['x'])
        right = max(rest,key=lambda p:p['x'])
        result += [left] if left==right else [left,right]
    return result


class Faces(BladeIK):
    """Expose the signed geometric blade axis instead of folding it onto +y."""

    def pose(self, joints, *, jacobian=False):
        result = super().pose(joints, jacobian=jacobian)
        raw = self.data.xmat[self.gripper].reshape(3,3) @ self.normal
        sign = 1. if raw[1] >= 0. else -1.
        fields = result[0], sign*result[1]
        if not jacobian:
            return fields
        return *fields, result[2], sign*result[3]


class Probe:
    def __init__(self,a):
        self.a = a
        self.ik = Faces(a)
        self.core = Rollout(a,1)
        self.axes = a._arm_ids[1][2]
        self.seed_q = qmc.scale(qmc.Sobol(5,scramble=True,seed=400812).random_base2(13),
                                self.ik.low, self.ik.high)
        values = np.array([np.r_[*self.ik.pose(q)] for q in self.seed_q])
        self.index = cKDTree(values/SCALE)

    def basis(self):
        a = self.a
        path = a.partner.flight.path(a.ball, a.velocity, a.spin, 1, bounced=a._received_bounce,horizon=1.10)
        if not len(path):
            return []
        path = path[(path[:,0]>.095) & (path[:,3]>.798) & (path[:,3]<1.26) & (abs(path[:,1])<.52)]
        if not len(path):
            return []
        ready = self.a.joints[:5]
        rows = []
        for depth in DEPTHS:
            if not path[:,2].min()-.018 < depth < path[:,2].max()+.018:
                continue
            i = np.argmin(abs(path[:,2]-depth))
            eta,p,v = path[i,0],path[i,1:4],path[i,4:7]
            for family, pace, desired_pitch in (('rise',4.15,.33),('drive',4.8,.39),('lift',3.8,.49)):
                fly = max(.30,(.70-p[1])/(pace*.89))
                wish = np.array([-.88*p[0]/fly, pace, (.78-p[2])/fly+4.905*fly])
                n = (wish-v) / max(1.,wish[1]-v[1])
                n[2] = .66*n[2]+.34*desired_pitch
                n /= np.linalg.norm(n)
                for face in (-1,1):
                    for lift in (0.,-.025,.023):
                        center = p-.025*n+[0.,0.,lift]
                        _, near = self.index.query(np.r_[center,face*n]/SCALE,8)
                        seeds = [ready, self.seed_q[near[0]]]
                        j = int(np.argmax(np.max(abs(self.seed_q[near]-self.seed_q[near[0]]),axis=1)))
                        if np.linalg.norm(self.seed_q[near[j]]-seeds[-1])>.38:
                            seeds.append(self.seed_q[near[j]])
                        for q0 in seeds:
                            q, (center_real, raw, linear, angular) = self.ik.fit(center,face*n,q0)
                            normal = face*raw
                            ray = p-center_real
                            height = ray@normal
                            tangent = np.linalg.norm(ray-height*normal)
                            error = np.linalg.norm(center-center_real)
                            if (normal@n < .86 or normal[1] < .62 or error>.079
                                    or not -.014 < height < .066 or tangent>.083):
                                continue
                            if np.any(abs(q-ready)>4.30*eta+.24):
                                continue
                            if any(old['face']==face and abs(old['depth']-depth)<.005
                                   and np.linalg.norm(old['q']-q)<.10 for old in rows):
                                continue
                            momentum = .83*normal + np.array([-.18*v[0], .12, .53 if family=='lift' else .23])
                            delta = np.linalg.lstsq(np.r_[linear,.050*angular,.018*np.eye(5)],
                                             np.r_[momentum,np.zeros(8)],rcond=None)[0].clip(-6.,6.)
                            rank = error + .35*tangent + .028*np.linalg.norm(q-ready) + .014*(family=='lift')
                            rows.append(dict(time=eta,depth=depth,q=q,dq=delta,
                                             normal=normal,face=face,family=family,rank=rank))
        return sorted(rows,key=lambda r:r['rank'])

    def search(self, seed, batch, rounds):
        a = self.a
        bases = self.basis()
        if not bases:
            return [], dict(windows=0, rounds=[], positive_candidates=0, negative_candidates=0)
        rng = np.random.default_rng(seed)
        pools = {f:[i for i,b in enumerate(bases) if b['face']==f] for f in (-1,1)}
        # Spend an equal share on both faces if both have a kinematic seed.
        start_groups = [v for v in pools.values() if v]
        ids = np.empty(batch,dtype=int)
        for face_no,pool in enumerate(start_groups):
            ids[face_no::len(start_groups)] = np.resize(pool,len(ids[face_no::len(start_groups)]))
        z = qmc.Sobol(len(SIZES),scramble=True,seed=int(rng.integers(1,2**31))).random_base2(int(np.ceil(np.log2(batch))))
        v = (2*z[:batch]-1)*SIZES*.82
        for i in range(len(ids)):
            if ids[i] not in ids[:i]:
                v[i] = 0.
        q = np.asarray([k['q'] for k in bases]); dq=np.asarray([k['dq'] for k in bases])
        time = np.asarray([k['time'] for k in bases]); dt=a.control_dt; until=float(time.max()+.102)
        n = round((until+.91)/dt)
        reports = []; archive = []; good = []
        total_faces = {-1:0,1:0}
        for k in range(rounds):
            posture = schedules(q[ids],dq[ids],time[ids],v,dt,n)
            feedback = np.zeros((len(v),9))
            rate, values = self.core.trajectories(posture,AIM,until,feedback=feedback)
            legal = np.isfinite(values[:,0])
            dense = landing_error(feedback)
            grade = np.where(feedback[:,1]>0, .8*dense + 4.*feedback[:,8],18.+42.*feedback[:,0])
            choice = np.where(legal,answerable(values)-15.,grade)
            for j in np.flatnonzero(legal):
                b = bases[ids[j]]
                end = max(1,int(np.ceil((values[j,1]+.103)/dt)))
                good.append(dict(cost=float(answerable(values[j])), quality=values[j].tolist(),
                                 feedback=feedback[j].tolist(), suffix=rate[j,:end].tolist(),
                                 face=b['face'],depth=b['depth'],family=b['family'], eta=float(time[ids[j]]),
                                 perturbation=v[j].tolist(),round=k))
            counts = {f:int(np.sum([bases[j]['face']==f for j in ids])) for f in (-1,1)}
            for f,c in counts.items():
                total_faces[f] += c
            reports.append(dict(trials=len(ids),round=k,positive=counts[1],negative=counts[-1],
                                legal=int(legal.sum()),contacts=int(feedback[:,1].sum()),
                                closest_m=float(feedback[:,0].min()),
                                best_answerable=None if not legal.any() else float(answerable(values[legal]).min())))
            if len(good) >= 5 and any(g['cost']<2.20 for g in good):
                break
            archive.extend(zip(choice,ids.copy(),v.copy(),strict=True))
            elite = sorted(archive,key=lambda g:g[0])[:min(20,len(archive))]
            if k+1 < rounds:
                top = rng.integers(len(elite),size=batch)
                ids = np.asarray([elite[x][1] for x in top])
                v = np.asarray([elite[x][2] for x in top])+rng.normal(0.,1.,(batch,len(SIZES)))*SIZES*max(.32,.69-.10*k)
                v = np.clip(v,-2.75*SIZES,2.75*SIZES)
                keep = min(6,len(elite))
                ids[:keep] = [e[1] for e in elite[:keep]]
                v[:keep] = [e[2] for e in elite[:keep]]
        chosen = []
        for g in sorted(good,key=lambda row:row['cost']):
            if any(g['face']==r['face'] and abs(g['quality'][1]-r['quality'][1])<.019
                   and np.linalg.norm(np.asarray(g['feedback'][2:5])-r['feedback'][2:5])<.20
                   for r in chosen):
                continue
            chosen.append(g)
            if len(chosen)==5:
                break
        return chosen, dict(windows=len(bases),rounds=reports, positive_candidates=total_faces[1],
                            negative_candidates=total_faces[-1], shortlist=len(chosen))


class TapeProbe:
    """Use the issued touch as one geometric prior, and explore signed clock/drive changes."""

    size = np.r_[[.12,.18,.21,.21,.34], [.7,.96,1.35,1.55,1.60], [.060,.11,.15]]

    def __init__(self,a, steering=False):
        self.a = a
        self.core = Rollout(a,1)
        self.steering = steering

    def steer(self, experience, anchors, rng):
        # A local signed inverse of actual native contact consequences.
        estimates = [(r,z) for r,z in experience if z[1]>.5]
        if len(estimates)<16:
            return np.empty((0,len(self.size)))
        actions = np.asarray([e[0] for e in estimates])/self.size
        sensed = np.asarray([e[1] for e in estimates])
        candidates = []
        for anchor in anchors[:7]:
            center=anchor/self.size
            d=np.linalg.norm(actions-center,axis=1)
            keep=np.argsort(d)[:min(len(d),45)]
            weight=np.exp(-.5*(d[keep]/max(.50,np.median(d[keep])))**2)
            weight/=weight.sum()
            x=actions[keep]; v=sensed[keep,2:5]; center_x=weight@x
            mx=x-center_x; known=weight@v
            jac=np.linalg.solve(mx.T@(weight[:,None]*mx)+.10*np.eye(len(center)),
                                mx.T@(weight[:,None]*(v-known))).T
            before=known+jac@(center-center_x)
            p=weight@sensed[keep,5:8]
            for pace in (2.9,3.5,4.0,4.6,5.15):
                delayed=max(.30,(.63-p[1])/max(1.,.88*pace))
                through=-p[1]/max(1.,.91*pace)
                launch=max((.78-p[2])/delayed+4.905*delayed,
                           (.99-p[2])/through+4.905*through)
                target=np.array([-.88*p[0]/delayed,pace,launch])
                delta=jac.T@np.linalg.solve(jac@jac.T+.17*np.eye(3),target-before)
                for gain in (.70,1.23):
                    v=center+gain*np.clip(delta,-1.7,1.7)+rng.normal(0.,.045,len(center))
                    candidates.append(np.clip(v,-3.2,3.2)*self.size)
        return np.asarray(candidates)

    def search(self,seed,batch,rounds,old,tick,leg):
        a = self.a
        axes = a._arm_ids[1][2]
        lo,hi = a._joint_low[axes],a._joint_high[axes]
        stride = a._joint_increments[axes]*self.core.micro
        ref = a.data.ctrl[axes].copy()
        source = old['actions'][tick:tick+95]
        previous = [ref.copy()]
        for u in source:
            ref = np.clip(ref+stride*np.clip(u,-a.rate_scale,a.rate_scale),lo,hi)
            previous.append(ref.copy())
        base = np.asarray(previous)
        arrival=old['placements']['delivered'][leg]
        touch=arrival['near_contact']
        if touch is None:
            line=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=a._received_bounce,horizon=.5)
            eta=float(line[np.argmin(abs(line[:,2]+1.56)),0]) if len(line) else .22
        else:
            eta=float(touch['time']-a.data.time)
        n=round((eta+1.07)/a.control_dt)
        timeline=(np.arange(n)+1)*a.control_dt
        rng=np.random.default_rng(seed)
        raw=qmc.Sobol(len(self.size),scramble=True,seed=int(rng.integers(1,2**31))).random_base2(
            int(np.ceil(np.log2(batch))))[:batch]
        value=(2*raw-1)*self.size
        value[:6]=0.
        value[1:6,10]=[-.065,-.035,-.015,.021,.048]
        reports=[]; archive=[]; win=[]; experience=[]
        for stage in range(rounds):
            shifted = np.clip((timeline[None,:]+value[:,10,None])/a.control_dt,0,len(base)-1)
            i = shifted.astype(int); j=np.minimum(i+1,len(base)-1)
            u=(shifted-i)[...,None]
            q=(1-u)*base[i]+u*base[j]
            # Build up the position change early and the angular drive near the actual touch.
            contact=np.clip((timeline[None,:]-eta+value[:,11,None]*.35)/(.14+abs(value[:,12,None])), -1,1)
            onset = np.clip(timeline/min(.105,max(.025,eta*.7)),0,1)
            onset = onset*onset*(3-2*onset)
            q[:,:,:5] += onset[None,:,None]*(
                value[:,None,:5] + .145*contact[:,:,None]*value[:,None,5:10])
            f=np.zeros((batch,9))
            rates,y=self.core.trajectories(q,AIM,max(.20,eta+.18),feedback=f)
            good=np.isfinite(y[:,0])
            dense=landing_error(f)
            quality=np.where(good,answerable(y)-16.,np.where(f[:,1]>0,dense+3.*f[:,8],18.+45.*f[:,0]))
            for k in np.flatnonzero(good):
                end=max(1,int(np.ceil((y[k,1]+.11)/a.control_dt)))
                win.append(dict(cost=float(answerable(y[k])),quality=y[k].tolist(),feedback=f[k].tolist(),
                                suffix=rates[k,:end].tolist(),face=None,depth=float(f[k,6]),
                                family='issued-touch-signed-inverse' if self.steering else 'issued-touch-residual',
                                eta=eta,perturbation=value[k].tolist(),round=stage))
            reports.append(dict(trials=batch,round=stage,legal=int(good.sum()),contacts=int(f[:,1].sum()),
                                closest_m=float(f[:,0].min()),old_tape_contact=bool(stage==0 and f[0,1]),
                                old_tape_feedback=f[0].tolist() if stage==0 else None))
            if any(rr['cost']<2.65 for rr in win):
                break
            archive.extend(zip(quality,value.copy(),strict=True))
            if self.steering:
                experience.extend(zip(value.copy(),f.copy(),strict=True))
            top=sorted(archive,key=lambda x:x[0])[:12]
            ix=rng.integers(len(top),size=batch)
            value=np.array([top[x][1] for x in ix])+rng.normal(size=(batch,len(self.size)))*self.size*(.64 if stage<2 else .40)
            value=np.clip(value,-3.2*self.size,3.2*self.size)
            if self.steering:
                proposals=self.steer(experience,[e[1] for e in top],rng)[:batch-len(top)]
                if len(proposals):
                    value[-len(proposals):]=proposals
            value[:len(top)]=np.asarray([e[1] for e in top])
        chosen=[]
        for g in sorted(win,key=lambda x:x['cost']):
            if any(abs(g['quality'][1]-r['quality'][1])<.015
                   and np.linalg.norm(np.asarray(g['feedback'][2:5])-r['feedback'][2:5])<.22 for r in chosen):
                continue
            chosen.append(g)
            if len(chosen)==4:
                break
        return chosen,dict(windows=1,rounds=reports,positive_candidates=0,negative_candidates=0,shortlist=len(chosen))


def mark_old(row, leg):
    offer = row['placements']['delivered'][leg]
    more = row['placements']['delivered'][leg+1:]
    return dict(legal=bool(offer['near_landed']),contact=offer.get('near_contact') is not None,
                answered=bool(offer['near_landed'] and more), offered=offer)


def replay_prefix(a, row, card, tick):
    a.launch(row['shot'],card=card)
    a.partner.reset(seed=card['seed'])
    for x in row['actions'][:tick]:
        a.exchange(x,FAR_COMMAND)
    before = [e for e in row['events'] if e['time'] <= float(a.data.time)+1e-8]
    if digest_json(before)!=digest_json(a.events):
        raise ValueError('Matched old physical prefix did not replay exactly')
    return digest_json(before)


def trial(a, row, card, plan, tick, leg, *, already=False):
    if not already:
        replay_prefix(a,row,card,tick)
    actions = list(row['actions'][:tick])
    for j in range(180):
        action = plan['suffix'][j] if j<len(plan['suffix']) else ZERO.tolist()
        _,_,done,capped,_ = a.exchange(action,FAR_COMMAND)
        actions.append(action)
        if done or capped or a.rival_returns>=leg+2:
            break
    y = a.point_placements()
    target = y['delivered'][leg] if len(y['delivered'])>leg else None
    expected = row['placements']['delivered'][leg]['far_contact']
    if target is not None and digest_json(expected)!=digest_json(target['far_contact']):
        raise ValueError('Replacement did not receive the identical far hit')
    check = dict(legal=bool(target and target['near_landed']),contact=bool(target and target['near_contact']),
                 answered=bool(target and target['near_landed'] and len(y['delivered'])>leg+1))
    req = y['requested']
    orig = row['placements']['requested']
    if [o['target'] for o in orig[:leg+1]] != [p['target'] for p in req[:leg+1]]:
        raise ValueError('Reply target changed through the identical prefix')
    return dict(check,actions=actions,events=a.events,placements=y,
                score=audit(a.events,opening_side=1),motor_audit=a.motor_audit(),
                replacement={k:v for k,v in plan.items() if k!='suffix'})


def run(src,output,limit,batch,rounds,starts,anchored=False,steering=False,include=None):
    data=_read(src)
    if data.get('group')!='practice':
        raise ValueError('Feasibility school accepts a development-practice receipt only')
    deck=[r for r in roster(data) if include is None or r['id'] in include][:limit]
    if steering and not anchored:
        raise ValueError('Signed consequence search needs an archived-contact clock')
    panel=measure(src)
    receipt=dict(schema='tt40-low-feasibility-1', source=str(src), source_sha256=_hash(src),
                 ready=data['ready'], denominator=panel['bands']['below_0.99'], cards=deck,
                 role='privileged coach only, all acceptance on untouched physical governor and opponent',
                 depths=DEPTHS, anchored=anchored, steering=steering, batch=batch,max_rounds=rounds, rows=[])
    with centered(data['ready']):
        a=arena(protocol='random')
        probe=TapeProbe(a,steering=steering) if anchored else Probe(a)
        try:
            for item in deck:
                started=process_time()
                card=data['cards'][item['index']]
                old=data['rows'][item['index']]
                j=item['leg']
                before=mark_old(old,j)
                prior=before['offered']['far_contact']['time']
                separations=[e['time'] for e in old['events'] if e['kind']=='separation'
                             and e.get('collider')=='rival_racket' and prior<=e['time']<prior+.12]
                if not separations:
                    raise ValueError('Targeted received shot lacks an audited far separation')
                early=int(np.ceil((separations[0]+1e-7)/a.control_dt))+1
                ticks=[early,early+7,early+14][:starts]
                if anchored:
                    bounce=before['offered']['near_bounce']['time']
                    lift=next((e['time'] for e in old['events'] if e['kind']=='separation'
                               and e.get('collider') in ('table','table_right') and bounce<e['time']<bounce+.060),
                              bounce+.023)
                    after=int(np.ceil((lift+1e-7)/a.control_dt))+1
                    ticks=[after, max(early,after-6), after+3][:starts]
                presented=[]; trial_rows=[]
                win=None; best=None
                for tick in ticks:
                    stamp=replay_prefix(a,old,card,tick)
                    if a.terminated or a.truncated:
                        raise ValueError('Existing point was stopped before the prefix finished')
                    seed=np.random.SeedSequence([400811,card['seed'],j,tick])
                    options,stats=(probe.search(seed,batch,rounds,old,tick,j) if anchored else probe.search(seed,batch,rounds))
                    presented.append(dict(tick=tick, prefix_events_sha256=stamp,
                                          prefix_actions_sha256=digest_json(old['actions'][:tick]),
                                          q=a.joints.tolist(),**stats))
                    for z,plan in enumerate(options[:3]):
                        current=trial(a,old,card,plan,tick,j,already=z==0)
                        trial_rows.append({k:v for k,v in current.items() if k in
                                           ('legal','contact','answered','replacement','score')})
                        if best is None or (current['legal'],current['contact'])>(best['legal'],best['contact']):
                            best=current
                            best['fork_tick']=tick
                        if current['answered']:
                            win=current
                            win['fork_tick']=tick
                            break
                    if win is not None:
                        break
                picked=win if win is not None else best
                if picked is not None:
                    picked.update(shot=old['shot'],opponent_seed=card['seed'])
                outcome=dict(item, card=card, old=before, old_score=old['score']['learner'],
                             observation_attempts=presented, live_attempts=trial_rows,
                             result=picked, cpu_s=process_time()-started)
                receipt['rows'].append(outcome)
                _write(output,receipt)
                print(card['id'],j,round(item['height'],3),[x['rounds'] for x in presented],
                      [(x['legal'],x['answered']) for x in trial_rows],flush=True)
        finally:
            a.close()
    receipt['complete']=True
    _write(output,receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--output',type=Path,default=OUT/'low-feasibility.json.gz')
    p.add_argument('--limit',type=int,default=5)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--starts',type=int,default=3)
    p.add_argument('--anchored',action='store_true')
    p.add_argument('--steering',action='store_true')
    p.add_argument('--include',type=int,nargs='+')
    args=p.parse_args()
    args.output.parent.mkdir(exist_ok=True,parents=True)
    run(args.source,args.output,args.limit,args.batch,args.rounds,args.starts,args.anchored,
        args.steering,args.include)
