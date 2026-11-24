"""Free landing window and inverse rebound for the low-ball SO-101 diagnostic."""
import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from . import tt38_joint_mpc as mpc, tt40_low_lift as archive
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BladeIK
from .tt29_vision import flight_terms
from .tt39_terminal_early import EarlyServo, ExtendedBody, HORIZON
from .tt39_terminal_reachable import shoot
from .tt39_terminal_servo import desired_pose, endpoint_vector

GRAVITY = 9.81
GROUND = .782
NET = .9325
DECAY = .23   # First-pass linear drag only. Final test always uses the native collision and flight.
RESTITUTION = .78
SCALE = np.r_[.063,.079,.055,.40,.40,.40]
ROSTER = (4239, 13718, 14951, 11846)
COURSE = Path("artifacts/tt39/endurance-plan.json")


def launchers(p):
    """Analytically invert the flight for a window of landing coordinates and durations."""
    x, y, period = np.meshgrid([-.29, -.08,.08,.29], [.26,.43,.63,.85,1.05],
                               np.arange(.46,1.035,.034), indexing="ij")
    x,y,period = (a.ravel() for a in (x,y,period))
    travel, drop = flight_terms(period, DECAY)
    u = np.c_[(x-p[0])/travel, (y-p[1])/travel,
               (GROUND-p[2]+GRAVITY*drop)/travel]
    step = -p[1] / u[:,1]
    tn = -np.log1p(-DECAY*step)/DECAY
    a, b = flight_terms(tn, DECAY)
    z = p[2] + a*u[:,2] - GRAVITY*b
    # To remain a cooperative skill, avoid seeding dead drops. This is a
    # rough analytic filter only, not an inferred opponent response.
    descent = (u[:,2]*np.exp(-DECAY*period)-GRAVITY*travel)
    ok = (z>=NET+.018) & (z<1.35) & (descent < -2.15) & (u[:,1] > 2.35)
    return u[ok], np.c_[x,y,period,z][ok]


def rebalance(p, vin, n, *, budget=1.78):
    """Solve the signed paddle-normal motion for a fixed *achievable* orientation."""
    v = np.linspace(-.18, max(.05,budget), 30)
    out = vin + (1+RESTITUTION)*(v-vin@n)[:,None]*n
    vz,vy = out[:,2],out[:,1]
    root = (vz+np.sqrt(np.maximum(0.,vz*vz+2*GRAVITY*(p[2]-GROUND))))/GRAVITY
    # Vectorized Newton correction for the modest linear drag used by the screen.
    for _ in range(3):
        t,b = flight_terms(root,DECAY)
        dz = vz * np.exp(-DECAY*root)-GRAVITY*t
        root = np.clip(root-(p[2]+vz*t-GRAVITY*b-GROUND)/np.minimum(-.7,dz),.14,1.12)
    t,_ = flight_terms(root,DECAY)
    land = p[:2]+out[:,:2]*t[:,None]
    dt = -p[1]/np.maximum(.1,vy)
    net_t = -np.log(np.maximum(.1,1-DECAY*dt))/DECAY
    a,b = flight_terms(net_t,DECAY)
    z = p[2]+vz*a-GRAVITY*b
    ok = ((vy>2.35) & (z>=NET+.015) & (z<=1.39) &
          (land[:,1]>.17) & (land[:,1]<1.20) & (abs(land[:,0])<.57))
    if not ok.any():
        return None
    merit = (0.72 * abs(v) + .16*abs(land[:,1]-.54) + .10*abs(land[:,0]) +
             2.4*np.maximum(0.,NET+.042-z))
    k = np.flatnonzero(ok)[np.argmin(merit[ok])]
    return float(v[k]), np.r_[land[k],root[k],z[k]], out[k], float(merit[k])


def targets_and_rates(arr, reference, stride, low, high, *, inputs_are_rates):
    """Map a bounded control sequence to the *reachable* reference interface."""
    x=np.asarray(arr,float)
    prior=np.broadcast_to(reference,(len(x),len(reference))).copy()
    output=np.empty_like(x)
    issued=np.empty_like(x)
    for k in range(x.shape[1]):
        rate=x[:,k] if inputs_are_rates else (np.clip(x[:,k],low,high)-prior)/stride
        sent=np.clip(prior+stride*np.clip(rate,-.8,.8),low,high)
        issued[:,k]=(sent-prior)/stride
        output[:,k]=sent
        prior=sent
    return output,issued


def rate_shoot(body, anchor, goal, reference):
    """Shoot the same desired strike, differentiating bounded commands before the slew limiter."""
    engine=body.engine
    axes=engine.actuators
    stride=engine.twin._joint_increments[axes]*engine.predict.micro
    low,high=engine.twin._joint_low[axes],engine.twin._joint_high[axes]
    n=len(anchor)
    t=(np.arange(n)+.5)/n
    basis=np.c_[1-t,t]
    trajectories=0
    base_goals,base=targets_and_rates(anchor[None],reference,stride,low,high,inputs_are_rates=False)
    base=base[0]
    def review(u):
        nonlocal trajectories
        adjust=np.einsum("tb,kbd->ktd",basis,u.reshape(-1,2,5))
        target,rates=targets_and_rates(base+np.pad(adjust,((0,0),(0,0),(0,1))),
                                       reference,stride,low,high,inputs_are_rates=True)
        ends=body.endpoints(target)
        trajectories+=len(u)
        return ends,rates,target
    start=body.endpoints(base_goals)[0]
    if not np.isfinite(start).all():
        return None
    score=float(np.linalg.norm(endpoint_vector(start[None],goal)))
    initial_score=score
    center=np.zeros(10)
    result=None
    for _ in range(3):
        pert=np.eye(10)*.095
        end,_,_=review(np.r_[center[None],center+pert,center-pert])
        change=endpoint_vector(end,goal)
        v=np.where(np.isfinite(change[0]),change[0],0.)
        jac=((change[1:11]-change[11:])/ .19).T
        jac=np.nan_to_num(jac)
        move=-jac.T@np.linalg.solve(jac@jac.T+.020**2*np.eye(len(v)),v)
        probe=np.clip(center+np.array([.62,1.16,1.69])[:,None]*np.clip(move,-.50,.50),-.78,.78)
        out,ur,tr=review(probe)
        err=np.linalg.norm(endpoint_vector(out,goal),axis=1)
        err[~np.isfinite(err)]=np.inf
        choice=int(np.argmin(err))
        if err[choice]>=score-.00045:
            break
        score=float(err[choice]); center=probe[choice]
        result=(out[choice],ur[choice],tr[choice])
    if result is None or score>.08:
        return None
    end,applied,target=result
    difference=np.max(abs(applied-base),axis=1)
    changed=np.flatnonzero(difference>.025)
    if not len(changed) or changed[0]>=.65*n:
        return None
    meta=dict(before=start.tolist(),after=end.tolist(),goal=goal.tolist(),trajectories=trajectories+1,
              weighted_before=initial_score,weighted_after=score,
              first_private_rate_change=int(changed[0]),max_normalized_rate_change=float(difference.max()))
    return target,meta


class FreeArc:
    def __init__(self, engine):
        self.engine = engine
        self.geometry = BladeIK(engine.twin)
        self.body = ExtendedBody(engine)
        sampler = qmc.Sobol(5,scramble=True,seed=403091).random_base2(12)
        self.seeds = qmc.scale(sampler,self.geometry.low,self.geometry.high)
        self.tree = cKDTree(np.array([np.r_[*self.geometry.pose(q)] for q in self.seeds])/SCALE)

    def rows(self,q):
        t=self.engine.twin
        path=self.engine.nominal.flight.path(t.ball,t.velocity,t.spin,1,bounced=self.engine.bounced,horizon=.52)
        if not len(path):
            return [], []
        when,p = path[:,0],path[:,1:4]
        near = ((when>=.078) & (when<HORIZON*mpc.DT) & (p[:,1]>-1.75) & (p[:,1]<-1.18)
                & (p[:,2]>.82) & (p[:,2]<1.10) & (abs(p[:,0])<.36))
        if not near.any():
            return [], []
        eligible=np.flatnonzero(near)
        sample=np.unique(np.rint(np.linspace(eligible[0],eligible[-1],7)).astype(int))
        candidates=[]
        archive_aim=[]
        for j in sample:
            frames=round(when[j]/mpc.DT)
            if not (5<=frames<=HORIZON and near[j]):
                continue
            k=int(np.argmin(abs(when-frames*mpc.DT)))
            contact,vi=path[k,1:4],path[k,4:7]
            outgoing,land=launchers(contact)
            if not len(outgoing):
                continue
            norm=outgoing-vi
            norm/=np.linalg.norm(norm,axis=1)[:,None]
            u=np.einsum('ij,ij->i',outgoing+RESTITUTION*vi,norm)/(1+RESTITUTION)
            central = np.abs(land[:,0])
            existing = desired_pose(contact,vi,4.20)
            archive_aim.append([frames,existing[-1],existing[-2],len(outgoing),float(np.min(abs(u)))])
            # Different windows can expose different feasible face tilts. First preserve
            # normal-speed savings, then cover the trajectory geometry independently.
            order = np.argsort(abs(u) + .12*central + .10*np.abs(land[:,1]-.59))
            picked=[]
            for i in order:
                if not (-.22<u[i]<1.83) or any(np.linalg.norm(norm[i]-norm[j])<.080 for j in picked):
                    continue
                picked.append(i)
                if len(picked)>=7:
                    break
            for i in picked:
                desired=norm[i]
                center=contact-.025*desired
                _, idx=self.tree.query(np.r_[center,desired]/SCALE,k=5)
                alt = self.seeds[idx]
                for seed in [q[:5],*(s for s in alt if np.linalg.norm(s-q[:5],ord=np.inf)<4.35*frames*mpc.DT+.22)][:3]:
                    angles, kin=self.geometry.fit(center,desired,seed)
                    cf,n,J,_=kin
                    delta=contact-cf
                    plane=delta@n
                    tang=np.linalg.norm(delta-plane*n)
                    qtravel=np.linalg.norm(angles-q[:5],ord=np.inf)
                    if not (.003<plane<.053 and tang<.066 and qtravel<4.35*frames*mpc.DT+.14):
                        continue
                    drive_bound=min(1.83,float(3.9*np.linalg.norm(n@J,ord=1)))
                    selected=rebalance(contact,vi,n,budget=drive_bound)
                    if selected is None:
                        continue
                    speed,window,launch,merit=selected
                    rank = merit + 5.*tang + 5.*abs(plane-.025)+.26*qtravel
                    candidates.append((rank,frames,np.r_[cf,n,speed],angles,kin,
                                       dict(landing=window.tolist(),launch=launch.tolist(),plane=float(plane),
                                            radius=float(tang),travel=float(qtravel),optical_p=contact.tolist())))
        keep=[]
        for e in sorted(candidates,key=lambda x:x[0]):
            if any(abs(e[1]-a[1])<2 and np.linalg.norm(e[2][3:6]-a[2][3:6])<.12 for a in keep):
                continue
            keep.append(e)
            if len(keep)>=5:
                break
        return keep,archive_aim

    def stock_rows(self, stock):
        """Keep the moving stock disk's tangent coordinate, solve only the contact plane."""
        engine = self.engine
        t = engine.twin
        path = engine.nominal.flight.path(t.ball,t.velocity,t.spin,1,bounced=engine.bounced,horizon=.42)
        if not len(path):
            return []
        duration = min(HORIZON,len(stock),round(path[-1,0]/mpc.DT))
        picks = []
        for frames in range(5,duration+1):
            k = int(np.argmin(abs(path[:,0]-frames*mpc.DT)))
            p,vi = path[k,1:4],path[k,4:7]
            if not (-1.73<p[1]<-1.23 and .84<p[2]<1.08):
                continue
            running = stock[:frames].copy()
            result = self.body.endpoints(running[None])[0]
            if not np.isfinite(result).all():
                continue
            center,actual_n,actual_v = result[:3],result[3:6],result[6:9]
            d = p-center
            if np.linalg.norm(d)>.119:
                continue
            vectors,land = launchers(p)
            impulse = vectors-vi
            n = impulse/np.linalg.norm(impulse,axis=1)[:,None]
            vn = np.einsum('ij,ij->i',vectors+RESTITUTION*vi,n)/(1+RESTITUTION)
            signed = n@d
            tangent = np.linalg.norm(d-signed[:,None]*n,axis=1)
            face = np.linalg.norm(n-actual_n,axis=1)
            speed = np.abs(vn-n@actual_v)
            weight = abs(vn)*.24+9.*abs(signed-.025)+4.*tangent+.44*face+.34*speed
            feasible = ((abs(signed-.025)<.043) & (tangent<.083) & (face<.67)
                        & (speed<1.42) & (vn>-.18) & (vn<1.80))
            good=[]
            for j in np.flatnonzero(feasible)[np.argsort(weight[feasible])]:
                if any(np.linalg.norm(n[j]-n[k])<.085 for k in good):
                    continue
                normal_goal = center+(signed[j]-.025)*n[j]
                progress = dict(landing=land[j].tolist(), launch=vectors[j].tolist(),
                                plane=float(signed[j]), radius=float(tangent[j]),
                                optical_p=p.tolist(), anchored_stock=True)
                picks.append((float(weight[j]),frames,np.r_[normal_goal,n[j],vn[j]],running,progress))
                good.append(j)
                if len(good)>=3:
                    break
        out=[]
        for r in sorted(picks,key=lambda x:x[0]):
            if any(abs(r[1]-o[1])<2 and np.linalg.norm(r[2][3:6]-o[2][3:6])<.12 for o in out):
                continue
            out.append(r)
            if len(out)>=5:
                break
        return out

    def plan(self,q,reference,stock=None,*,diagnostic=False,rates=False,challenge=False):
        self.body.snapshot()
        rows,profile=self.rows(q)
        opts=[]
        for _, frames, goal, angle, ik, why in rows:
            start,dq=EarlyServo.schedule(self,q,reference,frames,goal,angle,ik)
            result=rate_shoot(self.body,start,goal,reference) if rates else shoot(self.body,start,goal)
            if result is None:
                continue
            seq, trace=result
            tail=np.arange(1,max(1,HORIZON+1-frames))*mpc.DT
            body=np.c_[seq[-1,:5]+np.minimum(tail,.075)[:,None]*dq*.51,
                       np.full(len(tail),reference[-1])]
            complete=mpc.tail(np.vstack((seq,body)),100)
            opts.append((complete,trace,why))
        anchored = self.stock_rows(stock) if stock is not None else []
        for _,frames,goal,seed,why in anchored:
            result=rate_shoot(self.body,seed,goal,reference) if rates else shoot(self.body,seed,goal)
            if result is None:
                continue
            seq,trace=result
            dref=seq[-1]-seq[-2]
            extension=np.asarray([seq[-1]+min(k,4)*dref for k in range(1,7)],float)
            complete=mpc.tail(np.vstack((seq,extension)),100)
            opts.append((complete,trace,why))
        note = dict(retained_endpoints=len(rows), fixed_pace_screen=profile, anchored_endpoints=len(anchored),
                    shot_endpoints=[dict(trace=trace, **why) for _,trace,why in opts])
        if not opts:
            return None,note
        t=self.engine.twin
        goals=np.array([r[0] for r in opts])
        y0=t.ball.copy(); v0=t.velocity.copy()
        spin=t.data.qvel[t._ball_dof+3:t._ball_dof+6]; w0=spin.copy()
        valid=[]; measurements=[]; contacts=[]
        try:
            for eps in [0.,-1.,1.]:
                t.ball[:]=y0+eps*np.array([.0018,.0014,.0034])
                t.velocity[:]=v0+eps*np.array([.021,.045,.062])
                spin[:]=w0+[eps*18,0.,0.]
                f=np.zeros((len(goals),9))
                _,y=self.engine.trajectories(goals,mpc.AIM,.54,feedback=f)
                valid.append(np.isfinite(y[:,0]))
                measurements.append(y)
                contacts.append(f)
        finally:
            t.ball[:],t.velocity[:],spin[:]=y0,v0,w0
        y=measurements[0]
        wins=np.sum(valid,axis=0)
        note.update(legal=valid[0].tolist(),stress_trials=wins.tolist(),
                    private_departures=np.asarray(contacts)[0,:,2:5].tolist())
        valid_ids = np.flatnonzero(valid[0] & (wins>=2))
        if len(valid_ids):
            j=int(valid_ids[np.argmin(archive.answerable(y[valid_ids])+1.2*(3-wins[valid_ids]))])
        elif diagnostic:
            # Actual practice, not deployment. Issue one optical near-launch from the
            # anchored family when high enough and forward, even if the private
            # wall test disagrees. A real score tells whether this gate rejects too much.
            f = contacts[0]
            good = np.flatnonzero((f[:,1]>0)&(f[:,3]>(.72 if challenge else 2.30))&(f[:,4]>2.18))
            if not len(good):
                return None,note
            j=int(good[np.argmin(archive.clinic.landing_error(f[good])-.5*wins[good])])
            note["practice_only_weak_approval"]=True
        else:
            return None,note
        if valid[0][j]:
            stop = max(1,round((y[j,1]+.098)/mpc.DT))
        else:
            stop = round((.25+.098)/mpc.DT)
        note.update(issued=True, index=j, virtual_landing=y[j,3:5].tolist() if valid[0][j] else None,
                    virtual_contact_s=float(y[j,1]) if valid[0][j] else None,
                    virtual_forward_vz=contacts[0][j,3:5].tolist())
        return goals[j,:stop],note


def matched(args):
    source=_read(archive.SCHOOL)
    lookup={c["id"]:(c,r,first) for c,r,first in archive.train_cases(source)}
    releases=ROSTER[:args.limit]
    ready=source["ready"]
    archive.joint.RECEIVE_Y=source["depth"]
    model=_read(archive.PUPIL)
    d=dict(source=str(archive.SCHOOL),source_sha256=_hash(archive.SCHOOL),roster=list(releases),
           code_sha256=_hash(Path(__file__)), mode="fixed diagnostic prefixes on practice; all continuation actions physical",
           protocol="random",practice_only_weak=args.issue_diagnostic,rate_space=args.rates,
           challenge_weak_rate=args.challenge,rows=[])
    surface = archive.setup(source["depth"]) if args.anchored else None
    with archive.centered(ready):
        a=archive.arena()
        engine=archive.OpticalClinic(body=True)
        skill=FreeArc(engine)
        stereo,rig,stride,low,high=archive.joint.resources(a)
        try:
            for release in releases:
                card,row,first=lookup[release]
                started=process_time()
                a.launch(row["shot"],card=card)
                a.partner.reset(seed=card["seed"])
                actor=archive.joint.JointPolicy(model|{"ready":ready},stride,low,high)
                eye=archive.joint.Vision(rig,a.control_dt)
                eye.eye=archive.LearnedOptics(eye.eye,source["visual"])
                track=archive.ArrivalHistory()
                engine.previous=None;engine.bounced=False
                t0=first["tick"]
                records = {r["tick"]: r for r in row["records"]}
                schedule=set(range(t0-7,t0+7,3))
                if args.anchored:
                    schedule.update(records)
                seen=[]; actions=[]; slip=None
                motion=None
                for tick in range(a.max_steps):
                    q=a.joints.copy()
                    obs=eye.update(tick,stereo.capture() if tick%2==0 else None,q)
                    obs=track.update(tick*a.control_dt,obs)
                    engine.observe(tick,eye.eye,q,actor.reference.copy())
                    if slip is None:
                        if (tick in schedule and obs is not None and .84<obs[2]<1.01
                                and .076<obs[0]<.38 and engine.sync()):
                            stock = None
                            if args.anchored and tick in records:
                                decision=records[tick]
                                stock=archive.clinic.schedules(*archive.clinic.geometry(surface,obs),
                                      obs[0],np.array([decision["action"]]),a.control_dt,ready,29)[0]
                            motion,status=skill.plan(q,actor.reference.copy(),stock,
                                    diagnostic=args.issue_diagnostic and tick==max(records),rates=args.rates,challenge=args.challenge)
                            seen.append(dict(tick=tick,vision=obs[:6].tolist(),**status))
                            if motion is not None:
                                if digest_json(a.events)!=digest_json([e for e in row["events"] if e["step"]<=tick]):
                                    raise RuntimeError("The physical prefix disagrees before the new action")
                                slip=tick
                    if slip is None:
                        u=np.asarray(row["actions"][tick])
                        actor.reference=np.clip(actor.reference+u*stride,low,high)
                    else:
                        k=tick-slip
                        goal=motion[k] if k<len(motion) else np.asarray(ready)
                        u=actor.drive(goal,q,.8)
                    actions.append(u.tolist())
                    _,_,done,cap,_=a.exchange(u,archive.clinic.FAR)
                    if done or cap or (slip is not None and archive.audit(a.events,opening_side=1)["opponent"]>1):
                        break
                if slip is None and digest_json(a.events)!=digest_json(row["events"]):
                    raise RuntimeError("The no-action court failed exact replay")
                score=archive.audit(a.events,opening_side=1)
                result=dict(card=card,shot=row["shot"],fork_tick=slip,score=score,events=a.events,actions=actions,
                            old=row["score"]["learner"], replied=score["opponent"]>1, cases=seen,
                            cpu_s=process_time()-started)
                d["rows"].append(result)
                _write(args.output,d)
                print(release,slip,score["learner"],score["opponent"],
                      [(s["retained_endpoints"],s.get("stress_trials")) for s in seen],flush=True)
        finally:
            stereo.close();engine.twin.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--anchored",action="store_true")
    p.add_argument("--issue-diagnostic",action="store_true")
    p.add_argument("--rates",action="store_true")
    p.add_argument("--challenge",action="store_true",help="Physically test one weak but changed rate sequence")
    p.add_argument("--limit",type=int,choices=range(1,len(ROSTER)+1),default=len(ROSTER))
    a=p.parse_args()
    if a.challenge and not (a.anchored and a.rates and a.issue_diagnostic):
        raise ValueError("The challenge is only for rate-space practice")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    matched(a)
