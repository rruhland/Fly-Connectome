"""Calibrate an empirical signed-impact inverse from actual direct-joint rallies."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt34_results import paired
from .tt35_error import passage
from .tt36_impact_refine import Frames
from .tt37_joint_augment import NativeCorrection
from .tt37_joint_capture import observed
from .tt37_joint_deep import DeepField
from .tt37_opponent import arena

OUT = Path(__file__).resolve().parents[2] / "artifacts" / "tt37"
FIELD = OUT / "joint-deep-observations.npz"
SENSOR = OUT / "joint-native-sensor.npz"
IMPACT = OUT.parent / "tt36" / "impact-model.json"
ANCHORS = (4735, 11139, 12166, 5029, 15270, 6208)
NATIVE = (54, 58, 67)
REFINED = (0, 10, 30)
X_SCALE = np.array([.18,.22,.16,.85,1.4,1.55,1.2,1.4,1.4,1.4,1.35])
ACTION = np.array([.23,.33,.38,.36,.50,2.2,3.,2.8,3.,3.6])
RANGE = np.array([.16,.17,.18,.18,.22,1.25,1.35,1.5,1.5,1.75])
VEL = np.array([1.5,2.25,2.4])
CAP = 24
LATCH = -.88


def affine(rows):
    hit = (rows[:,21]>.5) & (np.linalg.norm(rows[:,22:25],axis=1)<13) & (rows[:,28]<.17)
    valid = rows[hit]
    if len(valid)<10:
        return None
    a = valid[:,10:20]/ACTION
    ball = valid[:,22:25]/VEL
    w = 1/(1+.32*np.sum(a*a,axis=1)+3*valid[:,28])
    w /= w.sum()
    ca = a-w@a
    sy = ball-w@ball
    coef = np.linalg.solve(ca.T@(w[:,None]*ca)+.065*np.eye(10),ca.T@(w[:,None]*sy)).T
    residual = sy-ca@coef.T
    # One robust reweighting limits distinct hit families from flattening
    # high-leverage slopes at the immediate control neighborhood.
    w /= 1+np.sum(residual*residual,axis=1)
    w /= w.sum()
    ca = a-w@a
    sy = ball-w@ball
    coef = np.linalg.solve(ca.T@(w[:,None]*ca)+.065*np.eye(10),ca.T@(w[:,None]*sy)).T
    return coef*(RANGE/ACTION)[None,:]


def fit():
    with np.load(SENSOR) as doc:
        sensor = NativeCorrection(doc["native"],doc["optical"],doc["use"])
    keys, matrices, goals, goal_keys = [],[],[],[]
    refined = []
    digests = {SENSOR.name:_hash(SENSOR)}
    for tag in REFINED:
        path = OUT/f"clinic-refined-experience-{tag}.npz"
        with np.load(path) as doc:
            rows=doc["rows"]
        for x in np.unique(rows[:,:10],axis=0):
            j=affine(rows[np.all(rows[:,:10]==x,axis=1)])
            if j is not None:
                refined.append(j)
        digests[path.name]=_hash(path)
    for tag in NATIVE:
        paths = [OUT/f"{prefix}-{tag}{suffix}" for prefix,suffix in (
            ("native",".npz"),("native-outcomes",".npz"),("clinic-labels",".json.gz"))]
        tapes = _read(paths[2])
        with np.load(paths[0]) as frames, np.load(paths[1]) as bundle:
            outcomes=bundle["rows"]
            gfull=np.concatenate((frames["group"][:,:4],frames["weak_group"][:,:4]))
            sfull=np.concatenate((frames["states"],frames["weak_states"]))
            for card,row in zip(tapes["cards"],tapes["rows"]):
                for leg in row["records"]:
                    group=np.array([card["id"],card["seed"],card["reply_seed"],leg["tick"]])
                    snap=sfull[np.all(gfull==group,axis=1)]
                    x=np.asarray(leg["context"],dtype=np.float32)
                    controls=outcomes[np.all(outcomes[:,:10]==x,axis=1)]
                    if not len(snap) or len(controls)!=leg["choices"]:
                        continue
                    at=snap[np.argmin(abs(snap[:,1]-LATCH))].copy()
                    at=sensor.predict(at[None])[0]
                    j=affine(controls)
                    if j is not None:
                        keys.append(at)
                        matrices.append(j)
                    for goal in controls[controls[:,-1]>=0,22:25]:
                        if np.linalg.norm(goal)<10:
                            goals.append(goal)
                            goal_keys.append(at)
        digests.update({p.name:_hash(p) for p in paths})
    if len(keys)<15 or len(goals)<10:
        raise ValueError("Insufficient independently keyed signed action responses")
    np.savez_compressed(OUT/"adaptive-model.npz", x=keys, j=matrices, goals=goals, goal_x=goal_keys,
                        global_j=np.median(refined,axis=0))
    _write(OUT/"adaptive-model.json",dict(native_contexts=len(keys),refined_contexts=len(refined),
                                      successful_native_outcomes=len(goals), sources=digests))
    print("fit",len(keys),len(refined),len(goals),flush=True)


class Forward:
    def __init__(self, source):
        with np.load(source) as net:
            self.x,self.j,self.goals,self.goal_x,self.prior=(net[k] for k in ("x","j","goals","goal_x","global_j"))

    def local(self, x, actual):
        d=np.linalg.norm((self.x-x)/X_SCALE,axis=1)
        best=int(np.argmin(d))
        neighbor=np.argsort(d)[:6]
        weight=np.exp(-.55*(d[neighbor]**2-d[best]**2))
        same=(self.j[neighbor]*self.j[best]).sum(axis=(1,2))>0
        weight*=same
        model=np.einsum("i,ijk->jk",weight/weight.sum(),self.j[neighbor])
        model=.70*model+.30*self.prior
        gd=np.linalg.norm((self.goal_x-x)/X_SCALE,axis=1)
        near=np.argsort(gd)[:16]
        difference=np.linalg.norm((self.goals[near]-actual)/VEL,axis=1)
        ideal=self.goals[near[np.argmin(.37*gd[near]+difference)]]
        return model,ideal,float(d[best])


def ridge_inverse(j, error, damping=.16):
    return j.T@np.linalg.solve(j@j.T+damping*np.eye(len(error)),error)


class Store:
    def __init__(self, forward, trials=(), body=()):
        self.forward,self.trials,self.body=forward,list(trials),list(body)

    def correct(self, x, rng, active, scramble=False):
        d=np.array([np.linalg.norm((x-r["context"])/X_SCALE) for r in self.trials])
        available=[r for r,w in zip(self.trials,d,strict=True) if w<1.1]
        if not available:
            return np.zeros(10), "cold"
        good=[r for r in available if r["legal"]]
        if good:
            r=min(good,key=lambda r:np.linalg.norm((x-r["context"])/X_SCALE))
            return np.asarray(r["delta"]), "rewarded"
        # Match the best observed consequence among nearby attempts, not a
        # blend of the actual commands that produced conflicting shots.
        touched=[]
        for r in available:
            v=r["velocity"]
            if v is None:
                continue
            score=np.linalg.norm((np.asarray(v)-[0,3.25,2.8])/VEL)
            touched.append((score+.48*np.linalg.norm((x-r["context"])/X_SCALE),r))
        if touched:
            _,trace=min(touched,key=lambda p:p[0])
            v=np.asarray(trace["velocity"])
            j,goal,distance=self.forward.local(x,v)
            if distance>3.7:
                return np.zeros(10), "unsupported"
            error=np.clip((goal-v)/VEL,-2.,2.)
            old=np.asarray(trace["delta"])
            change=ridge_inverse(j,error)
            # Learn an additional real response only from observed interventions
            # sharing the current sensory context. No unexecuted action adds evidence.
            if len(touched)>=3:
                y=np.array([r["velocity"] for _,r in touched])/VEL
                a=np.array([r["delta"] for _,r in touched])
                k=np.exp(-.65*np.array([(x-r["context"])@((x-r["context"])/(X_SCALE**2))
                                       for _,r in touched]))
                k/=k.sum()
                aa=a-k@a
                yy=y-k@y-aa@j.T
                correction=np.linalg.solve(aa.T@(k[:,None]*aa)+.17*np.eye(10),aa.T@(k[:,None]*yy)).T
                change=ridge_inverse(j+correction,error)
            step=np.clip(change,-.85,.85)*.81
            if scramble:
                # Keep normalized step size but use mismatched joint axes.
                step=np.r_[np.roll(step[:5],2),np.roll(step[5:],2)]
            if active:
                step=step*(1. if rng.random()<.78 else -.55)+rng.normal(0.,.16,10)
            return np.clip(old+step,-1.,1.),"signed"
        missed=[r for r in available if r["miss_adjustment"] is not None]
        if missed:
            r=min(missed,key=lambda r:np.linalg.norm((x-r["context"])/X_SCALE))
            sign=1. if not active or rng.random()<.80 else -.6
            d=np.asarray(r["delta"]).copy()
            d[:5]+=sign*np.asarray(r["miss_adjustment"])/RANGE[:5]
            return np.clip(d,-1.,1.),"miss"
        return np.zeros(10),"unseen"

    def record(self, trials, body):
        self.trials.extend(r for r in trials if r["legal"] or r["velocity"] is not None or r["miss_adjustment"] is not None)
        self.trials=self.trials[-160:]
        self.body=(self.body+body)[-1500:]


def plan():
    ref=_read(OUT/"joint-deep-fresh_replies-local.json.gz")
    cards={c["id"]:c for c in ref["cards"]}
    base=_read(OUT/"opponent-plan.json")
    doc=dict(new=base["held_openings"],other=base["held_replies"],feed_sha256=base["feed_sha256"])
    for k,blocks,start in (("practice",3,382101),("replies",2,382317)):
        doc[k]=[dict(cards[sid],reply_seed=start+13*i,key=f"adaptive-{k}-{i}")
                for i,sid in enumerate(ANCHORS*blocks)]
    _write(OUT/"adaptive-plan.json",doc)


def body_direction(q, gap, rows):
    if len(rows)<8:
        return None
    w=np.asarray(rows)
    x=w[:,:5]-q
    d=np.linalg.norm(x,axis=1)
    use=np.argsort(d)[:min(44,len(d))]
    if d[use[0]]>.21:
        return None
    a=x[use];p=w[use,5:]
    k=np.exp(-.5*(d[use]/max(.14,d[use[len(use)//3]]))**2)
    k/=k.sum()
    a=a-k@a
    p=p-k@p
    j=np.linalg.solve(a.T@(k[:,None]*a)+.004*np.eye(5),a.T@(k[:,None]*p)).T
    return np.clip(.62*ridge_inverse(j[[0,2]],gap,.006),-.14,.14).tolist()


def bouts(detections, current, view, qs, dt, body):
    optical=[(round(t/dt),p,b) for t,p,b in view.stereo if b is not None]
    frames=[np.r_[qs[i],b].tolist() for i,_,b in optical if i in qs]
    following_strokes=current[1:]+[{"tick":10**7}] if current else []
    for item,following in zip(current,following_strokes,strict=True):
        i,j=item["tick"],following["tick"]
        candidates=[d for d in detections if i*dt<=d["time"]<min(j,i+58)*dt]
        hit=min(candidates,key=lambda d:d["time"]) if candidates else None
        cross=passage([(t,p,b) for t,p,b in optical if i<=t<j],i) if hit is None and not item["legal"] else None
        correction=None
        if cross is not None:
            gap=np.asarray(cross["gap_xz_m"])
            t=round(cross["tick"])
            if t in qs and .02<np.linalg.norm(gap)<.30:
                correction=body_direction(qs[t],gap,body+frames)
        item.update(velocity=None if hit is None else hit["outgoing"],
                    measured_gap=None if cross is None else cross["gap_xz_m"],miss_adjustment=correction)
    return frames


def point(sim,camera,rig,stride,lo,hi,ready,field,store,observer,card,shot,mode,rng):
    sim.launch(shot,card=card);sim.partner.reset(seed=card["seed"])
    dt=sim.control_dt
    vision=joint.Vision(rig,dt,raw=True)
    eye=Frames(rig)
    reference=np.array(ready)
    commands=[];current=[];qs={};strike=None
    sensed=False;old_y=None
    start=process_time()
    for tick in range(sim.max_steps):
        q=sim.joints.copy()
        frames=camera.capture() if tick%2==0 or strike is not None else None
        raw=observed(vision,tick,frames if tick%2==0 else None,q)
        if frames is not None and (strike is not None or (raw is not None and -.4>raw[1]>-1.7)):
            eye.observe(frames,tick*dt,q)
            qs[tick]=q[:5]
        if raw is not None and raw[4]<-.9:
            d=raw[1]
            if old_y is not None and old_y< -1.1 and d>-.40:
                strike=None
            old_y=d
            if strike is None and -1.49<d<LATCH:
                x=np.r_[raw,q[:5]]
                active=mode in ("learn","scrambled")
                a,how=store.correct(x,rng,active,scramble=mode=="scrambled") if mode!="frozen" else (np.zeros(10),"frozen")
                if mode=="reverse":a=-a
                strike=dict(tick=tick,context=x.tolist(),delta=a.tolist(),channel=how,legal=False)
                current.append(strike)
            if d<-.90:
                sensed=True
        elif raw is not None and raw[4]>.9:
            strike=None
        desired=np.array(ready)
        if raw is not None and (raw[4]<-.9 or (sensed and raw[1]<.50)):
            goal,confidence=field.get(raw,q,reference=reference)
            if goal is not None:
                desired+=confidence*(goal-desired)
                if strike is not None and raw[4]<-.9 and np.any(strike["delta"]):
                    a=np.array(strike["delta"])*RANGE
                    depthphase=np.clip(.23*(-raw[1]-1.56),-.14,.038)
                    envelope=np.clip((-.86-raw[1])/.36,0,1)*np.clip((raw[1]+1.78)/.19,0,1)
                    desired[:5]+=envelope*(a[:5]+depthphase*a[5:])
        command=np.clip((np.clip(desired,lo,hi)-reference)/stride,-.8,.8)
        reference=np.clip(reference+stride*command,lo,hi)
        _,_,done,capped,info=sim.exchange(command,joint.FAR_COMMAND)
        commands.append(command.tolist())
        if current and any(e["kind"]=="legal_return" and e["side"]==1 for e in info["events"]):
            current[-1]["legal"]=True
        if done or capped or sim.legal_returns+sim.rival_returns-1>=CAP:
            break
    detections=eye.estimates(observer)
    body=bouts(detections,current,eye,qs,dt,store.body)
    if mode in ("learn","scrambled"):
        store.record(current,body)
    return dict(shot=shot,opponent_seed=card["seed"],actions=commands,events=sim.events.copy(),
                score=audit(sim.events,opening_side=1),placements=sim.point_placements(),
                runtime_cpu_s=process_time()-start,strokes=current,body_samples=len(body))


def run(args):
    routes=_read(OUT/"adaptive-plan.json")
    if routes["feed_sha256"]!=_hash(args.feeds):
        raise ValueError("Measured releases changed")
    cards=routes[args.phase][:args.limit]
    snap=OUT/"adaptive-memory.json"
    use=_read(snap) if args.mode in ("carry","reverse") else None
    if use and set(use["practice_seeds"])&{c["reply_seed"] for c in cards}:
        raise ValueError("Reused acquisition reply sequence")
    store=Store(Forward(OUT/"adaptive-model.npz"),use["trials"] if use else (),use["body"] if use else ())
    with np.load(FIELD) as bank: field=DeepField(bank,"local")
    ready=_read(FIELD.with_suffix(".json"))["ready"]
    observer=_read(IMPACT)["models"]["60"]
    shots=measured_launches(args.feeds)
    path=OUT/f"adaptive-{args.phase}-{args.mode}.json.gz"
    result=dict(cards=cards,rows=[],ready=ready,opening_side=1,mode=args.mode,cap=CAP,protocol="random",
                source_sha256=_hash(FIELD),model_sha256=_hash(OUT/"adaptive-model.npz"),
                memory_sha256=_hash(snap) if use else None)
    if args.resume:
        if args.mode in ("learn", "scrambled"):
            raise ValueError("Resume applies only to an unchanged frozen policy")
        old=_read(path)
        if any(old[k]!=v for k,v in result.items() if k!="rows") or len(old["rows"])>len(cards):
            raise ValueError("Resuming would mix different physical or motor settings")
        result=old
    offset=len(result["rows"])
    with centered(ready):
        sim=arena(protocol="random")
        camera,rig,stride,lo,hi=joint.resources(sim)
        try:
            for i,c in enumerate(cards[offset:],offset):
                r=point(sim,camera,rig,stride,lo,hi,ready,field,store,observer,c,shots[c["id"]],
                        args.mode,np.random.default_rng(382734+i))
                result["rows"].append(r)
                _write(path,result)
                if args.mode=="learn":
                    _write(snap,dict(trials=store.trials,body=store.body,practice_seeds=[x["reply_seed"] for x in cards[:i+1]],
                                    source_sha256=_hash(path)))
                print(args.phase,args.mode,i+1,c["id"],r["score"]["learner"],
                      [(s["channel"],s["legal"],s["velocity"] and round(s["velocity"][1],2)) for s in r["strokes"]],flush=True)
        finally:
            camera.close();sim.close()


def validate():
    """Check unseen nearby action differences at the same native context."""
    counts=[]
    errors=[]
    for tag in NATIVE:
        with np.load(OUT/f"native-outcomes-{tag}.npz") as z:
            data=z["rows"]
        seen=0
        for x in np.unique(data[:,:10],axis=0):
            rows=data[np.all(data[:,:10]==x,axis=1)]
            weights=(rows[:,21]>.5)&(rows[:,28]<.17)&(np.linalg.norm(rows[:,22:25],axis=1)<13)
            # A training/test split shares the ball context but not the action.
            counts_in=np.unique(rows[weights,10:20],axis=0,return_index=True)[1]
            rows=rows[weights][counts_in]
            if len(rows)<17:
                continue
            seed=np.random.default_rng([382838,seen])
            subset=seed.permutation(len(rows))
            n=max(10,round(.72*len(rows)))
            a,b=rows[subset[:n]],rows[subset[n:]]
            j=affine(a)
            if j is None:
                continue
            d=np.linalg.norm((b[:,None,10:20]-a[None,:,10:20])/ACTION,axis=2)
            near=d.argmin(axis=1)
            ok=d.min(axis=1)<1.65
            delta=(b[ok,10:20]-a[near[ok],10:20])/RANGE
            y=(b[ok,22:25]-a[near[ok],22:25])/VEL
            yy=delta@j.T
            for real,pred in zip(y,yy,strict=True):
                errors.append([float(np.linalg.norm(real)),float(np.linalg.norm(real-pred)),
                               float(real@pred>0)])
            seen+=1
        counts.append(seen)
    vals=np.array(errors)
    proof=dict(validation="Same native context, unique action split, signed hits only; nearest observed action versus that anchored action plus Jacobian",
               contexts=sum(counts), pairs=len(errors),
               mean_vector_error_normalized=dict(held_action=float(vals[:,1].mean()), no_change=float(vals[:,0].mean())),
               positive_direction=int(vals[:,2].sum()), model_sha256=_hash(OUT/"adaptive-model.npz"))
    _write(OUT/"adaptive-model-validation.json",proof)
    print(proof,flush=True)


def metrics(rows):
    own=[r["score"]["learner"] for r in rows]
    shots=[s for r in rows for s in r["strokes"]]
    return dict(points=len(own), legal=sum(own), openings=sum(i>0 for i in own),
                later=sum(max(i-1,0) for i in own), longest=max(own,default=0),
                alternating=sum(r["score"]["returns"] for r in rows),
                signed=sum(s["velocity"] is not None for s in shots),
                misses_with_body_fit=sum(s["miss_adjustment"] is not None for s in shots),
                edits=sum(any(s["delta"]) for s in shots), cpu_s=sum(r["runtime_cpu_s"] for r in rows),
                per_point=own)


def summary():
    data={}
    paths=OUT/"adaptive-memory.json"
    for phase,modes in (("practice",("frozen","learn")),("replies",("frozen","carry","reverse")),
                        ("new",("frozen","carry"))):
        docs={m:_read(OUT/f"adaptive-{phase}-{m}.json.gz") for m in modes}
        base=docs["frozen"]
        idx=[i for i,r in enumerate(base["rows"]) if r["placements"]["delivered"]]
        base_short=dict(base, cards=[base["cards"][i] for i in idx], rows=[base["rows"][i] for i in idx])
        data[phase]={}
        for m,d in docs.items():
            if d["cards"]!=base["cards"] or len(d["rows"])!=len(d["cards"]):
                raise ValueError("The course is incomplete or mismatched")
            if any(sum(s["legal"] for s in r["strokes"])!=r["score"]["learner"] for r in d["rows"]):
                raise ValueError("A referee reward was not uniquely attached to an optical stroke")
            if m in ("carry","reverse") and d["memory_sha256"]!=_hash(paths):
                raise ValueError("Held state differs from the acquired sample history")
            if idx!=[i for i,r in enumerate(d["rows"]) if r["placements"]["delivered"]]:
                raise ValueError("Learner changes whether first physical ball reaches the table")
            s=dict(d, cards=[d["cards"][i] for i in idx], rows=[d["rows"][i] for i in idx])
            f=[]
            for a,b in zip(base["rows"],d["rows"],strict=True):
                if not a["strokes"] or not b["strokes"]: continue
                v,w=a["strokes"][0]["velocity"],b["strokes"][0]["velocity"]
                if v is None or w is None: continue
                f.append([v,w])
            data[phase][m]=dict(stats=metrics(d["rows"]), comparison=paired(base_short,s),
                                actual_first_balls=len(idx), both_signed_openings=len(f),
                                mean_signed_opening_velocity=None if not f else np.mean(f,axis=0).tolist())
    path=OUT/"adaptive-practice-scrambled.json.gz"
    if path.is_file():
        probe=_read(path)
        aligned=_read(OUT/"adaptive-practice-learn.json.gz")
        first=len(probe["cards"])
        if probe["cards"]!=aligned["cards"][:first] or len(probe["rows"])!=first:
            raise ValueError("The matched actuator-axis challenge is incomplete")
        chosen=dict(aligned,cards=aligned["cards"][:first],rows=aligned["rows"][:first])
        data["actuator_axis_challenge"]=dict(aligned=metrics(chosen["rows"]),scrambled=metrics(probe["rows"]),
                                             comparison=paired(probe,chosen))
    _write(OUT/"adaptive-results.json",data)


def verify():
    report={}
    with TemporaryDirectory(prefix="tt37-adaptive-cad-") as tmp:
        tape,receipt=(Path(tmp)/p for p in ("commands.json.gz","proof.json"))
        for phase in ("replies","new"):
            path=OUT/f"adaptive-{phase}-carry.json.gz"
            a,b=(_read(OUT/f"adaptive-{phase}-{m}.json.gz") for m in ("frozen","carry"))
            changed=[i for i,(base,new) in enumerate(zip(a["rows"],b["rows"],strict=True))
                     if base["score"]["learner"]!=new["score"]["learner"]]
            if not changed:
                changed=[next((i for i,r in enumerate(b["rows"]) if any(any(s["delta"]) for s in r["strokes"])),0)]
            small=dict(b,cards=[b["cards"][i] for i in changed],rows=[b["rows"][i] for i in changed])
            _write(tape,small)
            with patch.object(joint,"replay_arena",lambda cards,**kw:arena(cards,protocol="random",**kw)):
                joint.verify(tape,receipt,cad=True)
            report[phase]=dict(source_sha256=_hash(path),replayed=_read(receipt)["points"])
    _write(OUT/"adaptive-audit.json",report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("fit","plan","practice","replies","new","other","validate","summary","audit"))
    p.add_argument("mode",choices=("frozen","learn","carry","reverse","scrambled"),nargs="?")
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--limit",type=int)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    step={"fit":fit,"plan":plan,"validate":validate,"summary":summary,"audit":verify}.get(args.phase)
    step() if step else run(args)
