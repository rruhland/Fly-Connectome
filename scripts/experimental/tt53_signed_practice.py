"""Learn a signed direct motor correction from a few new RGB-observed misses."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import DRAG
from .tt29_vision import VERTICAL, flight_terms
from .tt36_impact import candidates
from .tt36_impact_refine import Frames
from .tt40_real_contact import CAMERA
from .tt44_paired_credit import State, prefix
from .tt52_matched_movements import ROOT as MATCHED

ROOT=Path("artifacts/tt53_signed_practice")
PAIRS= MATCHED/"pairs.json.gz"
LINES= MATCHED/"lines.json.gz"
REPEATS=(("axis0+",(1,0)),("axis0-",(-1,0)),("axis1+",(0,1)),("axis1-",(0,-1)))
N=mpc.HORIZON
ALPHA=.58
FINAL=1.10
SCALE=np.array([.20,.34,.10,.58,.66,1.48,.62,.48,.60,.55,.77,.33,.36,.36,.44,.51,.60,.35,.61,.65,.63])


def eta(x):
    # Measured projected time along the ordinary motion, not a hindsight
    # contact time, table bounce phase or successful command suffix.
    return float(np.clip(60.*(x[1]+1.50)/max(1.6,-x[4]),4.,21.))


def state(x, q, ref, failure):
    use=np.asarray(failure)
    pre=use[:min(6,max(3,round(.45*eta(x)))),:5].mean(axis=0)
    return np.r_[x,q[:5],ref[:5]-q[:5],pre]/SCALE


def source_bank(cases,held):
    return [
        dict(id=r["id"],key=state(np.asarray(r["observed"]),np.asarray(r["q"]),np.asarray(r["ref"]),np.asarray(r["loss"])),
             eta=eta(r["observed"]),difference=np.cumsum(np.asarray(r["win"])-r["loss"],axis=0)[:,:5])
        for r in cases if r["id"]!=held and r["observed"] is not None
    ]


def phases(path,s0,s1):
    # Warp only across public optically projected approach duration.
    clock=(1+np.arange(N))*np.clip(s0/s1,.60,1.65)
    clock=np.clip(clock,0,N)
    p=np.r_[np.zeros((1,5)),path]
    return np.stack([np.interp(clock,np.arange(N+1),q) for q in p.T],axis=1)


def local_basis(bank, x, q, ref, failure):
    u=state(x,q,ref,failure)
    ds=np.asarray([np.linalg.norm(b["key"]-u) for b in bank])
    use=[]
    groups=set()
    for i in np.argsort(ds):
        if bank[i]["id"] not in groups:
            groups.add(bank[i]["id"])
            use.append(i)
        if len(use)==6:
            break
    if len(use)<2:
        raise ValueError("Need independent released movement contrasts")
    E=np.array([phases(bank[i]["difference"],bank[i]["eta"],eta(x)).ravel() for i in use])
    # Short anti-slew smoothing stabilizes noisy reference differences.
    # Most of each direction is a real preexisting bounded motion contrast.
    E=E.reshape(-1,N,5)
    P=np.pad(E,((0,0),(1,1),(0,0)),mode="edge")
    E=(.20*P[:,:-2]+.60*P[:,1:-1]+.20*P[:,2:]).reshape(len(use),-1)
    norm=np.linalg.norm(E,axis=1)
    weights=np.exp(-.5*(ds[use]-ds[use[0]])**2/1.7**2)
    U=E/np.maximum(norm[:,None],1.e-5)
    center=(weights[:,None]*U).sum(axis=0)/weights.sum()
    d0=.74*U[0]+.26*center
    d0/=np.linalg.norm(d0)
    perpendicular=U-np.outer(U@d0,d0)
    n=np.linalg.norm(perpendicular,axis=1)
    i1=int(np.argmax(weights[1:]*n[1:]))+1
    d1=perpendicular[i1]/max(n[i1],1.e-6)
    radius=.92*np.clip(.65*norm[0]+.35*np.average(norm,weights=weights),2.8,18.)
    basis=radius*np.vstack((d0,d1)).reshape(2,N,5)
    return basis,dict(donor_ids=[bank[i]["id"] for i in use],distances=ds[use].tolist(),
                      learned_norm=float(radius),second_donor=bank[use[i1]]["id"])


def issue(failure,basis,coef,stride,lo,hi,ref):
    # Convert from empirical changes in normalized achieved joint reference
    # into direct bounded commands, preserving all actuator and rate limits.
    failure=np.asarray(failure)
    desired=np.empty_like(failure)
    bias=np.einsum("k,ktj->tj",coef,basis)*stride[:5]
    prev=ref.copy()
    for i,u in enumerate(failure):
        prev=np.clip(prev+stride*u,lo,hi)
        desired[i]=prev
    desired[:N,:5]+=bias
    actual=[]
    prev=ref.copy()
    for i in range(len(desired)):
        if i>=N:
            # Continue the actual failed command family after the brief
            # intervention, without chasing a recorded target state.
            u=failure[i]
        else:
            u=np.clip((np.clip(desired[i],lo,hi)-prev)/stride,-.8,.8)
        prev=np.clip(prev+stride*u,lo,hi)
        actual.append(u)
    return np.asarray(actual)


def surrogate(p,v):
    """A signed cost from observed departure toward any ordinary playable arc."""
    p,v=np.asarray(p),np.asarray(v)
    t=np.arange(0,1.46,.01)
    travel,fall=flight_terms(t,DRAG*np.clip(np.linalg.norm(v),1.,8.))
    path=p[None,:]+travel[:,None]*v+fall[:,None]*VERTICAL
    table=np.flatnonzero((path[:,2]<.779)&(t>.018))
    land=path[table[0] if len(table) else -1]
    eligible=np.flatnonzero(path[:,1]>=0.)
    net_z=path[eligible[0],2] if len(eligible) else .77-.12*abs(path[-1,1])
    net_x=path[eligible[0],0] if len(eligible) else land[0]
    zscore=np.clip((net_z-.945)/.23,-2.5,.85)
    yscore=np.clip(1.-abs(land[1]-.67)/.80,-3.0,1.)
    side=np.clip(max(0.,abs(land[0])-.41)*1.9+max(0.,abs(net_x)-.60),0.,2.)
    back=max(0.,.85-v[1])
    value=.53*zscore+.76*yscore-.55*side-.20*back
    return float(value),dict(net_z=float(net_z),projected_landing=land.tolist())


def optical(frames,cal):
    strict=frames.estimates(cal)
    options=[m for m in strict if -.012<m["time"]<.53]
    if options:
        return max(options,key=lambda o:o["delta_v_mps"]/(1.+9.*o.get("join_m",0.)))
    # Visibility can hide the blade color on a real miss. This independently
    # supported incoming-to-outgoing stereo bend may still read direction.
    loose=[p for p in candidates(frames.stereo,np.asarray(cal["acceleration"]))
           if -.012<p["time"]<.53 and p["center"][1]<-.87 and p["center"][2]>.89
           and p["delta_v_mps"]>1.22 and p["join_m"]<.029 and p["outgoing"][1]>.15]
    if not loose:
        return None
    p=max(loose,key=lambda o:o["delta_v_mps"]/(1.+9.*o["join_m"]))
    return dict(p,estimator="stereo-bend-without-blade")


def replay(a,shot,card,refrow,case,ctx,initial,u,cal, *, observe):
    snap,cam,rig,tick=ctx
    snap.restore(a)
    frames=Frames(rig) if observe else None
    if frames is not None:
        # A few pre-fork images support an unusually early contact without
        # measuring native position or timing.
        for tag,p,b in initial:
            frames.stereo.append((tag,p,b))
    mark=len(a.events)
    tape=[]
    won=False
    for j,motor in enumerate(u):
        if frames is not None and j<34:
            frames.observe(cam.capture(),j*mpc.DT,a.joints)
        _,_,done,capped,info=a.exchange(motor,mpc.FAR_COMMAND)
        tape.append(motor.tolist())
        won=any(e["kind"]=="legal_return" and e.get("side")==1 for e in info["events"])
        if won or done or capped:
            break
    delivery=a.point_placements()["delivered"]
    leg=case["leg"]
    if leg>=len(delivery):
        raise ValueError("No legal matched incoming offer")
    want=refrow["placements"]["delivered"][leg]
    got=delivery[leg]
    if any(digest_json(got[k])!=digest_json(want[k]) for k in ("far_contact","near_bounce")):
        raise ValueError("The perturbation changed the common native incoming delivery")
    m=optical(frames,cal) if frames is not None else None
    if m is not None:
        value,path=surrogate(m["center"],m["outgoing"])
        m=dict(m,value=value,**path)
    return dict(scored=bool(won),sensed=m,
                controls=tape,events=a.events[mark:],matched_far_and_bounce=True,
                camera_samples=0 if frames is None else len(frames.stereo))


def select(measured):
    """Turn the two empirical antithetic RGB slopes into one new issued path."""
    scores={k:None if r["sensed"] is None else r["sensed"]["value"] for k,r in measured.items()}
    axes=[]
    for i in range(2):
        up,dn=scores[f"axis{i}+"],scores[f"axis{i}-"]
        if up is None or dn is None:
            continue
        axes.append((abs(up-dn),i,np.sign(up-dn)))
    if not axes:
        return np.array([1.,0.]),dict(signed=False,scores=scores)
    amount,ax,sign=max(axes)
    if amount<.055:
        return np.array([1.,0.]),dict(signed=False,scores=scores)
    direction=np.zeros(2)
    direction[ax]=sign
    return direction,dict(signed=True,selected_axis=ax,finite_difference=float(sign*amount/(2.*ALPHA)),scores=scores)


def stock_endpoint(pair,line,cache):
    # Pin the archived failed endpoint only. No successful endpoint or
    # contact time from this release is read by the learner.
    for c in pair["rows"]:
        if c["reward"]:
            continue
        source=Path(c["source"])
        if source not in cache:
            cache[source]=_read(source)
        row=cache[source]["rows"][c["row"]]
        u=np.asarray(row["actions"][pair["tick"]:pair["tick"]+N])
        u=np.pad(u,((0,N-len(u)),(0,0)))
        if np.array_equal(u,np.asarray(line["loss"])):
            return c,cache[source]
    raise ValueError("Failed native archive no longer matches the docket")


def test(case,line,other,row,card,a,cam,rig,stride,lo,hi,cal):
    tick=line["tick"]
    prefix_images=prefix(a,card,row,tick,cam,a._arm_ids[1][2],window=26)
    snap=State(a)
    eye=mpc.Vision(rig,a.control_dt,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
    warm=Frames(rig)
    for at,i,q,r in prefix_images[-26:]:
        mpc.observed(eye,at,i,q)
        if at>=tick-9 and i is not None:
            warm.observe(i,(at-tick)*mpc.DT,q)
    q=a.joints.copy()
    ref=a.data.ctrl[a._arm_ids[1][2]].copy()
    x=mpc.observed(eye,tick,cam.capture() if tick%2==0 else None,q)
    if x is None or np.linalg.norm(x-np.asarray(line["observed"]))>1.e-6:
        raise ValueError("Cannot recover the certified causal initiation frame")
    failed=np.asarray(row["actions"][tick:tick+100])
    failed=np.pad(failed,((0,100-len(failed)),(0,0)))
    basis,donor=local_basis(other,x,q,ref,failed)
    ctx=(snap,cam,rig,tick)
    def run(goal,opt):
        return replay(a,row["shot"],card,row,case,ctx,warm.stereo,goal,cal,observe=opt)
    baseline=run(failed,True)
    if baseline["scored"]:
        raise ValueError("Original native failure did not reproduce")
    measured={}
    for k,weight in REPEATS:
        command=issue(failed,basis,ALPHA*np.asarray(weight),stride,lo,hi,ref)
        measured[k]=run(command,True)
    common=any(m["scored"] for m in measured.values())
    direction,note=select(measured)
    # Equal-data controls share the exact four physical practice rolls,
    # then each gets one new try with the same basis and perturbation norm.
    rng=np.random.default_rng(530564+int(line["id"])*29+tick)
    random_direction=np.asarray(REPEATS[int(rng.integers(4))][1])
    chosen={"learned":direction,"blinded":random_direction,
            "reversed":-direction if note["signed"] else direction,
            "prior_only":np.array([1.,0.])}
    arms={}
    seen={}
    if not common:
        for k,d in chosen.items():
            tag=tuple(d)
            if tag not in seen:
                command=issue(failed,basis,FINAL*d,stride,lo,hi,ref)
                seen[tag]=run(command,False)
            arms[k]=dict(seen[tag],direction=d.tolist())
    return dict(id=line["id"],tick=tick,prefix=line["prefix"],context=x.tolist(),basis=basis.tolist(),
                donor=donor,baseline=baseline,practice=measured,decision=note,
                common_practice_rescue=common,arms=arms)


def run(out,length):
    pairs,lines=_read(PAIRS),_read(LINES)
    if not lines["complete"] or lines["source"]!=_hash(PAIRS) or len(pairs["pairs"])!=len(lines["cases"]):
        raise ValueError("Unmatched physical forks")
    data=dict(rows=[],complete=False,n=length,groups=len({x["id"] for x in lines["cases"]}),
              source_sha256=_hash(PAIRS),local_contrasts_sha256=_hash(LINES),
              code_sha256=_hash(Path(__file__)),camera_calibration=_hash(CAMERA),
              rule="Leave every other case of the same measured release out. One failed replay, four native antithetic proposals in a local movement subspace, one equal-budget final attempt per arm. Exact native reward only grades practice. Camera-only signed post-action response picks the learned direction. Current successful replay action, phase and true field never enters.")
    if out.exists():
        old=_read(out)
        for k,v in data.items():
            if k not in ("rows","complete") and old[k]!=v:
                raise ValueError("Changed experimental design")
        data=old
    out.parent.mkdir(exist_ok=True,parents=True)
    stack=ExitStack()
    cache={}
    ready=None
    cal=_read(CAMERA)["models"]["60"]
    try:
        for i,(pair,line) in enumerate(zip(pairs["pairs"][:length],lines["cases"][:length],strict=True)):
            if i<len(data["rows"]):
                continue
            if pair["id"]!=line["id"] or pair["prefix"]!=line["prefix"] or pair["tick"]!=line["tick"]:
                raise ValueError("Changed fixed prefix")
            use,doc=stock_endpoint(pair,line,cache)
            if _hash(Path(use["source"]))!=pairs["sources"][use["source"]]:
                raise ValueError("Physical failed source changed")
            row,card=doc["rows"][use["row"]],doc["cards"][use["row"]]
            if ready!=doc["ready"]:
                stack.close()
                ready=doc["ready"]
                stack.enter_context(mpc.centered(ready))
                a=mpc.arena()
                stack.callback(a.close)
                cam,rig,stride,lo,hi=mpc.resources(a)
                stack.callback(cam.close)
            new=test(use,line,source_bank(lines["cases"],line["id"]),row,card,a,cam,rig,stride,lo,hi,cal)
            data["rows"].append(new)
            data["complete"]=len(data["rows"])==length
            _write(out,data)
            print(i,new["id"],"shared",new["common_practice_rescue"],"signed",new["decision"]["signed"],
                  {k:v["scored"] for k,v in new["arms"].items()}, new["decision"]["scores"],flush=True)
    finally:
        stack.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",type=int,choices=range(1,18),default=17)
    p.add_argument("--out",type=Path,default=ROOT/"trial.json.gz")
    a=p.parse_args()
    run(a.out,a.n)
