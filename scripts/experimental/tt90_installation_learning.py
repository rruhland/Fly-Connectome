"""Fixture-specific motor learning on untouched, broad continuous random points."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt84_installation_play as mount
from . import tt84_installation_system as system
from . import tt69_clamp_school as school
from . import tt69_clamp_policy as optics
from . import tt90_joint_library as library
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_curriculum import card, opening_candidates
from .tt37_opponent import draw
from .tt40_long_low import same_trace

ROOT=Path("artifacts/tt90_installation_learning")
PLAN=ROOT/"protocol.json"
FEEDS=Path("/tmp/tt25-rallies.json")
HARD=ROOT/"hard.protocol.json"
BODIES=ROOT/"body-library.npz"
MODEL=ROOT/"body-library.json"
BASE=mount.MP
SEASON=mount.SEASON
COURSE=mount.COURSE
CAP=240


def scopes():
    base=(system.PLAN,system.BUILD,system.OUT,mount.PLAN,mount.SEASON_PLAN,mount.TT80,
          SEASON.CARRY,BASE.BANK,COURSE.OPTIC,SEASON.fresh.RISK,SEASON.fresh.servo.ROOT/"prior.npz",
          COURSE.PRACTICE)
    return {str(p):_hash(p) for p in base}


def excluded():
    p=_read(system.PLAN); out={r["id"] for r in p["physical"]+p["archive"]}
    for p in (mount.PLAN,mount.SEASON_PLAN):
        out.update(r["id"] for r in _read(p)["cards"])
    p=_read(SEASON.CARRY); out.update(r["id"] for r in p["cards"][:len(p["rows"])])
    for p in SEASON.fresh.servo.SOURCES:
        out.update(r["card"]["id"] for r in _read(p.with_suffix(".json"))["practice"])
    out.update(r["id"] for r in _read(SEASON.fresh.RISK)["rows"])
    return out


def material():
    d=_read(system.BUILD); fixed=d["rows"][0]["candidate"]["mount"]["base"]
    c=d["rows"][d["chosen"]]; m=c["candidate"]
    return m,c["commission"]["ready"],mount.Registration(fixed,m["mount"]["base"],m["mount"]["rotation"])


def freeze(feeds):
    """No rollout, contact, actor result or summary enters this release shuffle."""
    if PLAN.exists():
        contract(feeds)
        return
    block=excluded()
    rng=np.random.default_rng(904537451)
    pool=[s["id"] for s in opening_candidates(BASE.measured_launches(feeds)) if s["id"] not in block]
    if len(pool)<112:raise ValueError("Too few disjoint natural launches")
    roster=rng.permutation(pool)[:112]
    def make(i,start):
        seed=int(rng.integers(1,2**30))
        return dict(card(int(i),"full",start,seed),profile="reach",opening="random",
                    reply_seed=int(rng.integers(1,2**30)))
    p=dict(inputs=scopes(),feeds=_hash(feeds),excluded=sorted(block),support_candidates=len(pool),
           learn=[make(i,j) for j,i in enumerate(roster[:40])],
           assess=[make(i,j) for j,i in enumerate(roster[40:60])],
           hard_pool=[make(i,j) for j,i in enumerate(roster[60:])],cap=CAP,
           policy="TT84 full unchanged canted installation, broad native random opponent and calibrated RGB/encoders. Pick 40 practice and 20 unfiltered qualification recordings from standard measured launch support, outcome independent, one launch per point. Reserve 52 other releases for stratifying matched hard-offer diagnostic by neutral physical arrivals only. No use of TT84 fixture-design archive, its coached held, preliminary or TT80 releases, or initial replay-bank deliveries in TT90 practice or held. No coach truth in the actor.")
    ROOT.mkdir(exist_ok=True,parents=True);_write(PLAN,p)
    print("registered",len(pool),[(k,len(p[k])) for k in ("learn","assess","hard_pool")],flush=True)


def contract(feeds):
    d=_read(PLAN)
    # TT84 completed its preassigned routine audit after this roster was
    # written. The append-only physical witness is not an actor/training
    # input. Keep its originally seen hash without gating on final bytes.
    geometry=str(system.OUT)
    expected={k:v for k,v in d["inputs"].items() if k!=geometry}
    if expected!={k:v for k,v in scopes().items() if k!=geometry} or d["feeds"]!=_hash(feeds):
        raise ValueError("Changed physical or motor inputs")
    audit=_read(system.OUT)
    if not audit["complete"] or audit["build"]!=_hash(system.BUILD):
        raise ValueError("TT84 physical body qualification is incomplete or changed")
    ids=[r["id"] for k in ("learn","assess","hard_pool") for r in d[k]]
    if len(set(ids))!=len(ids) or set(ids)&set(d["excluded"]):raise ValueError("TT90 release overlap")
    return d


class ObservedPlayer(mount.Registered):
    """Instrument actual reward and the causal state/action tape, without touching control."""

    def __init__(self,*args,memory=None,**kw):
        super().__init__(*args,**kw)
        keep=self.priors.keep
        def paid(tape,tag):
            actual=keep(tape,tag)
            rows=[r for r in self.priors.online if r[2]==tag] if actual else []
            self.reward_traces.append(dict(tag=list(tag),native_rows=len(rows),frames=self.frames(tape)))
            return actual
        self.priors.keep=paid
        retrieve=self.priors.priors
        self.body_memory=memory
        def mixture(x,q,ref,limit):
            stock=retrieve(x,q,ref,limit)
            if self.body_memory is None:return stock
            paths=self.body_memory.propose(x,q,ref)[:max(0,len(stock)-3)]
            if paths:
                self.proposals.append(dict(tick=self.decision_tick,paths=[meta for _,meta in paths]))
                self.current_proposed=paths
            return stock[:len(stock)-len(paths)]+[(goals,True) for goals,_ in paths]
        self.priors.priors=mixture

    def begin(self):
        super().begin()
        self.reward_traces=[];self.missed_trace=[];self.proposals=[]

    def search(self,tick,x,q,reference,base):
        self.current_proposed=[]
        result=super().search(tick,x,q,reference,base)
        if self.current_proposed and self.best is not None and self.tick==tick:
            exact=[meta for trial,meta in self.current_proposed if np.array_equal(trial,self.best)]
            self.proposals[-1]["directly_selected"]=exact
        return result

    def frames(self,tape):
        # No simulator/referee field. q and the already-issued motor reference
        # come directly from the same camera/encoder loop that chose the action.
        if not tape:return []
        return [[t,x.tolist(),q.tolist(),ref.tolist(),u.tolist()] for t,x,q,ref,u in tape[-68:]
                if x is not None and x[4]<-.90 and -1.78<x[1]<-.37]

    def missed(self,tape):
        self.missed_trace=self.frames(tape)


def player(raw,reg,ready,ids,rank,memory=None):
    with patch.object(BASE,"Fragments",lambda bank,engine:mount.Carried(bank,engine,practice=[COURSE.PRACTICE],held=ids)):
        model=ObservedPlayer(BASE.BANK,raw,register=reg,variant="elastic",budget=28,coverage=rank,memory=memory)
    model.priors.home=np.asarray(ready)
    return model


def prior(model):
    start=_read(SEASON.CARRY)
    COURSE.resume(model,SEASON.CARRY.with_suffix("").with_suffix(".npz"),start)
    return start


def path(mode):
    return ROOT/(mode+".json.gz")


def bank(mode):
    return ROOT/(mode+".npz")


def restore(model,mode,d,previous):
    COURSE.resume(model,bank(mode),dict(d,rows=[None]*previous+d["rows"]))


def checked_play(a,cam,rig,stride,lo,hi,ready,c,shot,raw,model,mode,cap=CAP):
    start=a.launches
    r=BASE.play(a,cam,rig,stride,lo,hi,ready,c,shot,raw,model,COURSE.OPTIC,mode,cap,progress=20)
    if (a.launches!=start+1 or any(w["target"]!=list(draw(c,i,"random",None)[0]) or w["draw"]!="random"
            for i,w in enumerate(r["placements"]["requested"]))):
        raise ValueError("A native point or broad random replay changed")
    if mode=="frozen" and r["memory"]["before"]!=r["memory"]["after"]:raise ValueError("Frozen assessment learned")
    return r


def hard_plan(feeds):
    d=contract(feeds);m,ready,_=material();shots=BASE.measured_launches(feeds)
    if HARD.exists():return
    scout=[]; chosen=[]; quota={"low":8,"ordinary":4}
    with school.enabled(),system.installed(m["v"],m["mount"]),BASE.centered(ready):
        a=BASE.arena(width=64,height=48)
        try:
            for c in d["hard_pool"]:
                save,prefix=school.launch(a,c,shots[c["id"]],[],0,ready)
                r=None if save is None else school.trial(a,save,c,prefix,0,[school.ZERO]*126)
                offer=None if r is None else r["offer"];line=None if offer is None else offer["receive_line"]
                h=None if line is None else float(optics.projected_height(line)); x=None if line is None else line["xyz"][0]
                kind=None
                if h is not None and .935<=h<1.005 and -.3<x<.35:kind="low"
                elif h is not None and 1.028<=h<=1.15 and abs(x)<.41:kind="ordinary"
                row=dict(card=c,height=h,x=x,kind=kind,offer=None if offer is None else
                         {k:offer[k] for k in ("far_contact","near_bounce","receive_line")})
                scout.append(row)
                if kind and quota[kind]:
                    chosen.append(row);quota[kind]-=1
                if not any(quota.values()):break
        finally:a.close()
    _write(HARD,dict(protocol=_hash(PLAN),scout=scout,cases=chosen,remaining_quota=quota,
                    rule="First physically delivered neutral scouts in each prespecified stratum. No pupil outcome selects a case. Each independently seeded measured release appears at most once."))
    print("hard qualifications",len(scout),[(r["card"]["id"],r["kind"],r["height"]) for r in chosen],flush=True)


def training(feeds,n):
    p=contract(feeds); m,ready,reg=material()
    if not 0 < n <= len(p["learn"]):raise ValueError("Invalid independent practice count")
    d=dict(protocol=_hash(PLAN),body=_hash(system.BUILD),mode="broad live experience",cards=p["learn"],rows=[])
    dest=path("practice"); used={k["id"] for k in p["learn"]+p["assess"]+p["hard_pool"]}
    if dest.exists():
        e=_read(dest)
        if any(e[k]!=x for k,x in d.items() if k!="rows"):raise ValueError("Changed native training run")
        d=e
    with np.load(BASE.BANK) as f:raw=mount.LocalField(BASE.DeepField(f,"local"),reg)
    rank=SEASON.fresh.shot.Coverage(_read(SEASON.fresh.RISK)); shots=BASE.measured_launches(feeds)
    with school.enabled(),system.installed(m["v"],m["mount"]),BASE.centered(ready):
        a=BASE.arena();cam,rig,stride,lo,hi=BASE.resources(a)
        model=player(raw,reg,ready,used,rank)
        try:
            start=prior(model)
            if d["rows"]:restore(model,"practice",d,len(start["rows"]))
            for index,c in enumerate(p["learn"][len(d["rows"]):n],start=len(d["rows"])):
                r=checked_play(a,cam,rig,stride,lo,hi,ready,c,shots[c["id"]],raw,model,"live")
                r.update(optical_earned=model.reward_traces,optical_missed=model.missed_trace)
                d["rows"].append(r); d["checkpoint_sha256"]=COURSE.save(model,bank("practice"));_write(dest,d)
                print("practice",index+1,c["id"],r["score"]["learner"],len(r["optical_earned"]),len(r["optical_missed"]),flush=True)
        finally:model.close();cam.close();a.close()


def fit(feeds):
    from . import tt90_attained_practice as coach

    p=contract(feeds);m,ready,_=material(); d=_read(path("practice"))
    if d["protocol"]!=_hash(PLAN) or len(d["rows"])!=len(p["learn"]) or d["checkpoint_sha256"]!=_hash(bank("practice")):
        raise ValueError("Finish all predeclared mounted practice points before locking the body memory")
    coached=_read(coach.OUT)
    if not coached["complete"] or coached["practice"]!=_hash(path("practice")):
        raise ValueError("Mounted practice-only exact repair evidence changed")
    earned=[(d["cards"][s["index"]],r["sensory"]) for s,r in zip(coached["cases"],coached["rows"],strict=True) if r["sensory"]]
    acquisition=dict(cards=d["cards"]+[c for c,_ in earned],rows=d["rows"]+[r for _,r in earned])
    with school.enabled(),system.installed(m["v"],m["mount"]),BASE.centered(ready):
        a=BASE.arena(width=64,height=48)
        try:f,e=library.data(a,ready,acquisition)
        finally:a.close()
    if set(f["releases"]) & {k["id"] for k in p["assess"]+p["hard_pool"]}:raise ValueError("Held release in local body memory")
    np.savez_compressed(BODIES,**f)
    _write(MODEL,dict(protocol=_hash(PLAN),practice=_hash(path("practice")),carry=d["checkpoint_sha256"],
                      coached=_hash(coach.OUT),coached_sources=len(earned),library=_hash(BODIES),**e))
    print("frozen same-body library",e,flush=True)


def validate_game(a,c,shot,r,guard):
    return school.verify(a,c,shot,dict(r,controls=r["actions"]),guard)


def assess(feeds,n,hard=False):
    from . import tt90_attained_practice as coach

    p=contract(feeds);m,ready,reg=material(); initial=_read(SEASON.CARRY)
    journal=_read(path("practice")); learned=_read(MODEL)
    if (learned["practice"]!=_hash(path("practice")) or learned["carry"]!=_hash(bank("practice"))
        or learned["protocol"]!=_hash(PLAN) or learned["library"]!=_hash(BODIES) or learned["coached"]!=_hash(coach.OUT)):
        raise ValueError("Changed independent body acquisition")
    if hard:
        measured=_read(HARD)
        if measured["protocol"]!=_hash(PLAN):raise ValueError("Hard workload changed")
        using=measured["cases"]
    else:using=[dict(card=c,kind="unfiltered") for c in p["assess"]]
    n=min(n,len(using)); used={k["id"] for k in p["learn"]+p["assess"]+p["hard_pool"]}
    if used&set(learned["earned_releases"])!=set(learned["earned_releases"]):raise ValueError("Changed motor release registry")
    arm=("original","ordinary","body","sham") if hard else ("original","ordinary","body")
    dest=path("hard" if hard else "assess")
    d=dict(protocol=_hash(PLAN),library=_hash(MODEL),cases=using,arms=arm,rows=[])
    if dest.exists():
        e=_read(dest)
        if any(e[k]!=(list(x) if isinstance(x,tuple) else x) for k,x in d.items() if k!="rows"):
            raise ValueError("Changed qualification protocol")
        d=e
    with np.load(BASE.BANK) as f:raw=mount.LocalField(BASE.DeepField(f,"local"),reg)
    with np.load(BODIES) as f: memories={k:library.Library(f,sham=k=="sham") for k in arm if k in ("body","sham")}
    rank=SEASON.fresh.shot.Coverage(_read(SEASON.fresh.RISK)); shots=BASE.measured_launches(feeds)
    with school.enabled(),system.installed(m["v"],m["mount"]),BASE.centered(ready):
        a=BASE.arena();cad=BASE.arena(width=64,height=48,detail="cad");cam,rig,stride,lo,hi=BASE.resources(a)
        guard=school.physical.GripClearance(cad.model,stock=True)
        actors={k:player(raw,reg,ready,used,rank,memories.get(k)) for k in arm}
        try:
            for index,entry in enumerate(using[len(d["rows"]):n],start=len(d["rows"])):
                c=entry["card"]; shot=shots[c["id"]]; results={}
                for k,b in actors.items():
                    prior(b)
                    if k!="original":restore(b,"practice",journal,len(initial["rows"]))
                    r=checked_play(a,cam,rig,stride,lo,hi,ready,c,shot,raw,b,"frozen",3 if hard else CAP)
                    r["body_proposals"]=b.proposals
                    physical=r["placements"]["delivered"]
                    if hard:
                        s=entry["offer"]; q=physical[0] if physical else None
                        r["matched"]=bool(q and all(same_trace(q[z],s[z]) for z in ("far_contact","near_bounce")))
                        r["clearance"]=validate_game(a,c,shot,r,guard)
                    elif r["score"]["learner"]>=100:
                        r["clearance"]=validate_game(a,c,shot,r,guard)
                    results[k]=r
                d["rows"].append(dict(index=index,case=entry,results=results));_write(dest,d)
                print("hard" if hard else "whole",index+1,c["id"],[(k,r["score"]["learner"],len(r["body_proposals"]),r.get("matched")) for k,r in results.items()],flush=True)
        finally:
            for b in actors.values():b.close()
            cam.close();cad.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","practice","hard-plan","coach","fit","assess","hard"))
    p.add_argument("--feeds",type=Path,default=FEEDS);p.add_argument("--n",type=int,default=40)
    a=p.parse_args()
    if a.mode=="plan":freeze(a.feeds)
    elif a.mode=="practice":training(a.feeds,a.n)
    elif a.mode=="hard-plan":hard_plan(a.feeds)
    elif a.mode=="coach":
        from . import tt90_attained_practice as coach
        coach.run(a.feeds)
    elif a.mode=="fit":fit(a.feeds)
    else:assess(a.feeds,a.n,a.mode=="hard")
