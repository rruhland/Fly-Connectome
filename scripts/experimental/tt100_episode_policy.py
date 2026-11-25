"""Train a whole-point optical search policy with paired antithetic native play."""
import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt97_return_to_center as transfer
from . import tt80_prospective_course as reserved80
from .tt39_carry_joints import Carried
from .tt4_transfer_trial import _hash,_read,_write
from .tt54_training_speedups import enabled

season,course=transfer.season,transfer.course
mpc=course.mpc
ROOT=Path("artifacts/tt100_episode_policy")
PLAN=ROOT/"protocol.json"
PRACTICE=ROOT/"practice.json.gz"
CHECK=ROOT/"policy.json"
OUT=ROOT/"held.json.gz"
PARENTS=tuple(Path("artifacts")/(r+".json.gz") for r in ("tt42/after48","tt42/diverse-evaluation"))
MOTOR=transfer.PRACTICE.with_suffix("").with_suffix(".npz")
SIGMA=.68
BOUND=1.2
STEP=.46
RNG=1000721
NORMAL=8700
NTRAIN=6
NHELD=6


def settings(theta):
    z=np.clip(np.asarray(theta,dtype=float),-BOUND,BOUND)
    return -.49+.18*z[0], float(np.exp(.48*z[1]))


class Rhythm(season.Player):
    """Only change when search opens and how far a sampled proposal moves."""
    def __init__(self,*args,theta,**kw):
        super().__init__(*args,**kw)
        self.theta=np.array(theta,dtype=float)
        self.start,self.radius=settings(theta)

    def search(self,tick,x,q,ref,base):
        if self.theta[1]==0.0:
            return super().search(tick,x,q,ref,base)
        native=mpc.structured

        def propose(goal,noise,scale=1.):
            return native(goal,noise,scale*self.radius)
        with patch.object(mpc,"structured",propose):
            return super().search(tick,x,q,ref,base)

    def step(self,tick,eye,x,q,ref,base):
        if self.theta[0]==0.0:
            return super().step(tick,eye,x,q,ref,base)
        # Exact stock temporal/fallback arithmetic. A global public y onset
        # is the only modification to the decision schedule; once eligible,
        # ordinary cadence, confidence, safeguard, ranker and lease remain.
        if x is None or x[4]>=-.9:
            self.samples.clear()
        else:
            self.samples.append(np.r_[x,q[:5]])
            if len(self.samples)==7 and -.57<x[1]<1.35:
                servo=season.fresh.servo
                inferred=q[:5]+self.motor.predict(servo.feature(self.samples))[0]
                base=base.copy()
                base[:5]+=.65*np.clip(inferred-base[:5],-.25,.25)
                self.diag["temporal_inbound_steps"]+=1
        self.engine.observe(tick,eye,q,ref.copy())
        if x is not None and x[4]>.9:
            self.best=None
            return base
        if x is not None and x[4]<-.9 and self.start>x[1]>-1.62 and x[2]>.77 and tick>=self.next:
            self.next=tick+5
            self.search(tick,x,q,ref,base)
        if self.best is not None and 0<=tick-self.tick<self.lease:
            self.diag["screened_ticks"]+=1
            return self.best[tick-self.tick]
        return base


def available():
    bank,carry=(_read(f) for f in (transfer.PRACTICE,season.CARRY))
    ids={c["id"] for d in (bank,carry) for c in d["cards"][:len(d["rows"])]}
    for f in (mpc.BANK,course.PRACTICE):
        ids.update(r["card"]["id"] for r in _read(f.with_suffix(".json"))["practice"])
    future={c["id"] for f in (transfer.PLAN,reserved80.PLAN) for c in _read(f)["practice"]}
    candidates={}
    observed={}
    for path in PARENTS:
        d=_read(path)
        for c,r in zip(d["cards"],d["rows"]):
            if (c["id"] in ids|future or c["opening"]!="random" or c["profile"]!="reach"
                    or r["score"]["learner"]<1):
                continue
            candidates[c["id"]]=c
            observed[c["id"]]=max(observed.get(c["id"],0),r["score"]["learner"])
    return ids,future,candidates,observed


def plan(feeds):
    used,future,cards,observed=available()
    if len(cards)<NTRAIN:
        raise ValueError("Insufficient separate episode-practice releases")
    rng=np.random.default_rng(RNG)
    sorted_ids=sorted(cards,key=lambda k:(observed[k],k))
    strata=np.array_split(sorted_ids,3)
    # Six points spanning the prior stock success spectrum. Each block
    # mixes short and longer historical receiving, not picked treatment trials.
    picked=[list(rng.permutation(s)[:2]) for s in strata]
    ids=[picked[j][(i+j)%2] for i in (0,1) for j in range(3)]
    shots=mpc.measured_launches(feeds)
    train=[]
    for j in range(3):
        direction=rng.normal(size=2)
        direction/=np.linalg.norm(direction)
        draw=[]
        for k in (ids[j],ids[(j+1)%3+3]):
            card=dict(cards[k],seed=int(rng.integers(2**30)),reply_seed=int(rng.integers(2**30)))
            draw.append(dict(card=card,shot=shots[k],order=["minus","plus"] if rng.uniform()<.5 else ["plus","minus"],history=observed[k]))
        train.append(dict(direction=direction.tolist(),cards=draw))
    hold=[]
    seen=set()
    for c in _read(transfer.PLAN)["practice"]:
        if c["id"] in used or c["id"] in seen:
            continue
        hold.append(dict(card=c,shot=shots[c["id"]]))
        seen.add(c["id"])
        if len(hold)>=NHELD:
            break
    if len(hold)!=NHELD or seen.intersection(ids):
        raise ValueError("Cannot separate policy search from the reserved final stream")
    # TT80 exact cards are not used by this fit: all its releases are
    # already represented in the verified TT73 starting memory.
    available80=sum(c["id"] not in used for c in _read(reserved80.PLAN)["practice"])
    fixed=(*PARENTS,transfer.PRACTICE,MOTOR,transfer.PLAN,reserved80.PLAN,season.CARRY,
           mpc.BANK,course.OPTIC,course.PRACTICE,season.fresh.RISK)
    ROOT.mkdir(parents=True,exist_ok=True)
    _write(PLAN,dict(schema="tt100-episodic-native-policy-search-1",
         sources={str(f):_hash(f) for f in fixed},seed=RNG,
         ready=_read(mpc.BANK.with_suffix(".json"))["ready"],practices=train,held=hold,
         optimizer=dict(radius=SIGMA,step=STEP,bound=BOUND,utility="log(1+actual legal arm returns) + .015 times actual legal arm returns"),
         withheld_tt80_eligible=available80,train_release_ids=[int(i) for i in ids],
         body="original centered setup after acquired .12 m shifted-body TT73 uniform motor bank",
         role="No per-shot credit, fake resets, or selected benchmark reruns. Paired ± whole policies receive common preassigned native seeds during separate practice points, accumulate actual referee reward only, and may move the common two-weight policy after the block. The held pair uses the first six distinct untouched TT97 assignments, original bank fixed, with no per-point policy update. TT80 remains untouched because its release IDs are in TT73/source memory. No opens, replies, motors, or safety limits changed."))
    print("TT100 practice",[[e["card"]["id"] for e in b["cards"]] for b in train],
          "final",[x["card"]["id"] for x in hold],"TT80 eligible",available80,flush=True)


def design():
    p=_read(PLAN)
    if any(_hash(Path(f))!=v for f,v in p["sources"].items()):
        raise ValueError("Whole-policy protocol changed")
    return p


def utility(r):
    return float(np.log1p(r)+.015*r)


def update(theta,direction,values):
    raw=[item["plus"]-item["minus"] for item in values]
    delta=np.mean([utility(item["plus"])-utility(item["minus"]) for item in values])
    # Contradictory signs shrink, not amplify, the trust-region step.
    agreement=abs(sum(np.sign(raw)))/len(raw)
    confidence=.35+.65*agreement
    z=np.clip(theta+direction*STEP*np.tanh(2.0*delta)*confidence,-BOUND,BOUND)
    return z,dict(deltas=raw,utility_difference=float(delta),sign_agreement=float(agreement),used_contrast_weight=float(confidence))


class Course:
    """Identical original physical body and frozen installed motor memory."""
    def __init__(self, p):
        self.p=p
        self.ready=np.asarray(p["ready"])
        self.rank=season.fresh.shot.Coverage(_read(season.fresh.RISK))
        self.work=mpc.arena()
        self.cam,self.rig,self.stride,self.lo,self.hi=mpc.resources(self.work)
        with np.load(mpc.BANK) as f:
            self.raw=mpc.DeepField(f,"local")
        self.installed=_read(transfer.PRACTICE)
        self.installed=dict(self.installed,rows=[None]*len(_read(season.CARRY)["rows"])+self.installed["rows"])

    def play(self,item,theta):
        c=item["card"]
        with patch.object(mpc,"Fragments",lambda f,e:Carried(f,e,practice=[course.PRACTICE],held={c["id"]})):
            if np.any(theta):
                actor=Rhythm(mpc.BANK,self.raw,variant="elastic",budget=28,coverage=self.rank,theta=theta)
            else:
                actor=season.Player(mpc.BANK,self.raw,variant="elastic",budget=28,coverage=self.rank)
        try:
            course.resume(actor,MOTOR,self.installed)
            begin=process_time()
            from .tt29_endurance import audit
            next_return=20
            exchange=self.work.exchange

            def watched(u,far):
                nonlocal next_return
                packet=exchange(u,far)
                if self.work.legal_returns>=next_return:
                    actual=audit(self.work.events,opening_side=1)["learner"]
                    if actual>=next_return:
                        print("whole point progress",c["id"],actual,"near legal returns",
                              "at tick",self.work.steps,"parameters",theta.tolist(),flush=True)
                        next_return+=20
                return packet
            # Telemetry only; neither the learned policy nor the ordinary
            # actor receives the physical events or reward mid-point.
            with patch.object(self.work,"exchange",watched):
                r=mpc.play(self.work,self.cam,self.rig,self.stride,self.lo,self.hi,self.ready,c,item["shot"],self.raw,
                           actor,course.OPTIC,"frozen",240)
            r["cpu_s"]=process_time()-begin
            r["complete"]=r["events"][-1]["kind"]=="point"
            r["theta"]=theta.tolist()
            if r["memory"]["before"]!=r["memory"]["after"]:
                raise ValueError("Whole-policy test changed installed motor memory")
            from .tt37_opponent import draw
            for leg,x in enumerate(r["placements"]["requested"]):
                if x["target"]!=list(draw(c,leg,"random",None)[0]):
                    raise ValueError("Ordinary continuous opponent law changed")
            return r
        finally:
            actor.close()

    def close(self):
        self.cam.renderer._gl_context.make_current()
        self.cam.close();self.work.close()


def learn(n):
    p=design()
    d=dict(plan=_hash(PLAN),rounds=[],theta=[0.,0.],complete=False)
    if PRACTICE.exists():d=_read(PRACTICE)
    if d["plan"]!=_hash(PLAN):raise ValueError("Wrong policy-learning stream")
    with mpc.centered(p["ready"]),enabled():
        runner=Course(p)
        try:
            for j,block in enumerate(p["practices"][:n]):
                if j<len(d["rounds"]) and d["rounds"][j]["updated"]:
                    continue
                direction=np.asarray(block["direction"])
                if j>=len(d["rounds"]):
                    theta=np.asarray(d["theta"])
                    d["rounds"].append(dict(origin=theta.tolist(),updated=False,matches=[]))
                current=d["rounds"][j]
                theta=np.asarray(current["origin"])
                members={"plus":np.clip(theta+SIGMA*direction,-BOUND,BOUND),
                         "minus":np.clip(theta-SIGMA*direction,-BOUND,BOUND)}
                for k,item in enumerate(block["cards"]):
                    if k>=len(current["matches"]):
                        current["matches"].append(dict(item,arms={}))
                    pair=current["matches"][k]
                    for arm in item["order"]:
                        if arm in pair["arms"]:
                            continue
                        r=runner.play(item,members[arm])
                        pair["arms"][arm]=r
                        _write(PRACTICE,d)
                        print("actual round",j,item["card"]["id"],arm,r["score"]["learner"],
                              len(r["actions"]),round(r["cpu_s"],2),flush=True)
                scores=[{k:v["score"]["learner"] for k,v in pair["arms"].items()} for pair in current["matches"]]
                z,e=update(theta,direction,scores)
                current.update(updated=True,next=z.tolist(),paired=e)
                d["theta"]=z.tolist()
                d["complete"]=j+1==len(p["practices"])
                _write(PRACTICE,d)
                print("policy",j,theta.tolist(),z.tolist(),e,flush=True)
        finally:
            runner.close()
    if d["complete"]:
        _write(CHECK,dict(protocol=_hash(PLAN),evidence=_hash(PRACTICE),theta=d["theta"],
                         start_m=settings(d["theta"])[0],noise_multiplier=settings(d["theta"])[1]))


def held(n):
    p=design()
    ck=_read(CHECK)
    if _hash(PLAN)!=ck["protocol"] or _hash(PRACTICE)!=ck["evidence"]:
        raise ValueError("Changed sealed policy")
    d=dict(plan=_hash(PLAN),policy=_hash(CHECK),rows=[],complete=False)
    if OUT.exists():d=_read(OUT)
    if d["policy"]!=_hash(CHECK) or d["plan"]!=_hash(PLAN):raise ValueError("Wrong frozen comparison")
    options={"ordinary":np.zeros(2),"episode":np.asarray(ck["theta"])}
    with mpc.centered(p["ready"]),enabled():
        runner=Course(p)
        try:
            for j,item in enumerate(p["held"][:n]):
                if j>=len(d["rows"]):
                    d["rows"].append(dict(item,arms={}))
                a=d["rows"][j]
                order=tuple(options) if j%2==0 else tuple(reversed(options))
                for k in order:
                    if k in a["arms"]:continue
                    r=runner.play(item,options[k])
                    a["arms"][k]=r
                    d["complete"]=len(d["rows"])==len(p["held"]) and all(len(a0["arms"])==2 for a0 in d["rows"])
                    _write(OUT,d)
                    print("held",item["card"]["id"],k,r["score"]["learner"],len(r["actions"]),
                          round(r["cpu_s"],2),flush=True)
        finally:
            runner.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","practice","held"))
    p.add_argument("--n",type=int)
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    a=p.parse_args()
    if a.mode=="plan":plan(a.feeds)
    elif a.mode=="practice":learn(a.n or 3)
    else:held(a.n or NHELD)
