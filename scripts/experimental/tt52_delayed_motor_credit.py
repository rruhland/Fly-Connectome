"""Credit actually executed, scored joint fragments for later native returns."""

import argparse
from ast import literal_eval
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt4_transfer_trial import _hash, _read, _write

mpc=course.mpc
ROOT=Path("artifacts/tt52_delayed_motor_credit")
STOCK=season.ROOT/"learning-12.json.gz"
START=season.CARRY
PLAN=ROOT/"protocol.json"
OUT=ROOT/"paired.json.gz"
GAMMA=.55
RADIUS=2.65
SHIFT=.23
SHAM_SEED=52071


class Credit:
    """Causal two-step labels. No geometry, ray, forecast, or opponent intent."""

    def __init__(self):
        self.links={}
        self.serial=None
        self.last=0

    def begin(self,serial):
        self.serial=serial
        self.last=0

    def reward(self,score):
        if score!=self.last+1:
            raise ValueError("Reward score is not a single newly earned legal return")
        self.links[self.serial,score]=(0,0)
        if score>=2:
            key=(self.serial,score-1)
            self.links[key]=(1,self.links[key][1])
        if score>=3:
            key=(self.serial,score-2)
            self.links[key]=(self.links[key][0],1)
        self.last=score

    def loss(self):
        if self.last:
            self.links[self.serial,self.last]=(-1,0)
        if self.last>=2:
            self.links[self.serial,self.last-1]=(1,-1)

    def grade(self,serial):
        return {k:v for k,v in self.links.items() if k[0]==serial}

    def values(self):
        # 0 denotes an unresolved or censored outcome, not a miss.
        return {k:(a+GAMMA*b)/(1.+GAMMA*bool(b)) for k,(a,b) in self.links.items() if a}

    def prune(self,online):
        present={tag[1:] for _,_,tag in online if is_live(tag)}
        self.links={k:v for k,v in self.links.items() if k in present}


def is_live(tag):
    return len(tag)==3 and tag[0]=="live"


def annotate(row,serial):
    """Scorer anatomy using only completed public landing and terminal events."""
    c=Credit()
    c.begin(serial)
    own=[s["landing"] for s in row["score"]["shots"] if s["side"]==1]
    if len(own)!=row["score"]["learner"]:
        raise ValueError("Archived learner returns missing physical score")
    for i,_ in enumerate(own):
        c.reward(i+1)
    if own:
        last=own[-1]
        offered=any(e["kind"]=="legal_return" and e["side"]==-1 and e["time"]>last
                    for e in row["events"])
        done=any(e["kind"]=="point" for e in row["events"])
        if offered and done:
            c.loss()
    return c.links


def sham_values(values):
    """Preserve the same supervised outcomes per game, disrupt action attribution."""
    by={}
    for k,value in sorted(values.items()):
        by.setdefault(k[0],[]).append((k,value))
    mix={}
    for serial,rows in by.items():
        keys,v=zip(*rows,strict=True)
        rng=np.random.default_rng(SHAM_SEED+serial)
        mix.update(zip(keys,rng.permutation(v),strict=True))
    return mix


def grades(states,motors,values,ids,dist,known):
    """Compare matched state+action support to the same-state reference.

    A row holds one already scored stroke, never multiple temporal frames
    counted as different outcomes. Each historical game has at most one
    effective vote, preventing success streaks from swamping other games.
    """
    n=len(dist)
    if n<4 or known.sum()<4 or len(np.unique(ids[known]))<3:
        return np.zeros(n),np.zeros(n)
    z=states[:,None,:]-states[None,:,:]
    var=(z*z).sum(axis=-1)
    sr=np.exp(-.5*var/(1.16**2))*np.exp(-.5*(dist/1.92)**2)[None,:]
    independent=(ids[:,None]!=ids[None,:]) & known[None,:]
    sr*=independent
    _,idx=np.unique(ids,return_inverse=True)
    counts=np.maximum(1.,sr@np.eye(idx.max()+1)[idx])
    sr/=counts[:,idx]
    ref_n=sr.sum(axis=1)
    ref=sr@values / np.maximum(ref_n,1.e-9)
    da=(motors[:,None,:]-motors[None,:,:])
    ma=np.exp(-.5*np.mean(da*da,axis=-1)/(.19**2))
    wr=ma*sr
    agreement=wr.sum(axis=1)
    nr=(wr@(np.eye(idx.max()+1)[idx]) > .16).sum(axis=1)
    good=known&(dist<RADIUS)&(ref_n>1.25)&(agreement>.85)&(nr>=2)
    # A single old scored stroke is a small residual on independently
    # supported, similar motions. Negative terminal evidence never erases it.
    effect=((wr@values + .48*values) - (agreement+.48)*ref)/(agreement+1.25)
    credit=np.where(good, np.clip(.62*effect,-SHIFT,SHIFT),0.)
    return credit,good


class Bank(Carried):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.credit=Credit()
        self.view=None
        self.telemetry=Counter()
        self.changed_samples=[]
        self.tick=None

    def start(self,serial):
        self.credit.begin(serial)
        self.view=None
        self.telemetry=Counter()
        self.changed_samples=[]

    def keep(self,archive,leg):
        if not is_live(leg) or leg[1]!=self.credit.serial:
            raise ValueError("An unexpected motor credit tag reached rewards")
        self.credit.reward(leg[2])
        self.view=None
        self.telemetry["new_reward_callbacks"]+=1
        return super().keep(archive,leg)

    def lost(self):
        self.credit.loss()
        self.view=None
        self.telemetry["native_loss_callbacks"]+=1

    def sync(self):
        data=list(self.online)
        self.cache=data
        self.states=np.asarray([r[0] for r in data]).reshape(-1,len(mpc.OPEN_NORM))
        self.motors=np.asarray([r[1][[3,9,16],:5].ravel() for r in data]).reshape(-1,15)
        valid=self.credit.values()
        shuffled=sham_values(valid)
        self.values=np.asarray([valid.get(r[2][1:],0.) if is_live(r[2]) else 0. for r in data])
        self.sham=np.asarray([shuffled.get(r[2][1:],0.) if is_live(r[2]) else 0. for r in data])
        self.known=np.asarray([is_live(r[2]) and r[2][1:] in valid for r in data],dtype=bool)
        self.groups=np.asarray([r[2][1] if is_live(r[2]) else -1 for r in data],dtype=int)
        self.view=True

    def ordered(self,dist):
        selected=[]
        tags=set()
        for i in np.argsort(dist,kind="stable"):
            tag=self.cache[i][2]
            if tag not in tags:
                selected.append(i)
                tags.add(tag)
        return np.array(selected,dtype=int)

    def priors(self,x,q,ref,limit):
        if not self.view:
            self.sync()
        state=mpc.query(x,q,ref)
        sd,si=self.tree.query(state,min(len(self.rows),160))
        static=[(float(d),self.rows[j][1],self.rows[j][2],False) for d,j in zip(sd,si,strict=True)]
        dist=np.maximum(0.,np.linalg.norm(self.states-state,axis=1)-.10)
        ordered=self.ordered(dist)
        short=ordered[:32][dist[ordered[:32]]<RADIUS+.30]
        online=[(float(dist[i]),self.cache[i][1],self.cache[i][2],True) for i in ordered]
        def retrieve(delta):
            varied=[(max(0.,d-float(w)),g,k,b) for (d,g,k,b),w in zip(online,delta)]
            records=static+varied+online[len(varied):]
            used=set()
            keys=[]
            answer=[]
            for d,goals,tag,online_flag in sorted(records,key=lambda r:r[0]):
                if tag in used:
                    continue
                if d>3.1 and len(answer)>=2:
                    break
                keys.append(tag)
                answer.append((goals,online_flag))
                used.add(tag)
                if len(answer)>=limit:
                    break
            return keys,answer
        organic,stock=retrieve(())
        self.telemetry["retrievals"]+=1
        if self.telemetry["retrievals"]%64==1:
            literal=super().priors(x,q,ref,limit)
            if len(literal)!=len(stock) or any(a[1]!=b[1] or not np.array_equal(a[0],b[0])
                                                for a,b in zip(literal,stock,strict=True)):
                raise ValueError("Zero credit failed exact stock retrieval parity")
            self.telemetry["verified_stock_retrievals"]+=1
        if len(short)<4:
            return [(g.copy(),flag) for g,flag in stock]
        sets=(self.states[short],self.motors[short],self.groups[short],dist[short],self.known[short])
        w,qualified=grades(*sets[:2],self.values[short],*sets[2:])
        s,_=grades(*sets[:2],self.sham[short],*sets[2:])
        if np.any(qualified):
            self.telemetry["supported_queries"]+=1
        actual,answer=retrieve(w)
        pseudo,_=retrieve(s)
        if actual!=organic:
            self.telemetry["credit_changed_retrievals"]+=1
            if len(self.changed_samples)<40:
                self.changed_samples.append(dict(tick=self.tick,stock=organic,delayed=actual,sham=pseudo))
        if pseudo!=organic:
            self.telemetry["sham_changed_retrievals"]+=1
        return [(g.copy(),flag) for g,flag in answer]


class Actor(season.Player):
    def begin(self):
        super().begin()
        self.priors.start(self.serial)

    def search(self,tick,*args,**kwargs):
        self.priors.tick=int(tick)
        return super().search(tick,*args,**kwargs)

    def missed(self,history):
        self.priors.lost()


def document(out,n):
    stock,start=_read(STOCK),_read(START)
    sched=_read(season.PLAN)
    if not 1<=n<=len(stock["rows"]) or len(start["rows"])!=48:
        raise ValueError("Incomplete ancestor or physical control")
    if (stock["cards"]!=sched["practice"][:len(stock["cards"])] or stock["mode"]!="live"
            or stock["carry_sha256"]!=_hash(START) or stock["cap"]!=240
            or stock["rows"][0]["memory"]["before"]!=start["rows"][-1]["memory"]["after"]
            or stock["source_sha256"]!=_hash(mpc.BANK) or stock["optical_sha256"]!=_hash(course.OPTIC)):
        raise ValueError("Not the exact stock continuous season")
    cards=stock["cards"][:n]
    if len({c["id"] for c in cards})!=n or set(c["id"] for c in start["cards"]) & {c["id"] for c in cards}:
        raise ValueError("Training leaked into the physical matched course")
    dest=START.with_suffix("").with_suffix(".npz")
    with np.load(dest) as z:
        tags=list(map(literal_eval,z["tags"]))
        serial=int(z["serial"])
    all_links={}
    for i,r in enumerate(start["rows"],1):
        all_links.update(annotate(r,i))
    seed={k:all_links[k] for k in {t[1:] for t in tags if is_live(t)}}
    if serial!=len(start["rows"]) or not seed:
        raise ValueError("Unexpected carried label indexing")
    out.parent.mkdir(parents=True,exist_ok=True)
    _write(out,dict(n=n,cards=cards,source_sha256=_hash(Path(__file__)),stock_sha256=_hash(STOCK),
                    start_sha256=_hash(START),checkpoint_sha256=_hash(dest),risk_sha256=_hash(fresh.RISK),
                    schedule_sha256=_hash(season.PLAN),caps=[768,240],same_stock_actions_until_credit=True,
                    start_credit=[list(k+v) for k,v in sorted(seed.items())],
                    commitment="Fixed unfiltered stock prefix, complete live physical games with the exact same incoming streams. Only how existing scored motor fragments are retrieved changes. Keep/evict capacity and local trial budget stay stock. Never consume modeled effects or future optical/native state. Same-data sham is a shadow comparison on identical queries, not an actuated arm."))
    print("plan",n, len(seed), Counter(seed.values()),flush=True)


def play(p,out):
    spec,start,stock=_read(p),_read(START),_read(STOCK)
    files={"source_sha256":Path(__file__),"stock_sha256":STOCK,"start_sha256":START,
           "checkpoint_sha256":START.with_suffix("").with_suffix(".npz"),
           "risk_sha256":fresh.RISK,"schedule_sha256":season.PLAN}
    if any(spec[k]!=_hash(v) for k,v in files.items()) or spec["cards"]!=stock["cards"][:spec["n"]]:
        raise ValueError("Changed sealed credit protocol")
    result=dict(rows=[],plan_sha256=_hash(p),complete=False)
    doc=out.with_suffix("").with_suffix(".npz")
    if out.exists():
        result=_read(out)
        if result["plan_sha256"]!=_hash(p):
            raise ValueError("Cannot continue a different course")
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    ready=stock["ready"]
    with mpc.centered(ready):
        a=mpc.arena()
        cam,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda b,e: Bank(b,e,practice=[course.PRACTICE],
                         held={c["id"] for c in spec["cards"]})):
            player=Actor(mpc.BANK,raw,variant=stock["variant"],budget=stock["budget"],
                         coverage=fresh.shot.Coverage(_read(fresh.RISK)))
        try:
            credit=player.priors.credit
            if result["rows"]:
                course.resume(player,doc,{**result,"rows":start["rows"]+result["rows"]})
                credit.links={(g,s):(x,y) for g,s,x,y in result["retained_credit"]}
            else:
                course.resume(player,files["checkpoint_sha256"],start)
                credit.links={(g,s):(x,y) for g,s,x,y in spec["start_credit"]}
            for i in range(len(result["rows"]),spec["n"]):
                c,old=spec["cards"][i],stock["rows"][i]
                launches=a.launches
                r=mpc.play(a,cam,rig,stride,lo,hi,ready,c,old["shot"],raw,player,
                           course.OPTIC,"live",spec["caps"][1],0)
                if a.launches!=launches+1:
                    raise ValueError("Not one continuous game")
                expected=annotate(r,player.serial)
                if credit.grade(player.serial)!=expected:
                    raise ValueError("Causal callbacks disagree with independent scorer")
                for leg,resulting in enumerate(r["placements"]["requested"]):
                    if resulting["target"]!=list(draw(c,leg,"random",None)[0]):
                        raise ValueError("Opponent used a different random draw")
                d=next((j for j,(u,v) in enumerate(zip(old["actions"],r["actions"])) if u!=v),
                       None if len(old["actions"])==len(r["actions"]) else min(len(old["actions"]),len(r["actions"])))
                if d is None and digest_json(r["events"])!=digest_json(old["events"]):
                    raise ValueError("Equal actions produced unequal events")
                if d is not None and digest_json([e for e in r["events"] if e["step"]<=d])!=digest_json(
                     [e for e in old["events"] if e["step"]<=d]):
                    raise ValueError("Physical delivery diverged before issuance")
                r.update(credit_telemetry=dict(player.priors.telemetry),
                         first_different_action=d, changed_matchups=player.priors.changed_samples,
                         stock_score=old["score"]["learner"],id=c["id"])
                result["rows"].append(r)
                result["complete"]=len(result["rows"])==spec["n"]
                credit.prune(player.priors.online)
                result["retained_credit"]=[list(k+v) for k,v in sorted(credit.links.items())]
                result["checkpoint_sha256"]=course.save(player,doc)
                _write(out,result)
                print(i,c["id"],old["score"]["learner"],r["score"]["learner"],d,
                      dict(player.priors.telemetry),flush=True)
        finally:
            player.close();cam.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--n",type=int,default=8)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    document(a.plan,a.n) if a.mode=="plan" else play(a.plan,a.out)
