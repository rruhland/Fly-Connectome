"""Try five previously scored direct rates when the ordinary legal screen is empty."""
import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as combined
from . import tt46_continuous_season as season
from . import tt50_surprise_data as data
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried

ROOT = data.ROOT
HELD = ROOT / "held-roster.json"
SCALE = np.array([.12,.20,.083,.58,.78,.84,.061,.07,.042,.38,
                  .63,.72,.69,.62,.79,.16,.20,.20,.20,.25,.22,.26,.22,.22,.26]) * 5.
MOTOR = .46*np.sqrt(25.)
THRESHOLD = .72
WIDTH = .64
MODES = ("stock","state-only","action-conditional")


def freeze_held():
    plan=_read(data.PLAN); base=_read(data.BASELINE)
    if plan["held_source_sha256"]!=_hash(data.BASELINE):
        raise ValueError("The original frozen season reference changed")
    picks, used, controls = [], set(), 0
    for i,(c,r) in enumerate(zip(base["cards"],base["rows"],strict=True)):
        ident=c["id"]; offers=r["placements"]["delivered"]
        if ident in used or not offers:
            continue
        o=offers[0]
        opens=bool(o["near_landed"])
        if ident in data.OPENINGS and not opens and any(
            data.eligible(n) and n["tick"]>o["far_contact"]["step"] for n in r["searches"]
        ):
            role="weak opening"
        elif ident not in data.OPENINGS and opens and controls<4:
            role="ordinary control"
            controls+=1
        else:
            continue
        used.add(ident)
        picks.append(dict(i=i,id=ident,card=c,role=role))
    if len(picks)!=8 or len(picks)-controls!=4:
        raise RuntimeError("Need four actual archived weak and four successful opening assignments")
    _write(HELD,dict(training_plan_sha256=_hash(data.PLAN),source_sha256=_hash(data.BASELINE),
                     held=picks, reason="Repeated release IDs have different draw streams. Freeze the actual weak assignment, not the first occurrence of that release. No treated outcomes exist."))
    print([(r["id"],r["i"],r["role"]) for r in picks],flush=True)


def corpus():
    manifest, saved = _read(data.PLAN),_read(data.PACKET)
    if saved["plan_sha256"]!=_hash(data.PLAN) or not saved["complete"]:
        raise ValueError("Expected the complete original physical recovery corpus")
    rows=[]
    for r in saved["rows"]:
        if "actions" not in r:
            continue
        case=manifest["training"][r["index"]]
        rows.append(dict(r,id=case["id"],good=case["success"],decision=case["decision"],
                         earlier_native_no_legal=True,later_legal=case["later_nominal_legal"]))
    if {v["id"] for v in rows}&set(manifest["held_excluded"]):
        raise ValueError("Held release admitted into physical outcome memory")
    return rows


class Retrieval:
    def __init__(self, rows):
        self.rows=rows
        self.z=np.array([r["z"] for r in rows])
        self.x=self.z/SCALE
        self.u=np.array([r["actions"] for r in rows])
        self.u0=self.u[:,:,:5].reshape(len(rows),-1)/MOTOR
        self.y=np.array([r["good"] for r in rows],dtype=float)
        self.g=np.array([r["id"] for r in rows])
        self.tree=cKDTree(self.x)

    def local(self,z):
        z=np.asarray(z)
        d,j=self.tree.query(z/SCALE,k=min(80,len(self.rows)))
        d,j=np.atleast_1d(d),np.atleast_1d(j)
        # A five-tick prefix only has a useful interpretation at a similar
        # phase of the actual incoming flight and joint reference movement.
        aligned=(abs(self.z[j,1]-z[1])<.22)&(abs(self.z[j,2]-z[2])<.105)
        aligned&=np.max(abs(self.z[j,10:15]-z[10:15]),axis=1)<.66
        d,j=d[aligned],j[aligned]
        _, keep=np.unique(self.g[j],return_index=True)
        take=np.sort(keep)[:30]
        good=d[take]<1.4
        return d[take][good],j[take][good]

    def estimate(self, z, actions=None):
        d,j=self.local(z)
        w=np.exp(-.5*(d/WIDTH)**2)
        state=(1.4*self.y.mean()+w@self.y[j])/(1.4+w.sum())
        if actions is None:
            return state
        u=np.asarray(actions)[:,:5].ravel()/MOTOR
        action=np.linalg.norm(self.u0[j]-u,axis=1)
        c=w*np.exp(-.5*(action/.86)**2)
        return (.75*state+c@self.y[j])/(.75+c.sum())

    def choose(self, z, mode):
        d,j=self.local(z)
        use=np.flatnonzero((d<THRESHOLD)&(self.y[j]>0.))
        if not len(use):
            return None,dict(why="no close scored movement", nearest=float(d[0]) if len(d) else None)
        ww=np.exp(-.5*(d/WIDTH)**2)
        prior=(1.4*self.y.mean()+ww@self.y[j])/(1.4+ww.sum())
        if mode=="state-only":
            chosen=use[0]
            return self.u[j[chosen]],dict(donor=int(self.g[j[chosen]]), state_distance=float(d[chosen]),
                                         state_association=float(prior))
        votes=self.y[j]
        eligible_actions=[]
        for k in use[:12]:
            a=np.linalg.norm(self.u0[j]-self.u0[j[k]],axis=1)
            weight=ww*np.exp(-.5*(a/.86)**2)
            q=(.75*prior+weight@votes)/(.75+weight.sum())
            witnesses=(a<.82)&(d<.93)&(votes>0)
            if witnesses.sum()<2 or weight.sum()<1.2 or q-prior<.09:
                continue
            eligible_actions.append((float(q-.14*d[k]),k,float(q),int(witnesses.sum())))
        if not eligible_actions:
            return None,dict(why="no action-specific positive support", state_association=float(prior),
                             nearest_success=float(d[use[0]]))
        _,k,q,count=max(eligible_actions)
        return self.u[j[k]],dict(donor=int(self.g[j[k]]),state_distance=float(d[k]),
                    supported_releases=count,state_association=float(prior),action_association=q)


def validate():
    rows=corpus()
    g=np.array([r["id"] for r in rows])
    y=np.array([r["good"] for r in rows],dtype=float)
    groups=np.random.default_rng(500225).permutation(np.unique(g))
    p=np.empty((len(y),3))
    for s in np.array_split(groups,6):
        tst=np.isin(g,s)
        bank=Retrieval([r for r,t in zip(rows,tst,strict=True) if not t])
        for i in np.flatnonzero(tst):
            r=rows[i]
            p[i]=bank.y.mean(),bank.estimate(r["z"]),bank.estimate(r["z"],r["actions"])
    brier=np.mean((p-y[:,None])**2,axis=0)
    result=dict(corpus_sha256=_hash(data.PACKET),plan_sha256=_hash(data.PLAN),
                rows=len(y),real_legal=int(y.sum()), releases=len(groups),
                brier=dict(zip(("prevalence","state_only","state_and_issued_actions"),brier.tolist(),strict=True)),
                sample=[dict(id=int(g[i]),earned=bool(y[i]),scores=p[i].tolist()) for i in range(len(y))],
                scope="Group-held observational prediction of actual reward, not the effect of switching motions.")
    _write(ROOT/"validation.json",result)
    print(len(y), result["real_legal"],result["brier"],flush=True)


class Player(season.Player):
    def __init__(self,*a,retrieval,stride,**kw):
        super().__init__(*a,**kw)
        self.retrieval,self.stride=retrieval,stride
        self.mode="stock"

    def begin(self):
        super().begin()
        self.camera_history=deque(maxlen=7)
        self.borrowed=None
        self.at=-99
        self.already=False
        self.calls=[]
        self.sent=[]

    def step(self,t,eye,x,q,ref,base):
        self.camera_history.append((t,x,q,ref))
        original=super().step(t,eye,x,q,ref,base)
        if x is not None and x[4]>.9:
            self.borrowed=None
            self.already=False
        if self.mode=="stock":
            return original
        if self.borrowed is not None and 0<=t-self.at<data.STEPS:
            v=self.borrowed[t-self.at]
            self.sent.append(dict(tick=t,rates=v.tolist()))
            return ref+self.stride*v
        if self.already or not self.notes or self.notes[-1]["tick"]!=t or not data.eligible(self.notes[-1]):
            return original
        z=data.key(self.camera_history)
        if z is None:
            return original
        action, meta=self.retrieval.choose(z,self.mode)
        self.calls.append(dict(tick=t,treatment=action is not None,forecast_legal=0, **meta))
        if action is None:
            return original
        self.borrowed,self.at,self.already=action,t,True
        self.sent.append(dict(tick=t,rates=action[0].tolist()))
        return ref+self.stride*action[0]


def run(output):
    plan=_read(data.PLAN); baseline=_read(data.BASELINE)
    trial=_read(HELD)
    if plan["held_source_sha256"]!=_hash(data.BASELINE) or baseline["carry_sha256"]!=_hash(season.CARRY):
        raise ValueError("Frozen physical baseline or practice changed")
    if trial["source_sha256"]!=_hash(data.BASELINE) or trial["training_plan_sha256"]!=_hash(data.PLAN):
        raise ValueError("Different held assignments")
    receipt=dict(plan_sha256=_hash(data.PLAN), memory_sha256=_hash(data.PACKET),
                 evaluation_sha256=_hash(HELD),
                 actor_sha256=_hash(Path(__file__)), original_carry_sha256=_hash(season.CARRY),
                 rows=[],complete=False)
    if output.exists():
        old=_read(output)
        if any(old[k]!=v for k,v in receipt.items() if k not in ("rows","complete")):
            raise ValueError("Different evaluation configuration")
        receipt=old
    held={v["id"] for v in trial["held"]}
    bank=Retrieval(corpus())
    risk=combined.shot.Coverage(_read(combined.RISK))
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,"local")
    ready=baseline["ready"]
    with mpc.centered(ready):
        a=mpc.arena();res=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda p,e:Carried(p,e,practice=[course.PRACTICE],held=held)):
            player=Player(mpc.BANK,raw,coverage=risk,variant="elastic",budget=28,
                          retrieval=bank,stride=res[2])
        archived_bank=_read(season.CARRY)
        try:
            course.resume(player,season.CARRY.with_suffix("").with_suffix(".npz"),archived_bank)
            serial=player.serial
            for item in trial["held"][len(receipt["rows"]):]:
                i=item["i"]
                c=baseline["cards"][i]; row=baseline["rows"][i]
                if c!=item["card"] or row["memory"]["before"]!=player.priors.digest():
                    raise ValueError("Not the same archived physical start or motor memory")
                modes={}
                for m in MODES:
                    player.mode=m
                    player.serial=serial+i
                    r=mpc.play(a,*res,ready,c,row["shot"],raw,player,course.OPTIC,"frozen",2)
                    r.update(attempt=list(player.calls),borrowed=list(player.sent))
                    if any(q["target"] != list(course.draw(c,j,"random",None)[0])
                           for j,q in enumerate(r["placements"]["requested"])):
                        raise ValueError("Original continuous draw changed")
                    if m=="stock":
                        pre=baseline["rows"][i]
                        if r["actions"]!=pre["actions"][:len(r["actions"])]:
                            raise ValueError("Frozen stock did not reproduce its archived opening")
                    else:
                        base=modes["stock"]
                        n=next((k for k,(u,v) in enumerate(zip(base["actions"],r["actions"])) if u!=v),None)
                        if n is None and r["actions"]!=base["actions"]:
                            raise ValueError("Length changed despite identical native commands")
                        if n is not None and (not r["borrowed"] or n<r["borrowed"][0]["tick"]):
                            raise ValueError("Dynamics differed before the first recalled action")
                        r["same_offer"]=False
                        if base["placements"]["delivered"] and r["placements"]["delivered"]:
                            b0,m0=(w["placements"]["delivered"][0] for w in (base,r))
                            r["same_offer"]=all(digest_json(b0[k])==digest_json(m0[k])
                                                for k in ("far_contact","near_bounce"))
                        # Preserve any collision before the offer, including a
                        # physically changed delivery, without crediting a
                        # rescue against an unmatched ball.
                        r["first_change"]=n
                    modes[m]=r
                receipt["rows"].append(dict(case=item,arms=modes))
                receipt["complete"]=len(receipt["rows"])==len(trial["held"])
                _write(output,receipt)
                print(c["id"],item["role"],[(m,r["score"]["learner"],len(r["borrowed"])) for m,r in modes.items()],flush=True)
        finally:
            player.close();res[0].close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("plan-held","validate","run"))
    p.add_argument("--output",type=Path,default=ROOT/"held.json.gz")
    a=p.parse_args()
    {"plan-held":lambda a:freeze_held(),"validate":lambda a:validate(),
     "run":lambda a:run(a.output)}[a.phase](a)
