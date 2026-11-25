"""Learn broad recovery hubs directly from actually reached, scored encoder paths."""

import argparse
from collections import Counter, deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_safe_drive import PoseClearance
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt40_long_low import same_trace
from .tt48_contact_patch_report import replay
from .tt48_rim_recovery import STOCK
from .tt54_training_speedups import enabled

ROOT=Path("artifacts/tt56_postures")
MODEL=ROOT/"postures.json"
PLAN=ROOT/"protocol.json"
OUT=ROOT/"held.json.gz"
INPUTS=fresh.servo.SOURCES
CAP=12
N=6
LANES=np.array([-.102,.102])


def observations(paths,exclude):
    rows=[]
    seen=set()
    for table in paths:
        memo=_read(table.with_suffix(".json"))
        if _hash(table)!=memo["bank_sha256"]:
            raise ValueError("Changed actual stereo/encoder records")
        with np.load(table) as z:
            states,marks=z["states"],z["marks"]
        at=0
        for r in memo["practice"]:
            until=at+r["frames"]
            p,m=states[at:until],marks[at:until]
            at=until
            rel=r["card"]["id"]
            if rel in exclude:
                continue
            mask=(m[:,1]<-.018)&(m[:,1]>-.40)&(p[:,4]<-.90)
            ticks=np.rint((m[:,0]*mpc.DT-m[:,1])*480).astype(int)
            for sample in np.unique(ticks[mask]):
                key=rel,r["card"]["seed"],r["card"].get("reply_seed"),int(sample)
                if key in seen:
                    continue
                hit=mask&(ticks==sample)
                st,age=p[hit],m[hit,1]
                def close(t):
                    k=int(np.argmin(abs(age-t)))
                    return None if abs(age[k]-t)>.041 else st[k]
                a,b,c=close(-.225),close(-.120),close(-.050)
                if a is None or b is None or c is None:
                    continue
                seen.add(key)
                x=float(c[0])
                rows.append(dict(id=rel,seed=key[1],tick=key[-1], lane=int(np.searchsorted(LANES,x)),
                                 x=x,z=float(c[2]),early=a[6:11].tolist(),
                                 middle=b[6:11].tolist(),late=c[6:11].tolist()))
    return rows


def reach_cost(pose,targets,rate):
    return np.max(abs(np.asarray(targets)-np.asarray(pose))/rate,axis=-1)


def coverage(q,scored,lanes,face,rate):
    # A cautious mobility bound, not a simulated contact score. Use
    # equal lane representation, including ordinary balls on either side.
    reach=reach_cost(q,scored,rate)
    span=np.array([.55*np.mean(reach[lanes==i])+.45*np.quantile(reach[lanes==i],.80)
                      for i in range(3)])
    split=[float(np.mean(reach[face==c])) for c in np.unique(face) if sum(face==c)>=4]
    family=max(split,default=float(span.mean()))
    return dict(lanes=span.tolist(),face_worst=family,
                score=float(.54*max(span)+.34*np.mean(span)+.12*family))


def route(guard,a,b):
    return not any(guard.violations(a+t*(b-a)) for t in np.linspace(0,1,8))


def fit(out):
    archive=_read(STOCK)
    used={c["id"] for c in archive["cards"]}
    rows=observations(INPUTS,used)
    if len(rows)<30 or any(sum(r["lane"]==i for r in rows)<5 for i in range(3)):
        raise ValueError("Insufficient actual earned movement support across the full width")
    home=np.asarray(archive["ready"])
    with mpc.centered(home):
        side=mpc.arena(detail="cad",width=64,height=48)
        try:
            guard=PoseClearance(side)
            rate=.8*side._joint_increments[side._arm_ids[1][2]][:5]/side.model.opt.timestep
            qlate=np.asarray([r["late"] for r in rows])
            lane=np.asarray([r["lane"] for r in rows])
            face=np.asarray([np.sign(guard.ik.pose(q)[1][1]) for q in qlate])
            normal=np.asarray([guard.ik.pose(q)[1][2] for q in qlate])
            for i,r in enumerate(rows):
                r.update(face=int(face[i]),normal_z=float(normal[i]))
            baseline=coverage(home[:5],qlate,lane,face,rate)
            candidate=[]
            for row in rows:
                for tag in ("early","middle"):
                    sample=np.asarray(row[tag])
                    for gain in (.62,1.0):
                        q=np.clip(home[:5]+gain*(sample-home[:5]),guard.ik.low,guard.ik.high)
                        if not .045<np.max(abs(q-home[:5]))<.78:
                            continue
                        stats=coverage(q,qlate,lane,face,rate)
                        candidate.append(dict(stats,q=q.tolist(),id=row["id"],tag=tag,gain=gain))
            candidate.sort(key=lambda t:t["score"])
            portfolio=[]
            for option in candidate:
                q=np.asarray(option["q"])
                if not route(guard,home[:5],q):
                    continue
                if portfolio and any(np.max(abs(q-r["q"]))<.145 for r in portfolio):
                    continue
                if (option["score"]>baseline["score"]+.022
                        or max(option["lanes"])>max(baseline["lanes"])+.030):
                    continue
                p,n,*_=guard.ik.pose(q)
                portfolio.append(dict(option,blade_xyz=p.tolist(),blade_normal=n.tolist()))
                if len(portfolio)==3:
                    break
            if len(portfolio)<2:
                raise ValueError("No two independently reached safe compromise regions")
            _write(out,dict(rows=rows,portfolio=portfolio,home=home.tolist(),home_cost=baseline,
                           inputs={str(p):_hash(p) for p in INPUTS}, excluded=sorted(used),
                           bins=dict(Counter((str(r["lane"]) for r in rows))),
                           face=dict(Counter((str(r["face"]) for r in rows))),
                           scope="Recovery hubs are observed encoder waypoints from early or mid physically rewarded movements, plus their bounded interpolation toward ordinary home. Their score balances movement effort to near-contact successful encoders on all three laterally separated optical bins and observed faces. No outcome or posture from any attended release enters. Route clearance is separate from learner observations."))
            print(len(rows),Counter(r["lane"] for r in rows),Counter(r["face"] for r in rows),
                  baseline,portfolio,flush=True)
        finally:
            side.close()


class Recovery(season.Player):
    def __init__(self,*args,postures,kind,guard,**kw):
        super().__init__(*args,**kw)
        self.hubs=postures
        self.kind=kind
        self.guard=guard

    def begin(self):
        super().begin()
        self.hist=deque(maxlen=7)
        self.passes=[]
        self.inbound=False
        self.goal=None
        self.opened=None
        self.read_stamp=None
        self.agree=0
        self.routes=0
        self.note=None

    def ready_position(self,q):
        history=np.asarray(self.hist)
        rate=(history[-1]-history[0])/max(len(history)-1,1)
        # Use only own encoders to favor a safe route already consistent
        # with the body's current motion, before any target selection.
        extrapolate=q[:5]+np.clip(3.*rate,-.14,.14)
        ordered=list(range(len(self.hubs)))
        if self.kind=="ports":
            ips=[.38*r["score"]+.62*float(np.max(abs(np.asarray(r["q"])-extrapolate)))/5
                 for r in self.hubs]
            ordered=sorted(ordered,key=lambda j:ips[j])
        else:
            ordered=[0]
        for j in ordered:
            p=np.asarray(self.hubs[j]["q"])
            if route(self.guard,q[:5],p) and not self.guard.violations(extrapolate):
                return j,p
        return None,None

    def step(self,t,eye,x,q,ref,base):
        self.hist.append(q[:5].copy())
        if x is not None and x[4]<-.9:
            self.inbound=True
            self.goal=None
            self.opened=None
            self.agree=0
            self.note=None
            return super().step(t,eye,x,q,ref,base)
        if self.kind=="stock":
            return super().step(t,eye,x,q,ref,base)
        now=t*mpc.DT
        f=eye.latest
        positive=False
        if f is not None and -.001<=now-f.time<.085 and f.rms<2.10 and f.depth_sigma<.141:
            p,v=f.advance(now)
            positive=bool(v[1]>1.15 and -.66<p[1]<1.20)
            if self.read_stamp!=f.time:
                close=self.read_stamp is not None and .0<f.time-self.read_stamp<.055
                self.read_stamp=f.time
                self.agree=(self.agree+1 if close else 1) if positive else 0
        if (self.inbound and positive and self.agree>=2 and self.opened is None):
            self.inbound=False
            self.opened=t
            j,goal=self.ready_position(q)
            self.goal=goal
            self.routes+=1
            self.note=dict(start=t,pool=j,anchor=q.tolist(),goal=None if goal is None else goal.tolist(),moves=0)
            self.passes.append(self.note)
        last=base
        if self.goal is not None and self.opened is not None and 0<=t-self.opened<32:
            if positive or t-self.opened<=7:
                candidate=base.copy()
                candidate[:5]=ref[:5]+np.clip(self.goal-ref[:5],-.08,.08)
                # No cancellation of a committed ordinary stroke. Its
                # reference, if still leased, supersedes the supplied base.
                last=candidate
        result=super().step(t,eye,x,q,ref,last)
        if last is not base and np.array_equal(result,last):
            self.note["moves"]+=1
        return result


def plan(out):
    p=_read(MODEL)
    a=_read(STOCK)
    c=a["cards"][:N]
    if len({v["id"] for v in c})!=N or {r["id"] for r in p["rows"]}&{v["id"] for v in c}:
        raise ValueError("Posture training and screen roster overlap")
    _write(out,dict(cards=c,cap=CAP,arms=["stock","hub","ports"],
                    model_sha256=_hash(MODEL),stock_sha256=_hash(STOCK),carry_sha256=_hash(season.CARRY),
                    code_sha256=_hash(Path(__file__)), risk_sha256=_hash(fresh.RISK),
                    docket="First six distinct standard season evaluation streams, known independently seeded and fixed before any learned action. At most 12 alternating scored strokes only to bound the short diagnostic. Clamp, stock opening home, arms, cap mechanics, seed per leg, and ordinary incoming camera MPC are unchanged. The new policy can change motion only after two fresh outgoing images confirmed an earlier stroke from the learner."))


def baseline_truncated(row,cap):
    # The stock point starts with the recording's own landing, excluded by
    # the exact regulator. A complete archival stock tape authorizes the
    # same shorter evaluation cap without running an altered baseline.
    if row["score"]["returns"]<=cap:
        return row
    last=[e for e in row["events"] if e["kind"]=="legal_return"][cap]
    stop=int(last["step"])
    tape=row["actions"][:stop]
    events=[e for e in row["events"] if e["step"]<=stop]
    count=audit(events,opening_side=1)
    if count["returns"]!=cap:
        raise ValueError("Could not reconstruct the short stock endpoint")
    return dict(row,actions=tape,events=events,score=count)


def compare(control,row):
    d=next((j for j,(u,v) in enumerate(zip(control["actions"],row["actions"])) if u!=v),
           None if len(control["actions"])==len(row["actions"]) else min(len(control["actions"]),len(row["actions"])))
    if d is None and digest_json(control["events"])!=digest_json(row["events"]):
        raise ValueError("Same controls changed physical events")
    if d is not None and digest_json([e for e in control["events"] if e["step"]<=d])!=digest_json(
                                  [e for e in row["events"] if e["step"]<=d]):
        raise ValueError("Changed plant before the first issued difference")
    reference_n=control["score"]["opponent"]
    matched=[]
    for u,v in zip(control["placements"]["delivered"][:reference_n],row["placements"]["delivered"]):
        if all(same_trace(u[k],v[k]) for k in ("far_contact","near_bounce")):
            height=projected_height(u["receive_line"]) if u["receive_line"] is not None else None
            known=u["near_landed"]
            matched.append(dict(leg=u["leg"],height=height,stock=known,learned=v["near_landed"],
                                x=None if u["receive_line"] is None else u["receive_line"]["xyz"][0]))
    return dict(first_action_diff=d,matched=matched)


def run(out,protocol):
    rule,p,stock=map(_read,(protocol,MODEL,STOCK))
    for k,source in dict(model_sha256=MODEL,stock_sha256=STOCK,carry_sha256=season.CARRY,
                         code_sha256=Path(__file__),risk_sha256=fresh.RISK).items():
        if rule[k]!=_hash(source):
            raise ValueError("Changed receiving posture protocol")
    result=dict(protocol_sha256=_hash(protocol),rows=[],complete=False,archive_verified_first=False)
    if out.exists():
        result=_read(out)
        if result["protocol_sha256"]!=_hash(protocol) or not result["archive_verified_first"]:
            raise ValueError("Cannot resume a changed posture test")
    ready=stock["ready"]
    carry=_read(season.CARRY)
    with np.load(mpc.BANK) as bank:
        raw=mpc.DeepField(bank,"local")
    with mpc.centered(ready),enabled():
        a=mpc.arena()
        camera,rig,stride,lo,hi=mpc.resources(a)
        cad=mpc.arena(detail="cad",width=64,height=48)
        guard=PoseClearance(cad)
        cards=rule["cards"]
        with patch.object(mpc,"Fragments",lambda b,e: Carried(b,e,practice=[course.PRACTICE],held={c["id"] for c in cards})):
            actor=Recovery(mpc.BANK,raw,variant=stock["variant"],budget=stock["budget"],
                           coverage=fresh.shot.Coverage(_read(fresh.RISK)),postures=p["portfolio"],
                           kind="stock",guard=guard)
        try:
            for i,c in enumerate(cards[len(result["rows"]):],start=len(result["rows"])):
                original=stock["rows"][i]
                control=baseline_truncated(original,rule["cap"])
                arms={}
                modes=rule["arms"] if i==0 else rule["arms"][1:]
                for mode in modes:
                    course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),carry)
                    actor.kind=mode
                    r=mpc.play(a,camera,rig,stride,lo,hi,ready,c,original["shot"],raw,actor,course.OPTIC,"frozen",rule["cap"])
                    if any(v["target"]!=list(draw(c,j,"random",None)[0])
                           for j,v in enumerate(r["placements"]["requested"])):
                        raise ValueError("Changed wide reactive draw")
                    check=compare(control,r)
                    if mode=="stock":
                        if check["first_action_diff"] is not None:
                            raise ValueError("The archived stock conditional cap is inexact")
                        result["archive_verified_first"]=True
                    r.update(posture_routes=actor.passes,**check)
                    if mode!="stock" and any(s["learned"] and not s["stock"] for s in check["matched"]):
                        r["strict_cad"]={k:replay(ready,c,v) for k,v in (("stock",control),(mode,r))}
                    arms[mode]=r
                result["rows"].append(dict(id=c["id"],stock_score=control["score"]["learner"],arms=arms))
                result["complete"]=len(result["rows"])==len(cards)
                _write(out,result)
                print(c["id"],control["score"]["learner"],{k:(r["score"]["learner"],r["first_action_diff"],
                      sum(v["moves"]>0 for v in r["posture_routes"]),[(v["height"],v["stock"],v["learned"])
                      for v in r["matched"] if v["stock"]!=v["learned"]]) for k,r in arms.items()},flush=True)
        finally:
            actor.close();cad.close();camera.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("fit","plan","run"))
    p.add_argument("--model",type=Path,default=MODEL)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    ROOT.mkdir(exist_ok=True,parents=True)
    if a.task=="fit":
        fit(a.model)
    elif a.task=="plan":
        plan(a.plan)
    else:
        run(a.out,a.plan)
