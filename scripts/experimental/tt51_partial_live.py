"""Compare reversible preparation in complete broad TT37 points, with camera-earned skill updates."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt51_partial_finish as pf
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw

WHOLE=pf.root/"whole-plan.json"
OUT=pf.root/"whole.json.gz"


def plan():
    prior=pf.prior
    reg=_read(pf.plan_path)
    motor_ids={r["id"] for r in reg["rows"]}
    groups=[(Path(prior.course.PLAN),"practice"),(Path(prior.season.PLAN),"practice"),
            (Path(prior.fresh.PLAN),"assessment")]
    blocked={c["id"] for p,k in groups for c in _read(p)[k]}
    rng=np.random.default_rng(5123045)
    src=_read(prior.course.SOURCE)
    pool=sorted({c["id"]:c for c in src["practice"]+src["assessment"]
                 if c["id"] not in blocked|motor_ids and c["opening"]=="random" and c["profile"]=="reach"}.values(),
                 key=lambda r:r["id"])
    roster=[dict(pool[i],seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30)))
            for i in rng.permutation(len(pool))[:8]]
    if len(roster)<4:raise ValueError("Need disjoint measured openings")
    _write(WHOLE,dict(schema="tt51-whole-points-1",cards=roster,prep_plan_sha256=_hash(pf.plan_path),
                      pool=len(pool),rule="Disjoint release IDs and independently assigned TT37 targets. Each arm starts from identical motor memory."))


class Tap:
    def __init__(self, camera):
        self.camera=camera
        self.frame=None

    def capture(self):
        self.frame=self.camera.capture()
        return self.frame


def camera_motion(steps):
    # Native scoring is only a scalar reward. The action, departure and body
    # labels below are fitted from the footage already available to the arm.
    seen=[(t,x,q,ref,sent) for t,x,q,ref,sent in steps if x is not None]
    for j in range(len(seen)-1,0,-1):
        t,x,q,_,_=seen[j]
        t0,x0,*_=seen[j-1]
        if not (x[4]>.9 and x0[4]<-.9 and t-t0<=8 and x0[1]<-.94 and x[1]<-.89):continue
        depth=float((x0[1]+x[1])/2)
        phase=t-1
        sequences=[r for r in steps if phase-30<=r[0]<=phase+15]
        labels=[]
        for k,(y,v,s,ref,_) in enumerate(seen):
            if not(-31<y-phase<-7 and -.92<v[1]<-.33 and v[4]<-.9 and v[5]>.04) or k<2:continue
            old,_,qs,*_=seen[k-2]
            dt=(-1.1-v[1])/v[4]
            labels.append(dict(phase=int(y-phase),future=[v[0]+dt*v[3],v[2]+dt*v[5]-4.905*dt**2,
                          v[3],v[4],v[5]-9.81*dt],q=s[:5].tolist(),dq=((s[:5]-qs[:5])/((y-old)*pf.mpc.DT)).tolist(),
                          lag=(ref[:5]-s[:5]).tolist()))
        if len(labels)>=3 and len(sequences)>10:
            return dict(samples=labels,phases=[y-phase for y,_,_,_,_ in sequences],
                        refs=[s.tolist() for _,_,_,_,s in sequences],contact_y=depth)
    return None


class Tail(pf.prior.Carried):
    def keep(self, rows, tag):
        keep=super().keep(rows,tag)
        actor=self.owner
        if actor.mode=="partial-finish" and keep and actor.native_readiness_ticks:
            sample=camera_motion(rows)
            if sample is not None:
                self.serial_skills+=1
                actor.finish.moves.append((f"earned-{self.serial_skills}",sample))
                if len(actor.finish.moves)>self.original_skills+8:
                    actor.finish.moves.pop(self.original_skills)
        return keep


class Arm(pf.Player):
    def step(self,tick,eye,x,q,reference,base):
        return self.choose(tick,self.tap.frame if tick%2==0 else None,eye,x,q,reference,base)


def run(path,n,predict_path=None):
    plan=_read(WHOLE)
    if plan["prep_plan_sha256"]!=_hash(pf.plan_path):raise ValueError("Changed shooting roster")
    prior=pf.prior; mpc=pf.mpc
    old=_read(prior.ARCHIVE); school=_read(pf.school_path)
    if not school.get("complete"):raise ValueError("Need complete physical school")
    cards=plan["cards"][:n]
    shots=mpc.measured_launches(Path("/tmp/tt25-rallies.json"))
    doc=dict(schema="tt51-complete-point-shootout-1",plan_sha256=_hash(WHOLE),school_sha256=_hash(pf.school_path),
             cards=cards,controller_sha256=_hash(Path(pf.__file__)),
             predictor_sha256=None if predict_path is None else _hash(predict_path),rows=[])
    base={**old,"rows":[None]*len(_read(prior.season.CARRY)["rows"])+old["rows"]}
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    with mpc.centered(old["ready"]):
        a=mpc.arena();cad=mpc.arena(detail="cad")
        camera,rig,stride,lo,hi=mpc.resources(a); tap=Tap(camera)
        checker=prior.Reach(cad,old["ready"],old["ready"])
        try:
            for c in cards:
                row={"id":c["id"],"runs":{}}
                for mode in ("stock","partial-finish"):
                    predict=None
                    if predict_path is not None and mode!="stock":
                        from .tt51_flight_risk import RiskForecast
                        predict=RiskForecast(predict_path)
                    with patch.object(mpc,"Fragments",lambda f,e:Tail(f,e,practice=[prior.course.PRACTICE],held={z["id"] for z in cards})):
                        p=Arm(mpc.BANK,raw,mode=mode,finish=pf.Finish(school),checker=checker,predictor=predict,
                              coverage=prior.fresh.shot.Coverage(_read(prior.fresh.RISK)),budget=old["budget"],variant=old["variant"])
                    p.tap=tap;p.priors.owner=p;p.priors.original_skills=len(p.finish.moves);p.priors.serial_skills=0
                    try:
                        prior.course.resume(p,prior.BANK,base)
                        r=mpc.play(a,tap,rig,stride,lo,hi,old["ready"],c,shots[c["id"]],raw,p,prior.course.OPTIC,"live",240)
                        for k,req in enumerate(r["placements"]["requested"]):
                            if req["target"]!=list(draw(c,k,"random",None)[0]):
                                raise ValueError("Changed the fresh broad opponent")
                        row["runs"][mode]=r|dict(readiness_ticks=p.native_readiness_ticks,followthrough=p.pick,
                                                optical_updates=p.priors.serial_skills)
                    finally:
                        p.close()
                if len({v["memory"]["before"] for v in row["runs"].values()})!=1:
                    raise ValueError("Opening motor memory differed between variants")
                doc["rows"].append(row);_write(path,doc)
                print(c["id"],[(k,v["score"]["learner"],v["readiness_ticks"],v["optical_updates"])
                      for k,v in row["runs"].items()],flush=True)
        finally:
            a.close();camera.close();cad.close()
    doc["complete"]=True;_write(path,doc)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("plan","run"))
    p.add_argument("--n",type=int,default=4)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--predictor",type=Path)
    args=p.parse_args()
    plan() if args.phase=="plan" else run(args.out,args.n,args.predictor)
