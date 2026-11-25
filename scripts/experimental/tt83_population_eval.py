"""Score population search on sealed physical missions and ordinary continuous points."""
import argparse
from collections import Counter
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt83_population_search as opt
from . import tt59_fewshot_joint_games as games0
from . import tt59_online_corrections as live
from . import tt57_motor_school_transfer as transfer
from .tt37_optical_clinic import OpticalClinic
from .tt39_carry_joints import Carried
from .tt4_transfer_trial import _hash, _read, _write

school, mpc, course, season = opt.school,opt.mpc,games0.course,games0.season
ROOT=Path("artifacts/tt83_population_search")
PLAN,HELD,GAMES=(ROOT/c for c in ("protocol.json","forks.json.gz","games.json.gz"))
MODES=("static","iterative")
N_GAMES=6
SEED=830957
OTHER=tuple(Path("artifacts")/(x+".json.gz") for x in (
    "tt42/after12","tt42/after36","tt42/diverse-evaluation","tt44_combined/combined",
    "tt48_transfer/continued-64","tt52_practice_dose/live-from-12",
))
PRIOR=opt.sensitivity.OUT


def source(path):
    d=_read(path)
    if path==PRIOR:
        return dict(ready=_read(opt.sensitivity.PLAN)["ready"], cards=[r["card"] for r in d["rows"]],
                    rows=[r["arms"]["stock"] for r in d["rows"]])
    return d


def plan(feeds):
    rng=np.random.default_rng(SEED)
    d0=_read(PRIOR)
    if not d0["complete"]:
        raise ValueError("The predecessor diagnostic course must be complete")
    fit=_read(opt.sensitivity.MODEL)
    used={r["id"] for r in fit["observations"] if "entry" in r}
    old=_read(school.PLAN)
    used.update(r["id"] for deck in ("practice","held") for r in old[deck])
    seen={r["card"]["id"] for r in d0["rows"]}
    source_paths=(*OTHER,PRIOR)
    docs={str(p):source(p) for p in source_paths}
    cases=[]
    for i,r in enumerate(d0["rows"]):
        c=r["card"]; stock=r["arms"]["stock"]; terminal=(stock["placements"]["delivered"] or [{}])[-1]
        # These known TT74 vetoes are mechanism cases, not a new cohort.
        if terminal.get("near_landed",True) or not all(terminal.get(k) for k in ("far_contact","near_bounce","receive_line")):
            continue
        opts=[x for x in r["arms"]["conditional"]["repairs"]
              if x.get("reason")=="projection-not-robust" and x["tick"]>terminal["far_contact"]["step"]]
        if not opts:
            continue
        for n in opts[:1]:
            if terminal.get("near_contact") and terminal["near_contact"]["step"]<n["tick"]+7:
                continue
            cases.append(dict(source=str(PRIOR),id=c["id"],i=i,tick=n["tick"],leg=terminal["leg"],
                              roster="prior-veto",was_legal=False))
    terminal=[]
    positive=[]
    for path in OTHER:
        d=docs[str(path)]
        for i,(c,r) in enumerate(zip(d["cards"],d["rows"])):
            if c["id"] in used|seen or c.get("opening")!="random" or c.get("profile")!="reach":
                continue
            if any(u.get("draw")!="random" for u in r["placements"]["requested"]):
                continue
            for o in r["placements"]["delivered"]:
                if not all(o.get(k) for k in ("far_contact","near_bounce","receive_line")):
                    continue
                num=o["leg"]
                h=school.projected_height(o["receive_line"])
                x=o["receive_line"]["xyz"][0]
                if abs(x)>.24 or not .958<h<=1.15:
                    continue
                if o["near_landed"] and not (h>=1.056 and num+1<len(r["placements"]["delivered"])):
                    continue
                end=(o.get("near_contact") or {}).get("step",o["near_bounce"]["step"]+31)
                window=[a for a in r["searches"] if a.get("p") and -.995<a["p"][1]<-.53 and
                        o["far_contact"]["step"]+3<a["tick"]<end-8]
                if not window:
                    continue
                root=window[0]; lane=int(np.searchsorted([-.10,.10],x))
                if not o["near_landed"] and num!=len(r["placements"]["delivered"])-1:
                    continue
                group=positive if o["near_landed"] else terminal
                group.append(dict(source=str(path),id=c["id"],i=i,tick=root["tick"],leg=num,
                                  roster="routine" if o["near_landed"] else "independent-miss",
                                  was_legal=bool(o["near_landed"]),height=float(h),lane=lane))
    used_hold={r["id"] for r in cases}
    for available,count in ((terminal,7),(positive,3)):
        shuffled=[available[j] for j in rng.permutation(len(available))]
        for lane in ([1,0,2]*4):
            options=[r for r in shuffled if r["id"] not in used_hold and r["lane"]==lane]
            if not options:
                continue
            c=min(options,key=lambda x:(x["tick"]//600,abs(x["height"]-1.028)))
            cases.append(c);used_hold.add(c["id"]);count-=1
            if count<=0:
                break
    if sum(r["roster"]=="independent-miss" for r in cases)<5:
        raise ValueError("Insufficient additional held native losses")
    # Distinct measured releases from the two earlier matched-fork
    # families, with fresh standard opponent seeds and no treated roster.
    carry=_read(season.CARRY)
    banned=used|seen|used_hold|{c["id"] for c in carry["cards"][:len(carry["rows"])]}
    for p in (mpc.BANK,course.PRACTICE):
        banned.update(r["card"]["id"] for r in _read(p.with_suffix(".json"))["practice"])
    archive={}
    for path in OTHER:
        d=docs[str(path)]
        archive.update({c["id"]:c for c,r in zip(d["cards"],d["rows"]) if r["score"]["learner"]>0})
    pool=_read(course.SOURCE)
    available={c["id"]:c for deck in ("practice","assessment") for c in pool[deck]
               if c["id"] not in banned and c["opening"]=="random" and c["profile"]=="reach"}
    order=sorted(rng.permutation(sorted(available)),key=lambda k:k not in archive)
    if len(order)<N_GAMES:
        raise ValueError("Insufficient separate normal-start game releases")
    shots=mpc.measured_launches(feeds)
    roster=[dict(card=dict(available[k],seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30))),
                 shot=shots[k],exploratory_seed=int(rng.integers(1,2**30))) for k in order[:N_GAMES]]
    ROOT.mkdir(exist_ok=True,parents=True)
    fixed=(*source_paths,opt.sensitivity.PLAN,opt.sensitivity.MODEL,opt.few.MODEL,school.PLAN,
           mpc.BANK,course.OPTIC,season.CARRY,season.CARRY.with_suffix("").with_suffix(".npz"),season.fresh.RISK,course.PRACTICE)
    _write(PLAN,dict(schema="tt83-native-optic-population-1",cases=cases,games=roster,
                     ready=_read(mpc.BANK.with_suffix(".json"))["ready"],sources={str(p):_hash(p) for p in fixed},
                     rounds=opt.POP,permissions="Actor options are built only from known-model simulation initialized with RGB/encoders, not an exact private held state. Actual held rewards are not used to select candidates. Prior-veto cases are explicit mechanism tests. Separate independent archive losses and routine positive controls and distinct normal-start continuous games are frozen before native trials. Equal-twin-budget static and iterative both have the identical first batch plus the same finalized public perturbation checks; no real retries."))
    print("TT83",Counter(c["roster"] for c in cases),[x["card"]["id"] for x in roster],flush=True)


def read_plan():
    p=_read(PLAN)
    for name,h in p["sources"].items():
        f=Path(name)
        if not f.exists() and f.parent.name=="tt52_practice_dose" and f.name.endswith("-96.json.gz"):
            f=f.with_name(f.name.replace("-96.json.gz",".json.gz"))
        if _hash(f)!=h:
            raise ValueError(f"Frozen TT83 input changed: {name}")
    return p


def factory(raw,lim,bank,prior,mode,held,reload=False):
    rank=season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with patch.object(mpc,"Fragments",lambda b,e:Carried(b,e,practice=[course.PRACTICE],held=held)):
        actor=opt.Player(mpc.BANK,raw,population=opt.Population(prior,lim[1],lim[2],lim[0]),
                         bank=bank,mode=mode,coverage=rank,variant="elastic",budget=28)
    if reload:
        cp=_read(season.CARRY)
        course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),cp)
    return actor


def forks(n):
    p=read_plan()
    saved=dict(plan=_hash(PLAN),rows=[],complete=False)
    if HELD.exists():
        saved=_read(HELD)
        if saved["plan"]!=_hash(PLAN):
            raise ValueError("Changed physical fork order")
    prior=opt.MotorPrior()
    bank=opt.sensitivity.Bank(_read(opt.sensitivity.MODEL))
    docs={}
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,"local")
    with school.enabled():
        cad=mpc.arena(width=64,height=48,detail="cad")
        guard=school.CourtClearance(cad.model,articulated=True)
        try:
            for case in p["cases"][len(saved["rows"]):n]:
                start=process_time()
                s=case["source"]
                if s not in docs:
                    docs[s]=source(Path(s))
                doc=docs[s]; row=doc["rows"][case["i"]]; card=doc["cards"][case["i"]]
                tick=case["tick"]; ready=np.asarray(doc["ready"])
                with mpc.centered(ready):
                    a=mpc.arena()
                    cam,rig,stride,lo,hi=mpc.resources(a)
                    actor=factory(raw,(stride,lo,hi),bank,prior,"stock",set())
                    try:
                        actor.begin()
                        eye,x,q,ref,_,feature,hist=school.observed_prefix(a,cam,rig,card,row,tick,a._arm_ids[1][2])
                        snap=school.State(a)
                        stock=school.physical(a,snap,card,row,case,row["actions"][tick:])
                        if stock["landed"]!=case["was_legal"] or stock.get("unmatched"):
                            raise ValueError("Archived ordinary scoring control changed")
                        if not case["was_legal"] and stock["event_sha256"]!=school.digest_json(row["events"]):
                            raise ValueError("Exact stock terminal miss differs")
                        outcomes={}
                        public=bool(x is not None and feature is not None and x[4]<-1.)
                        stock_attempt=None
                        if public:
                            transfer.warm_clinic(actor.engine,rig,a.control_dt,hist,eye,tick,q,ref)
                            j, conf=raw.get(x,q)
                            base=ready if j is None else ready+conf*(j-ready)
                            actor._capture={}
                            try:
                                school.season.Player.search(actor,tick,x,q,ref,base)
                            finally:
                                caps=list(actor._capture.values())
                                actor._capture=None
                            current=[v[1]-ref for v in sorted(caps,key=lambda a:a[0])[:5]]
                            stock_attempt=(actor.notes[-1] if actor.notes else None)
                            if actor.engine.sync():
                                roots=opt.stock_anchors(raw,actor.engine,ready,x,q,ref,base,current,bank=bank)
                                for mode in MODES:
                                    action,_,diag=actor.population.search(actor.engine,actor.geometry,roots,ref,
                                                            np.random.default_rng([SEED,case["id"],tick]),mode)
                                    # Protect the same public perturbation-robust
                                    # candidates as the live actor. The routine
                                    # roster tag never participates in selection.
                                    protected=bool(stock_attempt and stock_attempt.get("robust"))
                                    issued=action is not None and not protected
                                    fit=stock
                                    if issued:
                                        commands=np.vstack((action,np.zeros((school.T+88-opt.H,6))))
                                        fit=school.physical(a,snap,card,row,case,commands)
                                        if fit["landed"]:
                                            audit=school.physical(a,snap,card,row,case,commands,guard)
                                            if audit["event_sha256"]!=fit["event_sha256"]:
                                                raise ValueError("Population candidate CAD replay changed physics")
                                            fit["clearance"]=audit["clearance"]
                                    outcomes[mode]=dict(search=diag,issued=issued,physical=fit,protected=protected)
                        saved["rows"].append(dict(case=case,public=public,stock=stock,stock_public=stock_attempt,
                                                   arms=outcomes,cpu_s=process_time()-start))
                        saved["complete"]=len(saved["rows"])==len(p["cases"])
                        _write(HELD,saved)
                        print(case["id"],case["roster"],[(m,a0["search"]["legal"],a0["search"]["audit"]["robust"],
                              a0["issued"],a0["physical"]["landed"],a0["physical"]["answered"])
                              for m,a0 in outcomes.items()],flush=True)
                    finally:
                        actor.close();cam.close();a.close()
        finally:
            cad.close()


def games(n):
    p=read_plan()
    prior=opt.MotorPrior()
    bank=opt.sensitivity.Bank(_read(opt.sensitivity.MODEL))
    saved=dict(plan=_hash(PLAN),rows=[],complete=False)
    if GAMES.exists():
        saved=_read(GAMES)
        if saved["plan"]!=_hash(PLAN):
            raise ValueError("Changed normal-start whole-point protocol")
    ready=np.asarray(p["ready"]); ids={r["card"]["id"] for r in p["games"]}
    with np.load(mpc.BANK) as f:raw=mpc.DeepField(f,"local")
    with mpc.centered(ready),school.enabled():
        a=mpc.arena()
        cam,rig,stride,lo,hi=mpc.resources(a)
        lim=(stride,lo,hi)
        actors={}
        try:
            for mode in ("stock","iterative"):
                actors[mode]=factory(raw,lim,bank,prior,mode,ids,reload=True)
            for item in p["games"][len(saved["rows"]):n]:
                c=item["card"]; outputs={}
                for mode,actor in actors.items():
                    started=process_time()
                    row=live.play(a,cam,rig,lim,ready,c,item["shot"],raw,actor,item["exploratory_seed"])
                    row.update(expanded=actor.expanded,ordinary_kept=actor.bookkeeping,cpu_s=process_time()-started)
                    if mode!="stock":
                        row["same_offer_changes"]=games0.compare(outputs["stock"],row)
                    outputs[mode]=row
                    print(c["id"],mode,row["score"]["learner"],len(row["placements"]["delivered"]),
                          len(row["expanded"]),sum(r["issued"] for r in row["expanded"]),round(row["cpu_s"],1),flush=True)
                saved["rows"].append(dict(item,arms=outputs))
                saved["complete"]=len(saved["rows"])==len(p["games"])
                _write(GAMES,saved)
        finally:
            for a0 in actors.values():
                a0.close()
            cam.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","forks","games"))
    p.add_argument("--n",type=int)
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    args=p.parse_args()
    if args.mode=="plan":
        plan(args.feeds)
    elif args.mode=="forks":
        forks(args.n or 100)
    else:
        games(args.n or N_GAMES)
