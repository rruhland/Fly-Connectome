"""Switch direct motor actions using matched native outcomes, not twin legality."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt57_motor_school as school
from . import tt59_fewshot_joints as repair
from . import tt74_sensitivity_bank as sensitivity
from .tt37_optical_clinic import OpticalClinic
from .tt4_transfer_trial import _hash, _read, _write

mpc,course,season=school.mpc,school.course,school.season
ROOT=Path("artifacts/tt78_empirical_switch")
PLAN,DATA,MODEL,REPORT=(ROOT/n for n in ("protocol.json","forks.json.gz","switch.json","report.json"))
SOURCES=(season.ROOT/"learning.json.gz",season.ROOT/"after128.json.gz",
         Path("artifacts/tt48_transfer/continued-64.json.gz"),Path("artifacts/tt44_combined/combined.json.gz"),
         Path("artifacts/tt42/diverse-evaluation.json.gz"),Path("artifacts/tt42/diverse-learning.json.gz"),
         Path("artifacts/tt52_practice_dose/live-from-12.json.gz"),Path("artifacts/tt52_practice_dose/live-from-48-96.json.gz"),
         Path("artifacts/tt53_responsive_school/ordinary-live.json.gz"))
SEED=780167
H=school.H
COUNTS=(34,18)
MODES=("sensitivity","paired-repair")
FEATURES=("side","projected_height","inbound_speed","stock_legal_count","stock_robust",
          "stock_margin","sensitivity","distance","estimated_touch","estimated_radius",
          "estimated_forward","estimated_legal")
CENTER, SCALE = np.array([.14,1.027,3.4,.7,.0,.061,.5,2.0,.5,.065,2.0,0.]), np.array([.13,.058,.9,2.,1.,.07,.5,1.3,.5,.05,2.,1.])
THRESHOLD=.12
ETA=-1.51


def gate(x):
    return (x is not None and np.isfinite(x).all() and -.995<x[1]<-.78
            and x[4]<-1.15 and x[5]>.15)


def archive_rows(path, d, banned):
    for i,(c,r) in enumerate(zip(d["cards"],d["rows"])):
        if c["id"] in banned or c["opening"]!="random" or c["profile"]!="reach":
            continue
        delivery=r["placements"]["delivered"]
        choices=r["searches"]
        if not delivery or any(p["draw"]!="random" for p in r["placements"]["requested"]):
            continue
        for leg,o in enumerate(delivery):
            if not o.get("far_contact") or not o.get("near_bounce"):
                continue
            lo, hi=o["far_contact"]["step"]+4, o["far_contact"]["step"]+80
            if lo>660:
                continue
            if leg+1<len(delivery) and delivery[leg+1].get("far_contact"):
                hi=min(hi,delivery[leg+1]["far_contact"]["step"]-18)
            if o.get("near_contact"):
                hi=min(hi,o["near_contact"]["step"]-6)
            possible=[n for n in choices if n.get("bounced") and n.get("p") and lo<n["tick"]<hi
                      and -.995<n["p"][1]<-.78]
            if not possible:
                continue
            note=possible[0]
            # The accepted stock policy can be vulnerable even when public legality
            # is nonzero. Keep some positive controls to measure that harm.
            lane=int(np.searchsorted([-.08,.08],note["p"][0]))
            weak=not note.get("legal_trials",0) and not note.get("robust",0)
            yield dict(id=c["id"],card=c,source=str(path),i=i,tick=note["tick"],leg=leg,
                       lane=lane,weak=weak,old_outcome=bool(o["near_landed"]),note=note)


def freeze():
    sources={p:_read(p) for p in SOURCES}
    active=_read(sensitivity.PLAN)
    banned={q["card"]["id"] for q in active["releases"]}|{q["id"] for q in _read(sensitivity.MODEL)["observations"]}
    banned|={q["id"] for q in _read(school.PLAN)["held"]}
    # The correction source cannot be a calibration or test fork.
    with np.load(repair.MODEL) as z:
        banned|=set(map(int,z["ids"]))
    ready=_read(mpc.BANK.with_suffix(".json"))["ready"]
    pool=[q for p,d in sources.items() if d["ready"]==ready for q in archive_rows(p,d,banned)]
    rng=np.random.default_rng(SEED)
    pool=[pool[k] for k in rng.permutation(len(pool))]
    ids=set()
    picks={}
    # Hold qualification only on public readiness and lane, never on the
    # outcome of the archived stock shot or on any candidate's label.
    for stage,n in (("held",COUNTS[1]),("train",COUNTS[0])):
        got=[]
        for step in range(n):
            target_lane=step%3
            weak=step%5!=4
            eligible=[r for r in pool if r["id"] not in ids and r["weak"]==weak and r["lane"]==target_lane]
            if not eligible:
                eligible=[r for r in pool if r["id"] not in ids and r["weak"]==weak]
            if not eligible:
                eligible=[r for r in pool if r["id"] not in ids]
            if not eligible:
                break
            # Bound past replay's cost without using hit or bounce geometry.
            row=min(eligible,key=lambda q:q["tick"]//220)
            row=dict(row,randomization=int(rng.integers(2**30)))
            got.append(row);ids.add(row["id"])
        picks[stage]=got
    if len(picks["train"])<20 or len(picks["held"])!=COUNTS[1]:
        raise ValueError(f"Insufficient independent physical decision forks {len(pool)} {[len(x) for x in picks.values()]}")
    files=(*SOURCES,school.PLAN,repair.PLAN,repair.MODEL,sensitivity.PLAN,sensitivity.MODEL,course.OPTIC,
           mpc.BANK,season.CARRY)
    ROOT.mkdir(parents=True,exist_ok=True)
    _write(PLAN,dict(schema="tt78-paired-reward-switch-1", sources={str(v):_hash(v) for v in files},
        ready=ready, assignments=picks,eligible_rows=len(pool),banned=sorted(banned),threshold=THRESHOLD,
        candidate_families=MODES, minimum_held_improvements=2,
        role="Release-disjoint ordinary attained centered RGB forks. Stratify by public stock rather than observed stock success. Compare the exact archived ordinary physical action against one randomized nominated movement per centered source family at the same received shot. No nominal-legal requirement. Electrical command limits and exact physically earned score remain unchanged. Randomize the order in the matched plant from the same native snapshot. The only fit targets are safe legal native return indicators. Held motor trials start only after the switch is frozen."))
    print("TT78",len(pool),[(name,len(rows),sum(r["weak"] for r in rows)) for name,rows in picks.items()],flush=True)


def design():
    p=_read(PLAN)
    if any(_hash(Path(k))!=v for k,v in p["sources"].items()):
        raise ValueError("Frozen data or motor source changed")
    return p


def main_features(x, q, ref, stock, mode, n, y=None, fb=None):
    lateral, z, tau=sensitivity.local.projected(x,ETA)
    return np.r_[
        min(.6,abs(lateral)), np.clip(z,.76,1.24), np.clip(-x[4],1.5,6.),
        np.clip(stock.get("legal_trials",0),0,4), stock.get("robust",0),
        np.clip(stock.get("best_separation_m",.2),0,.24), mode=="sensitivity",
        np.clip(n["distance"],0,8), bool(fb is not None and fb[1]>.5),
        .124 if fb is None or fb[1]<=.5 else np.clip(fb[8],0,.14),
        0. if fb is None or fb[1]<=.5 else np.clip(fb[3],-4,6),
        bool(y is not None and np.isfinite(y[0]))].astype(float)


class Candidates:
    def __init__(self):
        self.sensitivity=sensitivity.Bank(_read(sensitivity.MODEL))
        self.repair=repair.Corrections()

    def suggest(self, engine, raw, home, x, q, ref, base, stock, rng):
        if not gate(x) or not engine.sync():
            return []
        first,notes=self.sensitivity.suggest(x,q,ref)
        bank_mode=[]
        if len(first):
            bank_mode.append(("sensitivity",np.asarray(first[:4]),notes[:4]))
        k=school.context(x,q,ref,engine.rate,engine.bounced)
        anchors=repair.anchor_paths(engine,raw,home,x,q,ref,base)
        paths,details=self.repair.proposals(x,k,anchors,"paired")
        bank_mode.append(("paired-repair",paths,details))
        picked=[]
        for mode,paths,info in bank_mode:
            rates,y,fb,_,legal=school.native(engine.predict,paths,ref)
            # Soft contact feasibility only, never a legal/robust acceptance
            # rule. Estimated-legal and estimated-illegal remain eligible.
            distance=np.array([q["distance"] for q in info])
            near=np.clip(fb[:,0],0.,1.)+np.where(fb[:,1]>.5,0.,.09)+.013*distance
            ix=np.argsort(near,kind="stable")
            menu=[]
            for j in ix:
                if all(np.sqrt(np.mean((rates[j,:H]-rates[k,:H])**2))>.015 for k in menu):
                    menu.append(int(j))
                if len(menu)==min(len(ix),4):
                    break
            j=int(rng.choice(menu)); n=info[j]
            predictor=main_features(x,q,ref,stock,mode,n,y[j],fb[j])
            picked.append(dict(mode=mode,feature=predictor.tolist(),candidate=j,metadata=n,
                     menu=menu,probability=1/len(menu),nominal_legal=bool(legal[j]),feedback=fb[j].tolist(),
                     rates=rates[j,:H].copy()))
        return picked


def body_source(p):
    docs={}
    def doc(spec):
        path=spec["source"]
        if path not in docs:
            docs[path]=_read(Path(path))
        return docs[path]["rows"][spec["i"]]
    return doc


def run(stage, limit=None):
    p=design()
    path=DATA if stage=="train" else ROOT/"held-forks.json.gz"
    receipt=dict(plan=_hash(PLAN),mode=stage,rows=[],complete=False)
    if stage=="held":
        model=_read(MODEL)
        if not model["eligible"]:
            _write(path,dict(receipt,complete=True,status="Insufficient positive empirical matched training outcomes. Held motor trials unopened."))
            return
        receipt["model"]=_hash(MODEL)
    old=_read(path) if path.exists() else receipt
    if any(old.get(k)!=v for k,v in receipt.items() if k not in ("rows","complete")):
        raise ValueError("Changed paired data split or trained switch")
    if old["complete"]:
        return
    home=np.asarray(p["ready"]); archive=body_source(p); generator=Candidates()
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    with mpc.centered(home),school.enabled():
        a=mpc.arena()
        cad=mpc.arena(detail="cad",width=64,height=48)
        guard=school.CourtClearance(cad.model,articulated=True)
        cam,rig,*_=mpc.resources(a)
        clinic=OpticalClinic(body=True)
        try:
            for spec in p["assignments"][stage][len(old["rows"]):limit]:
                row=archive(spec); c=spec["card"]; tick=spec["tick"]
                a.sample_clearance(False)
                cli=school.observed_prefix(a,cam,rig,c,row,tick,a._arm_ids[1][2])
                eye,x,q,ref,_,_,history=cli
                if not gate(x):
                    old["rows"].append(dict(id=spec["id"],status="no optical exposure reproduced"));_write(path,old);continue
                repair.transfer.warm_clinic(clinic,rig,a.control_dt,history,eye,tick,q,ref)
                position,confidence=raw.get(x,q)
                base=home.copy() if position is None else home+confidence*(position-home)
                snapshot=school.State(a)
                rng=np.random.default_rng(spec["randomization"])
                offers=generator.suggest(clinic,raw,home,x,q,ref,base,spec["note"],rng)
                mx=dict(spec=spec,stock=None,offers=[])
                arms=[(-1,row["actions"][tick:])]
                arms.extend((i,np.vstack([r["rates"],np.zeros((125-H,6))])) for i,r in enumerate(offers))
                # Both choices are physically evaluated from the same current
                # snapshot, irrespective of arm execution order and outcome.
                for i in rng.permutation(len(arms)):
                    j,tape=arms[i]
                    out=school.physical(a,snapshot,c,row,spec,tape)
                    if j<0:
                        if out.get("unmatched") or bool(out["landed"])!=spec["old_outcome"]:
                            raise ValueError("Uncontrolled stock change at assigned paired fork")
                        mx["stock"]=out
                        continue
                    candidate=offers[j]
                    command=candidate.pop("rates")
                    out["clearance"]=None
                    if out["landed"]:
                        check=school.physical(a,snapshot,c,row,spec,tape,guard)
                        if out["event_sha256"]!=check["event_sha256"]:
                            raise ValueError("Candidate CAD replay changed physics")
                        out["clearance"]=check["clearance"]
                    reward=bool(out["landed"] and not out["clearance"]["penetration_samples"])
                    mx["offers"].append(dict(candidate,reward=reward,requested_command=command.tolist(),outcome=out))
                mx["offers"].sort(key=lambda v:v["mode"])
                old["rows"].append(mx)
                print(stage,spec["id"],bool(mx["stock"]["landed"]),[(v["mode"],v["nominal_legal"],v["reward"])
                      for v in mx["offers"]],flush=True)
                _write(path,old)
        finally:
            clinic.twin.close()
            cam.close()
            cad.close()
            a.close()
    old["complete"]=len(old["rows"])==len(p["assignments"][stage]);_write(path,old)


def data(rows):
    out=[]
    for i,row in enumerate(rows):
        if row.get("stock") is None:
            continue
        for a in row["offers"]:
            out.append((i,a["feature"],int(a["reward"])-int(row["stock"]["landed"])))
    return out


def basis(x):
    z=np.clip((np.asarray(x)-CENTER)/SCALE,-3.,3.)
    return np.r_[1.,z,.45*z[3:6]*z[8],.45*z[8]*z[[1,7,9,10]]]


def l2(data, rows, weight=7.5):
    if not data:
        raise ValueError("No physically randomized data")
    X=np.vstack([basis(x) for _,x,_ in data])
    y=np.array([v for _,_,v in data],dtype=float)
    counts=Counter(i for i,_,_ in data)
    scale=np.array([1./counts[i] for i,_,_ in data])
    S=X.T@(X*scale[:,None])+np.diag(np.r_[1.4,np.full(X.shape[1]-1,weight)])
    return np.linalg.solve(S,X.T@(y*scale))


def choice(mx, weights):
    scores=[float(basis(v["feature"])@weights) for v in mx["offers"]]
    if not scores:
        return None,scores
    top=int(np.argmax(scores))
    return (top if scores[top]>THRESHOLD else None),scores


def stats(rows, weights):
    d=Counter()
    traces=[]
    for row in rows:
        if row.get("stock") is None:
            continue
        stock=bool(row["stock"]["landed"]); options=row["offers"]
        idx,scores=choice(row,weights)
        direct=idx is not None
        reward=bool(options[idx]["reward"]) if direct else stock
        vetoes=[i for i,v in enumerate(options) if v["nominal_legal"]]
        veto=bool(options[max(vetoes,key=lambda j:scores[j])]["reward"]) if vetoes else stock
        no=bool(options[int(np.argmax(scores))]["reward"]) if options else stock
        d.update(seen=1,stock=int(stock),switch=int(reward),intervened=int(direct),
                 raw_max=int(no),twin_veto=int(veto),gain=int(reward and not stock),loss=int(stock and not reward),
                 oracle=int(any(v["reward"] for v in options) and not stock),
                 model_false_negative=sum(v["reward"] and not v["nominal_legal"] for v in options))
        traces.append(dict(id=row["spec"]["id"],scores=scores,pick=idx,labels=[v["reward"] for v in options],stock=stock))
    return dict(d),traces


def fit():
    p=design(); r=_read(DATA)
    if not r["complete"] or r["plan"]!=_hash(PLAN):
        raise ValueError("Finish paired training first")
    train=data(r["rows"]); w=l2(train,r["rows"])
    cross=Counter()
    identities={i for i,_,_ in train}
    for i in identities:
        fit_without=l2([(j,x,y) for j,x,y in train if j!=i],r["rows"])
        d,_=stats([r["rows"][i]],fit_without);cross.update(d)
    observed,details=stats(r["rows"],w)
    useful=cross["gain"]-cross["loss"]>0 and cross["oracle"]>=3 and cross["gain"]>=2
    _write(MODEL,dict(training_sha256=_hash(DATA),weights=w.tolist(),feature_names=FEATURES,threshold=THRESHOLD,
                       cross_fitted=dict(cross),apparent=observed,eligible=useful))
    print("cross-fit",dict(cross),"eligible",useful,flush=True)


def report():
    m=_read(MODEL); h=_read(ROOT/"held-forks.json.gz")
    if not h["complete"]:
        raise ValueError("Incomplete held fork set")
    values,trace=stats(h["rows"],np.asarray(m["weights"]))
    earned=bool(values and values["gain"]>=2 and not values["loss"] and values["switch"]>values["twin_veto"])
    _write(REPORT,dict(model_sha256=_hash(MODEL),holds_sha256=_hash(ROOT/"held-forks.json.gz"),held=values,
                       whole_games_admissible=earned,trace=trace,status=h.get("status")))
    print("held",values,h.get("status"),"games",earned,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","train","fit","held","report"))
    p.add_argument("--n",type=int)
    a=p.parse_args()
    if a.mode in ("train","held"):run(a.mode,a.n)
    else:{"plan":freeze,"fit":fit,"report":report}[a.mode]()
