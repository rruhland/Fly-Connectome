"""Sample independent physical replies from a wider set of unselected actual prior strokes."""
import argparse
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_opponent as live
from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt51_afterstate as model
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt44_paired_credit import State

PLAN=model.ROOT/"native-plan.json"
TAPE=model.ROOT/"native-redraws.json.gz"
EXCLUDED_RESERVES=(
    Path("artifacts/tt44_combined/fresh.json"),
    Path("artifacts/tt50_conditional_prep/plan.json"),
    Path("artifacts/tt51_low_curriculum/plan.json"),
)


def withheld(d):
    ids={c["id"] for c in _read(course.PLAN)["assessment"]}
    ids.update(d["prohibited"])
    ids.update(r["id"] for r in d["rows"])
    # Preserve any top-level assessment or explicitly reserved release-ID list.
    for p in EXCLUDED_RESERVES:
        if not p.exists():
            continue
        doc=_read(p)
        for k in ("assessment","held","reserved","holdout"):
            for t in doc.get(k,()):
                if isinstance(t,dict):
                    ident=t.get("id", (t.get("card") or {}).get("id"))
                else:
                    ident=t
                if isinstance(ident,int):
                    ids.add(ident)
    return ids


def freeze(n):
    first=_read(model.SOURCE)
    excluded=withheld(first)
    files=_read(model.INPUTS)["input_files"]
    rng=np.random.default_rng(511812)
    pool=defaultdict(list)
    fingerprints=defaultdict(set)
    offers={}
    for p,fp in files.items():
        if _hash(Path(p))!=fp:
            raise ValueError("Physical outcome source changed")
        old=_read(Path(p))
        for i,(c,r) in enumerate(zip(old["cards"],old["rows"],strict=True)):
            if c["id"] in excluded or c.get("profile")!="reach" or not r.get("searches"):
                continue
            pp=r["placements"]; req, recv=pp["requested"],pp["delivered"]
            if any(z["draw"]!="random" for z in req):
                continue
            for leg in range(1,min(4,len(req))):
                if leg-1>=len(recv) or not recv[leg-1].get("near_landed"):
                    continue
                if leg>=len(recv) and not any(e["kind"]=="point" for e in r["events"]):
                    continue # cap is a censored label, not a measured failure
                try:
                    dep, shape, cand, context=model.episode(r,leg)
                except (ValueError,StopIteration):
                    continue
                if not np.isfinite(shape).all():
                    continue
                fp_n=digest_json((dep,req[leg]["carriage"]))
                if fp_n in fingerprints[c["id"]]:
                    continue
                fingerprints[c["id"]].add(fp_n)
                source=dict(id=c["id"],source=p,index=i,leg=leg,card=c,
                            anchor=int(np.floor(req[leg]["time_s"]/mpc.DT+1e-7)),
                            native_departure=dep,shape=shape.tolist(),
                            physical_dep=dep["ball"]+dep["velocity"],spin=dep["spin"],
                            incoming_context=context,native_candidate=cand,optical_shape=None)
                pool[c["id"]].append(source)
                offers[p]=fp
    order=rng.permutation(sorted(pool))
    if len(order)<n:
        raise ValueError(f"Need {n} distinct releases, found {len(order)}")
    selected=[pool[i][int(rng.integers(len(pool[i])))] for i in order[:n]]
    for c in selected:
        c["seeds"]=rng.integers(1,2**30,6).tolist()
    _write(PLAN,dict(cases=selected,known_excluded=sorted(excluded),archive_sha256=_hash(model.SOURCE),
                     sources={r["source"]:offers[r["source"]] for r in selected},
                     role="Unselected physically prior actual landed arm actions, up to third reply. One sampled outgoing state per unrelated release. No original next height enters assignment. Original branch is just a verification, never a prediction label."))
    print("selected",len(selected),"releases",[(c["id"],c["leg"]) for c in selected],flush=True)


def physical_reply(a,state,r,old,seed):
    state.restore(a)
    k=r["leg"]
    std=live.draw

    def draw(c, j, mode, rng):
        if j==k and seed is not None:
            c=dict(c,reply_seed=int(seed))
        return std(c,j,mode,rng)

    x=r["anchor"]
    until=r["native_departure"]["step"]+2
    hold=np.zeros(6)
    end=x+175
    with patch.object(live,"draw",draw):
        for t in range(x,end):
            u=old["actions"][t] if t<min(until,len(old["actions"])) else hold
            _,_,terminal,capped,_=a.exchange(u,mpc.FAR_COMMAND)
            if terminal or capped or k+1 in a.crossings:
                break
    if not any(e.get("collider")=="learner_racket" and digest_json(e)==digest_json(r["native_departure"])
               for e in a.events):
        raise ValueError("Redraw physically changed the prior arm stroke")
    p=a.point_placements()
    if len(p["requested"])<=k or any(row["target"]!=list(draw(r["card"],j,"random",None)[0])
                                    for j,row in enumerate(p["requested"])):
        raise ValueError("Native opponent departed from the assigned continuous draw")
    got=p["delivered"][k] if len(p["delivered"])>k else None
    pre=p["delivered"][k-1] if len(p["delivered"])>=k else None
    line=got["receive_line"] if got else None
    return dict(answered=bool(got),prior_arm_landed=bool(pre and pre["near_landed"]),
                target=p["requested"][k]["target"],measured=line,
                h=model.projected(np.r_[line["xyz"],line["velocity"]]).tolist() if line else None,
                far_contact=got["far_contact"] if got else None,
                bounce=got["near_bounce"] if got else None,
                terminal=bool(terminal), seed=seed)


def agrees(a,b):
    """Native physical equality up to negligible roundoff, with exact discrete events."""
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(agrees(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(agrees(x,y) for x,y in zip(a,b,strict=True))
    if isinstance(a,float) and isinstance(b,float):
        return abs(a-b)<2e-9
    return a==b


def run():
    plan=_read(PLAN)
    if plan["archive_sha256"]!=_hash(model.SOURCE):
        raise ValueError("Earlier causal snapshot changed after native assignment")
    receipt=_read(TAPE) if TAPE.exists() else dict(plan_sha256=_hash(PLAN),rows=[],complete=False)
    if receipt["plan_sha256"]!=_hash(PLAN):
        raise ValueError("Earlier assignment changed")
    cache={}
    cached_ready=None
    a=None
    try:
        for r in plan["cases"][len(receipt["rows"]):]:
            p=r["source"]
            if p not in cache:
                if _hash(Path(p))!=plan["sources"][p]:
                    raise ValueError("Changed actual source")
                cache[p]=_read(Path(p))
            old=cache[p]["rows"][r["index"]]; ready=cache[p]["ready"]
            if cached_ready!=ready:
                if a:
                    a.close()
                with mpc.centered(ready):
                    a=live.arena(width=64,height=48)
                cached_ready=ready
            with mpc.centered(ready):
                c=r["card"]
                a.launch(old["shot"],card=c)
                a.partner.reset(seed=c["seed"])
                for z in old["actions"][:r["anchor"]]:
                    a.exchange(z,mpc.FAR_COMMAND)
            prefix=[e for e in old["events"] if e["time"]<=a.data.time+1e-8]
            if digest_json(prefix)!=digest_json(a.events) or len(a.partner.requests)!=r["leg"]:
                raise ValueError("Anchor did not reproduce the old native prefix before the new request")
            warm=State(a)
            check=physical_reply(a,warm,r,old,None)
            i=r["leg"]
            orig=old["placements"]["delivered"]
            got=orig[i] if len(orig)>i else None
            if (check["answered"]!=bool(got) or not check["prior_arm_landed"] or
               (got and any(not agrees(check[k],got[p])
                    for k,p in (("far_contact","far_contact"),("bounce","near_bounce")))) or
               (got and got["receive_line"] is not None and
                    not agrees(check["measured"],got["receive_line"]))):
                raise ValueError("Old native target reference no longer matches at this afterstate")
            test=[physical_reply(a,warm,r,old,s) for s in r["seeds"]]
            row=r|dict(cohort="native",replies=[x["h"] for x in test],
                       branch=test,matched_original=check,pre_request_sha256=digest_json(prefix),
                       original_float_tolerance=2e-9,
                       external_hindsight_height=None)
            receipt["rows"].append(row)
            receipt["complete"]=len(receipt["rows"])==len(plan["cases"])
            _write(TAPE,receipt)
            print(len(receipt["rows"]),r["id"],sum(x["answered"] for x in test),
                  sum(x["h"] is not None for x in test),flush=True)
    finally:
        if a:
            a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("plan","run"))
    p.add_argument("--n",type=int,default=32)
    a=p.parse_args()
    freeze(a.n) if a.phase=="plan" else run()
