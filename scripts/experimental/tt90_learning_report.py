"""Summarize native TT90 practice and frozen matched generalization."""

from collections import Counter

import numpy as np

from . import tt90_installation_learning as tt
from . import tt90_attained_practice as coach
from .tt4_transfer_trial import _hash, _read, _write
from .tt40_long_low import same_trace


def first(r):
    offers=r["placements"]["delivered"]
    return (bool(offers and offers[0]["near_landed"]),
            bool(offers and offers[0]["near_landed"] and len(offers)>1 and offers[1]["near_bounce"]))


def offered(rows):
    out={}
    for key in ("low","transition","ordinary"):
        out[key]=dict(delivered=0,scored=0,first=0,first_scored=0)
    for r in rows:
        for i,offer in enumerate(r["placements"]["delivered"]):
            line=offer["receive_line"]
            if line is None or offer["near_bounce"] is None:continue
            z=tt.optics.projected_height(line)
            grade="low" if z<1.005 else "transition" if z<1.047 else "ordinary"
            a=out[grade]; scored=int(offer["near_landed"])
            a["delivered"]+=1;a["scored"]+=scored;a["first"]+=int(i==0);a["first_scored"]+=int(i==0)*scored
    return out


def compare(d,kind):
    names=d["arms"]; rows=d["rows"]; out=dict(point_count=len(rows),mixed_roster=len(d["cases"]),arms={},contrasts={})
    for k in names:
        games=[x["results"][k] for x in rows]; u=[r["score"]["learner"] for r in games]
        value=dict(total_returns=sum(u),best=max(u,default=0),points_by_returns={str(i):sum(x>=i for x in u) for i in (1,3,8,16,32,64,100)},
                   actual_deliveries=offered(games),points_with_body_queries=sum(bool(r["body_proposals"]) for r in games),
                   learned_proposals=sum(len(p["paths"]) for r in games for p in r["body_proposals"]),
                   directly_selected=sum(len(p.get("directly_selected",[])) for r in games for p in r["body_proposals"]))
        if kind=="hard":
            value["by_stratum"]={label:dict(n=len(a),matched=sum(r["matched"] for r in a),
                scored=sum(first(r)[0] for r in a),answered=sum(first(r)[1] for r in a),
                exact_cad_passes=sum(not r["clearance"]["penetration_samples"] for r in a)) for label in ("low","ordinary")
                if (a:=[x["results"][k] for x in rows if x["case"]["kind"]==label])}
        out["arms"][k]=value
    for lhs,rhs in (("ordinary","original"),("body","ordinary"),("body","original"),("body","sham")):
        if lhs not in names or rhs not in names:continue
        step=[]; differences=[]; exact=0
        for x in rows:
            a=x["results"][lhs]; b=x["results"][rhs]
            av=a["placements"]["delivered"]; bv=b["placements"]["delivered"]
            delta=a["score"]["learner"]-b["score"]["learner"]; differences.append(delta)
            if av and bv and all(same_trace(av[0][k],bv[0][k]) for k in ("far_contact","near_bounce")):
                exact+=1
                ap=first(a);bp=first(b)
                if ap!=bp:
                    step.append(dict(id=x["case"]["card"]["id"],grade=x["case"]["kind"],test=list(ap),control=list(bp)))
        dif=np.asarray(differences)
        out["contrasts"][lhs+"_versus_"+rhs]=dict(n_same_first=exact,different_same_first=step,
            gained_complete_points=int(sum(dif>0)),lost_complete_points=int(sum(dif<0)),difference_total=int(sum(dif)))
    return out


def report():
    p=_read(tt.PLAN); out=dict(protocol=_hash(tt.PLAN),planned={k:len(p[k]) for k in ("learn","assess","hard_pool")})
    if tt.path("practice").exists():
        d=_read(tt.path("practice")); rows=d["rows"]; u=[r["score"]["learner"] for r in rows]
        out["practice"]=dict(n=len(u),score_total=sum(u),peak=max(u,default=0),scores=u,actual_deliveries=offered(rows),
            optical_earned=sum(len(r["optical_earned"]) for r in rows),optical_missed=sum(bool(r["optical_missed"]) for r in rows))
    if coach.OUT.exists():
        d=_read(coach.OUT); hist=Counter(); openings={}
        for m,r in zip(d["cases"],d["rows"]):
            q=m["kind"];hist[q]+=1
            a=next((v for v in r["attempts"] if v.get("strict",{}).get("penetration_samples")==0),None)
            if a is not None:openings.setdefault(q,[]).append(dict(id=m["id"],leg=m["leg"],radius=a["contact"]["radial_m"]))
        out["coaching"]=dict(attempted=dict(hist),actual_scored_and_answered=openings,coached_sha256=_hash(coach.OUT))
    for kind in ("hard","assess"):
        if tt.path(kind).exists():out[kind]=compare(_read(tt.path(kind)),kind)
    _write(tt.ROOT/"summary.json",out)
    print(out,flush=True)


if __name__=="__main__":report()
