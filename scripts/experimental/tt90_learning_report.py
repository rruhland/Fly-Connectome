"""Summarize native TT90 practice and frozen matched generalization."""

from collections import Counter
import argparse

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


def audit_practice(prefix="practice-dose16"):
    p=tt.contract(tt.FEEDS);m,ready,_=tt.material(); src=tt.path(prefix);d=_read(src)
    cases=[]
    peak=max(range(len(d["rows"])),key=lambda i:d["rows"][i]["score"]["learner"])
    for i,(c,r) in enumerate(zip(d["cards"],d["rows"])):
        low=[v for v in r["placements"]["delivered"] if v["near_landed"] and v["receive_line"] and
             tt.optics.projected_height(v["receive_line"])<1.0]
        if i==peak or low:cases.append((c,r,low,i==peak))
    result=[]
    with tt.school.enabled(),tt.system.installed(m["v"],m["mount"]),tt.BASE.centered(ready):
        a=tt.BASE.arena(width=64,height=48); cad=tt.BASE.arena(width=64,height=48,detail="cad")
        guard=tt.school.physical.GripClearance(cad.model,stock=True)
        try:
            for c,r,low,peak in cases:
                found=[]
                with tt.school.contact.physical_contacts(a,found):
                    clear=tt.validate_game(a,c,r["shot"],r,guard)
                impacts=[]
                for offer in low:
                    e=offer["near_contact"]; t=e["time"]
                    hit=min((w for w in found if w["side"]==1),key=lambda w:abs(w["time_s"]-t))
                    if abs(hit["time_s"]-t)>1e-7:raise ValueError("Archive low differs from actual replay")
                    impacts.append(dict(leg=offer["leg"],height=tt.optics.projected_height(offer["receive_line"]),contact=hit))
                result.append(dict(id=c["id"],index=c["index"],peak=peak,scored=r["score"]["learner"],
                                   strict_cad=clear,matched_events=True,hard_contacts=impacts))
        finally:cad.close();a.close()
    _write(tt.ROOT/(prefix+"-audit.json"),dict(native_source=_hash(src),protocol=_hash(tt.PLAN),rows=result))
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-practice")
    a=p.parse_args()
    audit_practice(a.audit_practice) if a.audit_practice else report()
