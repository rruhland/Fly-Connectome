"""Report complete on-policy update evidence and independently replay the held winner."""
import argparse
from itertools import product

import numpy as np

from . import tt100_episode_policy as tt
from .tt4_transfer_trial import _hash,_read,_write
from .tt40_long_low import same_trace
from .tt48_contact_patch_report import replay

OUT=tt.ROOT/"report.json"


def same_inbound(a,b):
    return (a is not None and a.get("far_contact") is not None
            and same_trace(a["far_contact"],b["far_contact"])
            and same_trace(a["near_bounce"],b["near_bounce"]))


def first_difference(a,b):
    t=next((j for j,(u,v) in enumerate(zip(a["actions"],b["actions"])) if u!=v),None)
    if t is None:return None
    base=a["placements"]["delivered"]
    now=b["placements"]["delivered"]
    inbound=[o for o in now if o.get("far_contact") and o["far_contact"]["step"]<=t]
    o=max(inbound,key=lambda x:x["leg"],default=None)
    if o is None:return dict(tick=t,same_delivery=None)
    j=o["leg"]
    current=base[j] if j<len(base) else None
    same=same_inbound(current,o)
    return dict(tick=t,leg=j,same_delivery=same,baseline_scored=None if not same else current["near_landed"],
                candidate_scored=o["near_landed"])


def summarize(verify, extension=False):
    d=tt.design()
    check,tape,out=(tt.NEXT,tt.EXTRA,tt.ROOT/"extension-report.json") if extension else (tt.PLAN,tt.OUT,OUT)
    practice=_read(tt.PRACTICE)
    if practice["plan"]!=_hash(tt.PLAN):raise ValueError("Wrong exploration docket")
    result=dict(protocol=_hash(tt.PLAN),practice=_hash(tt.PRACTICE),practice_done=practice["complete"],
                rounds=[],held=None)
    for j,b in enumerate(practice["rounds"]):
        r=dict(block=j,theta=b["origin"],updated=b["updated"],matches=[])
        for m in b["matches"]:
            r["matches"].append(dict(id=m["card"]["id"],arms={
                k:dict(legal=e["score"]["learner"],complete=e["complete"],
                       cpu_s=e["cpu_s"],theta=e["theta"]) for k,e in m["arms"].items()}))
        if b["updated"]:
            r.update(next=b["next"],paired=b["paired"])
        result["rounds"].append(r)
    if tape.exists():
        h=_read(tape)
        if h["plan"]!=_hash(check) or h["policy"]!=_hash(tt.CHECK):raise ValueError("Wrong frozen weights or schedule")
        old=_read(out) if out.exists() else {}
        prior={r["id"]:r.get("replays",{}) for r in (old.get("held") or {}).get("rows",[])}
        hd=dict(receipt=_hash(tape),complete=h["complete"],rows=[],summary={})
        paired=[]
        indexed=list(r for r in h["rows"] if len(r["arms"])==2)
        longest=max(indexed,key=lambda r:r["arms"]["episode"]["score"]["learner"],default=None)
        gain=max(indexed,key=lambda r:r["arms"]["episode"]["score"]["learner"]-r["arms"]["ordinary"]["score"]["learner"],default=None)
        for r in indexed:
            c=r["card"]
            a,b=(r["arms"][k] for k in ("ordinary","episode"))
            if a["memory"]["before"]!=b["memory"]["before"]:
                raise ValueError("Rematched episode policies changed motor bank")
            still_matched=[]
            for old,now in zip(a["placements"]["delivered"],b["placements"]["delivered"]):
                if not same_inbound(old,now):break
                still_matched.append(dict(leg=old["leg"],returns=[old["near_landed"],now["near_landed"]],
                     exact_near_contact=same_trace(old["near_contact"],now["near_contact"])))
            current=dict(id=c["id"],same_commands=a["actions"]==b["actions"],
                         first_effect=first_difference(a,b),arms={},
                         same_incoming_initial_legs=still_matched,
                         replays=prior.get(c["id"],{}))
            for k,v in r["arms"].items():
                arr=v["placements"]["delivered"]
                current["arms"][k]=dict(legal=v["score"]["learner"],offers=len(arr),
                    touches=sum(o.get("near_contact") is not None for o in arr),
                    complete=v["complete"],cpu_s=v["cpu_s"],ticks=len(v["actions"]),
                    searches=v["planner"].get("searches",0),
                    native_candidates=v["planner"].get("native_candidates",0),
                    accepted=v["planner"].get("accepted",0))
                selected=(verify and (r is gain or r is longest or v["score"]["learner"]>=100))
                if selected and k not in current["replays"]:
                    with tt.enabled(cameras=False):
                        current["replays"][k]=replay(d["ready"],c,v)
            paired.append([a["score"]["learner"],b["score"]["learner"]])
            hd["rows"].append(current)
        if paired:
            pair=np.asarray(paired);diff=pair[:,1]-pair[:,0]
            maximum=np.max(pair,axis=0).tolist()
            d0=np.abs(diff[diff!=0])
            flips=np.asarray(list(product([-1,1],repeat=len(d0)))) if len(d0) else np.zeros((1,0))
            p_exact=float(np.mean(np.abs(flips@d0)>=abs(diff.sum())))
            hd["summary"]=dict(points=len(paired),score_pairs=pair.tolist(),totals=pair.sum(0).tolist(),
                              medians=np.median(pair,axis=0).tolist(),maxima=maximum,
                              gain=int(diff.sum()),wins=int((diff>0).sum()),losses=int((diff<0).sum()),
                              ties=int((diff==0).sum()),two_sided_paired_randomization=p_exact,
                              reaches_100=np.sum(pair>=100,axis=0).tolist())
            hd["summary"]["pooled_work"]={k:{f:sum(r["arms"][k][f] for r in hd["rows"])
                 for f in ("offers","touches","ticks","cpu_s","searches","native_candidates","accepted")}
                 for k in ("ordinary","episode")}
        result["held"]=hd
    _write(out,result)
    print("practice",[(r["block"],r["updated"],r.get("paired")) for r in result["rounds"]],flush=True)
    if result["held"] is not None:print(result["held"]["complete"],result["held"]["summary"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify",action="store_true")
    p.add_argument("--extension",action="store_true")
    a=p.parse_args()
    summarize(a.verify,a.extension)
