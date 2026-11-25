"""Summarize equal-budget signed correction and optionally replay genuine rescues."""

import argparse
from pathlib import Path

import numpy as np

from . import tt53_signed_practice as practice
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt48_contact_patch_report import replay

ARMS=("learned","blinded","reversed","prior_only")


def summary(data):
    rows=[r for r in data["rows"] if "baseline" in r]
    shared=[r for r in rows if r["common_practice_rescue"]]
    held=[r for r in rows if not r["common_practice_rescue"]]
    wins={k:[r["id"] for r in held if r["arms"][k]["scored"]] for k in ARMS}
    counts={k:len(shared)+len(w) for k,w in wins.items()}
    matches=[]
    for r in held:
        this={k:v["scored"] for k,v in r["arms"].items()}
        matches.append(dict(id=r["id"],tick=r["tick"],signed=r["decision"]["signed"],wins=this))
    return dict(n=len(rows),distinct=len({r["id"] for r in rows}),complete=data["complete"],
                signed=sum(r["decision"]["signed"] for r in rows),
                baseline_rgb=sum(r["baseline"]["sensed"] is not None for r in rows),
                observed_rgb=sum(e["sensed"] is not None for r in rows for e in r["practice"].values()),
                total_probes=sum(len(r["practice"]) for r in rows),
                scored_rgb_probes=sum(e["scored"] and e["sensed"] is not None for r in rows for e in r["practice"].values()),
                total_scored_probes=sum(e["scored"] for r in rows for e in r["practice"].values()),
                baseline=0,common_rescues=len(shared),common_releases=sorted({r["id"] for r in shared}),
                one_final_try=counts,additional_ids=wins,disagreements=matches,
                endpoints_used_in_actor=False,
                qualification_supported=(len(set(wins["learned"]))>=2 and counts["learned"]>max(counts[k] for k in ARMS[1:])))


def span(data):
    """Projection ceiling after unsealing the held known alternate only in evaluation."""
    lines=_read(practice.LINES)
    looked={(c["id"],c["tick"]):c for c in lines["cases"]}
    results=[]
    for case in data["rows"]:
        r=looked.get((case["id"],case["tick"]))
        if r is None or "basis" not in case:
            continue
        tensor=np.asarray(case["basis"]).reshape(2,-1).T
        d=np.cumsum(np.asarray(r["win"])-r["loss"],axis=0)[:,:5].ravel()
        cs=np.linalg.lstsq(tensor,d,rcond=None)[0]
        fit=tensor@cs
        results.append(dict(id=case["id"],tick=case["tick"],fraction_norm=float(np.linalg.norm(fit)/np.linalg.norm(d)),
                            coefficient=cs.tolist(),true_motion_norm=float(np.linalg.norm(d))))
    return dict(known_corrective_reference_projection=results,
                median_rms_fraction=float(np.median([x["fraction_norm"] for x in results])) if results else None,
                caveat="Uses the held successful endpoint only after all treatments. Projection is not evidence that a legal motor neighborhood is excluded or reachable.")


def verify(data):
    p,l=_read(practice.PAIRS),_read(practice.LINES)
    confirmed=[]
    cache={}
    for r in data["rows"]:
        if r.get("common_practice_rescue") or not r["arms"]["learned"]["scored"] or any(
                r["arms"][k]["scored"] for k in ARMS[1:]):
            continue
        i=next(j for j,c in enumerate(p["pairs"]) if c["prefix"]==r["prefix"] and c["id"]==r["id"])
        entry,doc=practice.stock_endpoint(p["pairs"][i],l["cases"][i],cache)
        old=doc["rows"][entry["row"]]; card=doc["cards"][entry["row"]]
        tick=r["tick"]
        past=[e for e in old["events"] if e["time"]<=tick*practice.mpc.DT+1.e-8]
        check={}
        for name,result in (("stock",r["baseline"]),("learned",r["arms"]["learned"])):
            events=past+result["events"]
            tape=dict(shot=old["shot"],actions=old["actions"][:tick]+result["controls"],
                      events=events,score=audit(events,opening_side=1))
            check[name]=replay(doc["ready"],card,tape)
        confirmed.append(dict(id=r["id"],tick=tick,arms=check))
    return confirmed


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",type=Path,nargs="?",default=practice.ROOT/"trial.json.gz")
    p.add_argument("--out",type=Path,default=practice.ROOT/"report.json")
    p.add_argument("--verify",action="store_true")
    p.add_argument("--span",action="store_true")
    a=p.parse_args()
    data=_read(a.data)
    result=dict(input_sha256=_hash(a.data), native_summary=summary(data))
    if a.span:
        result["evaluator_span"]=span(data)
    if a.verify:
        result["full_cad"]=verify(data)
    _write(a.out,result)
    print(result["native_summary"],flush=True)
