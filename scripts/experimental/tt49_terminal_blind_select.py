"""Select one non-regressing blind compromise from fixed training draws only."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt49_terminal_blind import ROOT, run


def select(plan,training,out):
    p,d=_read(plan),_read(training)
    expected=[r for r in p["roster"] if r["split"]=="training"]
    if (not d.get("complete") or d["split"]!="training" or d["plan_sha256"]!=_hash(plan)
            or [r["stream"] for r in d["rows"]]!=[r["stream"] for r in expected]
            or any(not r["same_physical_incoming"] for r in d["rows"])):
        raise ValueError("Train only on the predeclared paired physical training draws")
    branches=[m for m in p["variants"] if m!="stock"]
    results={}
    for m in branches:
        outcome=dict(easy=0,stock_easy=0,hard=False,stock_hard=False,regressions=0)
        for r in d["rows"]:
            a,b=(r["branches"][k] for k in ("stock",m))
            if r.get("labelled"):
                outcome["stock_hard"]=a["legal"]
                outcome["hard"]=b["legal"]
            else:
                outcome["stock_easy"]+=int(a["legal"])
                outcome["easy"]+=int(b["legal"])
                outcome["regressions"]+=int(a["legal"] and not b["legal"])
        results[m]=outcome
    # An existing success must remain reachable. If immediate outcomes are
    # equal, choose the smaller centered physical change before any holdout.
    eligible=[m for m in branches if results[m]["regressions"]==0]
    choice=(max(eligible,key=lambda k:(results[k]["hard"],results[k]["easy"],k=="partial"))
            if eligible else None)
    result=dict(schema="tt49-blind-compromise-selection-1",plan_sha256=_hash(plan),
                training_sha256=_hash(training),selected=choice,outcomes=results,
                rule="No regression on fresh training offers. Prefer a hard-source repair, then easy successes, then smaller centered excursion.")
    _write(out,result)
    return choice


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=ROOT/"compromise-protocol.json")
    p.add_argument("--training",type=Path,default=ROOT/"compromise-training.json.gz")
    p.add_argument("--decision",type=Path,default=ROOT/"compromise-selection.json")
    p.add_argument("--held-output",type=Path)
    a=p.parse_args()
    chosen=select(a.plan,a.training,a.decision)
    print("selected",chosen,flush=True)
    if chosen is not None and a.held_output is not None:
        run(a.plan,a.held_output,"held",chosen=chosen,extra=2)
