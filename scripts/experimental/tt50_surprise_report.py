"""Summarize the exact opening contrasts, including unsupported and failed transfers."""
import argparse
from pathlib import Path

from . import tt50_surprise_actor as trial
from . import tt50_surprise_data as data
from .tt4_transfer_trial import _hash, _read, _write
from .tt43_outgoing_coverage_report import verify


def report(tape, output, audit):
    d, plan = _read(tape), _read(trial.HELD)
    if d["evaluation_sha256"] != _hash(trial.HELD) or [r["case"] for r in d["rows"]] != plan["held"][:len(d["rows"])]:
        raise ValueError("Unmatched challenge slate")
    rows, wins = [], []
    for i,r in enumerate(d["rows"]):
        q={}
        for m,a in r["arms"].items():
            first = a["placements"]["delivered"][0] if a["placements"]["delivered"] else {}
            q[m]=dict(landed=bool(first.get("near_landed")),
                      touched=bool(first.get("near_contact")), rates=len(a["borrowed"]),
                      first_change=a.get("first_change"), same_offer=a.get("same_offer",m=="stock"),
                      attempts=a["attempt"],
                      first_after_borrow_predicted_legal=sum(bool(p.get("legal_trials")) for p in a["searches"]
                                                           if a["borrowed"] and p["tick"]>=a["borrowed"][0]["tick"]))
            if q[m]["landed"] and q[m]["same_offer"] and not bool(r["arms"]["stock"]["score"]["learner"]):
                wins.append((i,m,a))
        rows.append(dict(r["case"],arms=q))
    totals={}
    for m in trial.MODES:
        a=[r["arms"][m] for r in rows]
        b=[r["arms"][m] for r in rows if r["role"]=="weak opening"]
        totals[m]=dict(landed=sum(w["landed"] for w in a),weak_landed=sum(w["landed"] for w in b),
                       touched=sum(w["touched"] for w in a),fired=sum(w["rates"]>0 for w in a),
                       retained_offers=sum(w["same_offer"] for w in a),
                       issued=sum(w["rates"] for w in a))
    result=dict(rows=rows, totals=totals, complete=d["complete"], tape_sha256=_hash(tape))
    if audit and wins:
        i,mode,arm=max(wins,key=lambda q:q[1]=="action-conditional")
        original=_read(data.BASELINE)
        row=d["rows"][i]
        witness=dict(cards=[row["case"]["card"]], rows=[arm], ready=original["ready"], opening_side=1)
        result["first_rescue_audit"]=verify(witness,0)|dict(mode=mode)
        p=output.with_name("first-rescue-tape.json.gz")
        _write(p,witness);result["first_rescue_audit"]["tape_sha256"]=_hash(p)
    _write(output,result)
    print(totals,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tape",type=Path,default=data.ROOT/"held.json.gz")
    p.add_argument("--output",type=Path,default=data.ROOT/"held-report.json")
    p.add_argument("--audit",action="store_true")
    a=p.parse_args()
    report(a.tape,a.output,a.audit)
