"""Summarize measured easy returns and damage separately from branch-dependent lengths."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt56_postures import MODEL, OUT, PLAN, ROOT


def summary(tape):
    rows=tape["rows"]
    data=dict(complete=tape["complete"],matched_games=len(rows),stock=sum(r["stock_score"] for r in rows),
              order=[r["id"] for r in rows],stock_verified=tape["archive_verified_first"],arms={})
    for k in ("hub","ports"):
        rr=[r["arms"][k] for r in rows]
        comparisons=[dict(o,id=r["id"]) for r,arm in zip(rows,rr,strict=True) for o in arm["matched"]]
        def counts(f):
            use=[r for r in comparisons if f(r)]
            return dict(offered=len(use),stock=sum(r["stock"] for r in use),new=sum(r["learned"] for r in use),
                        rescues=[r for r in use if r["learned"] and not r["stock"]],
                        damage=[r for r in use if r["stock"] and not r["learned"]])
        out=dict(score=sum(v["score"]["learner"] for v in rr),
                 per_game=[v["score"]["learner"] for v in rr],
                 visited=sum(p["pool"] is not None and p["moves"]>0 for r in rr for p in r["posture_routes"]),
                 modes=sorted({p["pool"] for r in rr for p in r["posture_routes"] if p["pool"] is not None and p["moves"]>0}),
                 changed_games=sum(r["first_action_diff"] is not None for r in rr),
                 first_diff=[r["first_action_diff"] for r in rr],
                 matched=counts(lambda r:True),ordinary=counts(lambda r:r["height"] is not None and r["height"]>=1.025),
                 low=counts(lambda r:r["height"] is not None and r["height"]<1.025))
        data["arms"][k]=out
    return data


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=OUT)
    p.add_argument("--out",type=Path,default=ROOT/"report.json")
    a=p.parse_args()
    tape,fit=_read(a.input),_read(MODEL)
    if tape["protocol_sha256"]!=_hash(PLAN):
        raise ValueError("Wrong receiving posture comparison")
    result=dict(summary(tape),input_sha256=_hash(a.input),plan_sha256=_hash(PLAN),
                fit=dict(trained=len(fit["rows"]),lanes=fit["bins"],observed_face=fit["face"],
                         home_cost=fit["home_cost"],waypoints=fit["portfolio"]))
    _write(a.out,result)
    print(result,flush=True)
