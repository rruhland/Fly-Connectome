"""Confirm exact branch parity and candidate costs in the bounded physical fallback test."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt53_geometry_live import OUT, PLAN, STOCK, geometry
from .tt19_view_full_game import digest_json


def summary(path,protocol):
    d,p,s=map(_read,(path,protocol,STOCK))
    if (d["protocol_sha256"]!=_hash(protocol) or p["stock_sha256"]!=_hash(STOCK)
            or p["cards"]!=s["cards"][:len(p["cards"])]):
        raise ValueError("Different fixed native comparison")
    detail=[]
    for old,new,c in zip(s["rows"],d["rows"],p["cards"]):
        if c["id"]!=new["id"]:
            raise ValueError("Changed native game order")
        exact=(old["actions"]==new["actions"] and digest_json(old["events"])==digest_json(new["events"]))
        detail.append(dict(id=c["id"],stock=old["score"]["learner"],learned=new["score"]["learner"],
                          identical_controls_and_events=exact,decisions=len(new["basis"]),
                          slots=sum(r["candidates"] for r in new["basis"]),
                          issued=sum(r["issued"] is not None for r in new["basis"]),
                          same_ball_changes=new["exact_offer_leg_changes"],
                          native_logical_rows=[r["planner"].get("native_candidates",0) for r in (old,new)]))
    add=lambda name:sum(r[name] for r in detail)
    counts=[sum(r["native_logical_rows"][i] for r in detail) for i in (0,1)]
    return dict(complete=d["complete"],n=len(detail),games=detail,
                stock=add("stock"),learned=add("learned"),decisions=add("decisions"),
                slots=add("slots"),issued=add("issued"),native_logical_rows=counts,
                rescues=sum(r["learned_landed"] for g in detail for r in g["same_ball_changes"]),
                tape_sha256=_hash(path),protocol_sha256=_hash(protocol))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=OUT)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--out",type=Path,default=geometry.ROOT/"geometry-live-report.json")
    a=p.parse_args()
    result=summary(a.input,a.plan)
    _write(a.out,result)
    print(result,flush=True)
