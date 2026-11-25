"""Outcome and exact-tape audit for the original, centered sensitivity-bank course."""
import argparse
from collections import Counter

import numpy as np

from . import tt74_sensitivity_bank as tt
from .tt4_transfer_trial import _hash, _read, _write
from .tt40_long_low import same_trace

REPORT = tt.ROOT / "report.json"


def counts(tape):
    points=Counter()
    for o in tape["placements"]["delivered"]:
        receive=o.get("receive_line")
        if receive:
            x=float(receive["xyz"][0])
            height=float(tt.school.projected_height(receive))
            lane=int(np.searchsorted([-.10,.10],x))
            group=["left","middle","right"][lane]+("_low" if height<1.035 else "_other")
        else:
            group="no_receive_line"
        points[(group,"delivered")]+=1
        points[(group,"touched")]+=bool(o.get("near_contact"))
        points[(group,"scored")]+=bool(o["near_landed"])
    return {str(label):{k:int(v) for (g,k),v in points.items() if g==label}
            for label in sorted({g for g,k in points})}


def assumptions(r, stock):
    offers=[]
    for s in r["repairs"]:
        if not s["issued"]:
            continue
        near = [o for o in r["placements"]["delivered"] if o.get("far_contact")
                and o["far_contact"]["step"]<=s["tick"]]
        o=max(near,key=lambda x:x["leg"],default=None)
        if not o:
            offers.append(dict(tick=s["tick"],delivered=False, source=s["selection"]["experience"]))
            continue
        l=o["leg"]
        old=stock["placements"]["delivered"]
        matched = len(old)>l and same_trace(old[l]["far_contact"],o["far_contact"]) and same_trace(old[l]["near_bounce"],o["near_bounce"])
        touched=bool(o.get("near_contact") and 0<=o["near_contact"]["step"]-s["tick"]<=tt.H+2)
        offers.append(dict(tick=s["tick"],leg=l,delivered=True,matched_incoming=matched,
                           stock_score=bool(old[l]["near_landed"]) if matched else None,
                           touched=touched, scored=bool(touched and o["near_landed"]),
                           answered=bool(touched and o["near_landed"] and len(r["placements"]["delivered"])>l+1),
                           source=s["selection"]["experience"]))
    return offers


def report(verify):
    doc=_read(tt.OUT);p=_read(tt.PLAN);bank=_read(tt.MODEL)
    if doc["plan"]!=_hash(tt.PLAN) or doc["bank"]!=_hash(tt.MODEL):
        raise ValueError("Changed experiment")
    prior=_read(REPORT) if REPORT.exists() else dict(rows=[])
    replayed={r["id"]:r["replays"] for r in prior["rows"] if r.get("replays")}
    result=dict(protocol=_hash(tt.PLAN),bank=_hash(tt.MODEL),games=_hash(tt.OUT),
                complete=doc["complete"],rows=[],authority=tt.AUTHORITY,
                catalog=Counter(r["entry"]["authority"]>=tt.AUTHORITY
                              for r in bank["observations"] if "entry" in r),
                summary={k:Counter() for k in tt.MODES})
    for row,spec in zip(doc["rows"],p["releases"]):
        c=row["card"]
        if c!=spec["card"] or row["shot"]!=spec["shot"]:
            raise ValueError("Order or measured launch changed")
        stock,learn=(row["arms"][k] for k in tt.MODES)
        delta=next((i for i,(x,y) in enumerate(zip(stock["actions"],learn["actions"])) if x!=y),None)
        x=dict(id=c["id"],first_difference=delta, exact_changes=learn["same_offer_changes"],
               interventions=assumptions(learn,stock), replays=replayed.get(c["id"],{}), arms={})
        if verify and delta is not None and not x["replays"]:
            with tt.school.enabled(cameras=False):
                for mode,a in row["arms"].items():
                    audit=tt.game.check.replay(p["ready"],c,a)
                    audit["clearance_positive"]=not audit["clearance"]["penetration_samples"] and audit["clearance"]["nearest"]["signed_m"]>0.
                    x["replays"][mode]=audit
        for mode,a in row["arms"].items():
            by=counts(a)
            simple=dict(legal=a["score"]["learner"],opponent=a["score"]["opponent"],
                        observed_deliveries=len(a["placements"]["delivered"]),
                        elapsed_ticks=len(a["actions"]),complete=a["complete_point"],
                        by_lane=by, actual_contacts=sum(v["touched"] for v in by.values()),
                        repairs=Counter(k.get("reason","issued") for k in a["repairs"]),
                        learned=a["motor_lessons"])
            x["arms"][mode]=simple
            result["summary"][mode].update(games=1,returns=simple["legal"],offers=simple["observed_deliveries"],
                                           contacts=simple["actual_contacts"],learned=simple["learned"],
                                           complete=int(simple["complete"]),**simple["repairs"])
        result["rows"].append(x)
    _write(REPORT,result)
    print(result["complete"],result["summary"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify",action="store_true")
    report(p.parse_args().verify)
