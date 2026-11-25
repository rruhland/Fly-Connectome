"""Matched nominal budgets, physical same-offer outcomes, and full-point audit."""
import argparse
from collections import Counter

from . import tt83_population_eval as tt
from . import tt83_playable_course as viable
from .tt4_transfer_trial import _hash,_read,_write
from .tt40_long_low import same_trace

REPORT=tt.ROOT/"report.json"


def audit(seed, receipt, mode):
    p=_read(tt.PLAN)
    with tt.school.enabled(cameras=False):
        v=tt.games0.check.replay(p["ready"],seed,receipt)
    v["clearance_positive"]=not v["clearance"]["penetration_samples"] and v["clearance"]["nearest"]["signed_m"]>0.
    return v


def first_affected(stock, treatment):
    t=next((i for i,(a,b) in enumerate(zip(stock["actions"],treatment["actions"])) if a!=b),None)
    if t is None:
        return None
    base=stock["placements"]["delivered"]
    candidates=[o for o in treatment["placements"]["delivered"] if o.get("far_contact") and o["far_contact"]["step"]<=t]
    o=max(candidates,key=lambda c:c["leg"],default=None)
    if o is None:
        return dict(tick=t)
    leg=o["leg"]
    old=base[leg] if len(base)>leg else None
    same=bool(old and same_trace(o["far_contact"],old["far_contact"]) and same_trace(o["near_bounce"],old["near_bounce"]))
    return dict(tick=t,leg=leg,same_incoming=same,
                stock_legal=bool(old["near_landed"]) if same else None,
                new_legal=bool(o["near_landed"]),
                contact_differs=not old or not same_trace(o.get("near_contact"),old.get("near_contact")))


def result(verify):
    p=_read(tt.PLAN)
    held=_read(tt.HELD)
    if held["plan"]!=_hash(tt.PLAN):
        raise ValueError("Wrong matched fork")
    receipt=dict(protocol=_hash(tt.PLAN),forks=_hash(tt.HELD),fork_complete=held["complete"],
                 fork_summary={m:Counter() for m in tt.MODES},fork_rows=[],games=None)
    for a in held["rows"]:
        c=a["case"]; cell={}
        for m,v in a["arms"].items():
            t=v["search"]
            p0=v["physical"]
            clear=p0.get("clearance")
            safe=bool(clear and not clear["penetration_samples"] and clear["nearest"]["signed_m"]>0.)
            matched=not p0.get("unmatched")
            success=v["issued"] and p0["landed"] and safe
            counts=dict(cases=1,population_candidates=t["screened"],final_audit_trials=t["audit"]["candidates"]+t["audit"]["refinement"],
                        first_nominal_legal=t["rounds"][0]["new_legal"],nominal_legal_trial_evaluations=t["legal"],
                        unique_nominal=t["unique_legal"],novel_after_first=t["unique_legal"]-t["rounds"][0]["new_legal"],
                        cases_with_nominal=int(t["legal"]>0),model_issued=int(t["issued"]),
                        actually_issued=int(v["issued"]),legal_physically=int(p0["landed"]),
                        rescued=int(success and not a["stock"]["landed"]),
                        rescued_answered=int(success and not a["stock"]["landed"] and p0["answered"]),
                        unmatched=int(not matched),cad_samples=0 if not clear else clear["samples"])
            receipt["fork_summary"][m].update(**counts)
            actual=p0.get("actual")
            cell[m]=dict(counts,clearance_positive=safe,contact_radius_m=None if not actual else actual["radius"])
        receipt["fork_rows"].append(dict(id=c["id"],roster=c["roster"],stock=a["stock"]["landed"], arms=cell))
    previous=_read(REPORT) if REPORT.exists() else {}
    for label,path,protocol in (("games",tt.GAMES,tt.PLAN),("playable_games",viable.OUT,viable.PLAN)):
        if not path.exists():
            continue
        g=_read(path)
        if g["plan"]!=_hash(protocol):
            raise ValueError("Wrong emitted normal-start tapes")
        done={r["id"]:r.get("replays",{}) for r in (previous.get(label) or {}).get("rows",[])}
        top=max(g["rows"],key=lambda j:max(a["score"]["learner"] for a in j["arms"].values()),default=None)
        value=dict(receipt=_hash(path),complete=g["complete"],rows=[],summary={m:Counter() for m in ("stock","iterative")})
        for row in g["rows"]:
            c=row["card"]
            stock,other=(row["arms"][k] for k in ("stock","iterative"))
            found=dict(id=c["id"],same_commands=stock["actions"]==other["actions"],
                       same_referee=tt.school.digest_json(stock["events"])==tt.school.digest_json(other["events"]),
                       changes=other["same_offer_changes"],first_affected=first_affected(stock,other),
                       replays=done.get(c["id"],{}),arms={})
            for m,a in row["arms"].items():
                actual=a["placements"]["delivered"]
                count=dict(complete=int(a["complete_point"]),games=1,legal=a["score"]["learner"],
                           opponent=a["score"]["opponent"],actual_deliveries=len(actual),
                           actual_contacts=sum(bool(x.get("near_contact")) for x in actual),
                           gates=len(a["expanded"]),online_issued=sum(x["issued"] for x in a["expanded"]),
                           model_legal=sum(x["legal"] for x in a["expanded"]),
                           unique_legal=sum(x["unique_legal"] for x in a["expanded"]),
                           from_later_round=sum(x.get("decision",{}).get("origin_round",0)>0 for x in a["expanded"]),
                           warm_gates=sum(x["warm_roots"]>0 for x in a["expanded"]),
                           kept_stock=a["ordinary_kept"].get("kept-stock",0),cpu_s=a["cpu_s"])
                value["summary"][m].update(**count)
                found["arms"][m]=count
                if verify and m not in found["replays"] and (not found["same_commands"] or row is top and m=="stock"):
                    found["replays"][m]=audit(c,a,m)
            value["rows"].append(found)
        receipt[label]=value
    _write(REPORT,receipt)
    print(receipt["fork_complete"],receipt["fork_summary"],flush=True)
    for group in ("games","playable_games"):
        if receipt.get(group):
            print(group,receipt[group]["complete"],receipt[group]["summary"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify",action="store_true")
    result(p.parse_args().verify)
