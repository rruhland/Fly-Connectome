"""Independently replay changed paired returns with the full unchanged CAD court."""

import argparse
from pathlib import Path

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt40_long_low import same_trace
from .tt44_paired_credit import ROOT, sources


def audit(training,output,limit,include):
    data=_read(training)
    eligible=[]
    for case in data["rows"]:
        if include and case["id"] not in include:
            continue
        base=case["variants"]["stock"]["landed"]
        changed=[k for k,v in case["variants"].items() if v["landed"]!=base]
        if changed:
            eligible.append((case,["stock",*changed]))
    original=sources()
    boot=next(iter(original.values()))[1]["ready"]
    receipt=dict(training_sha256=_hash(training),rows=[])
    with mpc.centered(boot):
        a=mpc.arena()
        cad=mpc.arena(detail="cad")
        guard=CourtClearance(cad.model,articulated=True)
        try:
            for case,roles in eligible[:limit]:
                _,source=original[case["source"]]
                old=source["rows"][case["index"]]
                card=case["card"]
                expect=old["placements"]["delivered"][case["leg"]]
                branch=[]
                for role in roles:
                    contacts=[]
                    a.launch(old["shot"],card=card)
                    a.partner.reset(seed=card["seed"])
                    guarded(a,guard)
                    result=case["variants"][role]
                    tape=old["actions"][:case["tick"]]+result["controls"]
                    with contact_observer(a,contacts):
                        for u in tape:
                            a.exchange(u,mpc.FAR_COMMAND)
                    verify_clearance(a)
                    if digest_json(a.events)!=result["events_sha256"]:
                        raise ValueError("Restored-state branch didn't replay from the original launch")
                    o=a.point_placements()["delivered"][case["leg"]]
                    if not all(same_trace(o[k],expect[k]) for k in ("far_contact","near_bounce")):
                        raise ValueError("Different physical delivery")
                    branch.append(dict(role=role,landed=o["near_landed"],
                                       native_contacts=[e for e in contacts if e["time_s"]>o["near_bounce"]["time"]],
                                       event_sha256=digest_json(a.events),clearance=a.clearance_result()))
                receipt["rows"].append(dict(id=case["id"],leg=case["leg"],exact_archive_delivery=True,variants=branch))
                _write(output,receipt)
                print(case["id"],[(v["role"],v["landed"],v["clearance"]["penetration_samples"]) for v in branch],flush=True)
        finally:
            a.close();cad.close()
    receipt["complete"]=True
    _write(output,receipt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training",type=Path,default=ROOT/"training.json.gz")
    p.add_argument("--output",type=Path,default=ROOT/"audit.json")
    p.add_argument("--limit",type=int,default=3)
    p.add_argument("--ids",type=int,nargs="+")
    a=p.parse_args()
    audit(a.training,a.output,a.limit,a.ids)
