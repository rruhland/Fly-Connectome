"""Check strong-carried exact-offer interventions on original full native launches."""

import argparse
from pathlib import Path

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt40_long_low import same_trace
from .tt48_exact_repair import ROOT, SOURCE


def audit(path,output,which):
    repaired,archive=_read(path),_read(SOURCE)
    if repaired["source_sha256"]!=_hash(SOURCE) or not repaired.get("complete"):
        raise ValueError("Need the complete repair receipt for the original physical source")
    report=dict(repair_sha256=_hash(path),source_sha256=_hash(SOURCE),rows=[])
    with mpc.centered(archive["ready"]):
        a=mpc.arena();cad=mpc.arena(detail="cad")
        checker=CourtClearance(cad.model,articulated=True)
        try:
            for c in repaired["rows"]:
                if c["id"] not in which: continue
                original=archive["rows"][c["index"]]
                offers=original["placements"]["delivered"]
                o=offers[c["leg"]]
                replay=[]
                for label,row in c["variants"].items():
                    a.launch(original["shot"],card=c["card"])
                    a.partner.reset(seed=c["card"]["seed"])
                    guarded(a,checker)
                    tape=original["actions"][:c["tick"]]+row["controls"]
                    contacts=[]
                    with contact_observer(a,contacts):
                        for cmd in tape:
                            a.exchange(cmd,mpc.FAR_COMMAND)
                    verify_clearance(a)
                    if digest_json(a.events)!=row["events_sha256"]:
                        raise ValueError("New action tape diverged from physical launch")
                    v=a.point_placements()["delivered"][c["leg"]]
                    if not all(same_trace(v[k],o[k]) for k in ("far_contact","near_bounce")):
                        raise ValueError("Target actual physical delivery changed")
                    r=dict(label=label,returned=v["near_landed"],contact_face_and_position=[
                        p for p in contacts if p["side"]==1 and p["time_s"]>v["near_bounce"]["time"]],
                        clearance=a.clearance_result(),events_sha256=digest_json(a.events))
                    replay.append(r)
                report["rows"].append(dict(id=c["id"],leg=c["leg"],variants=replay))
                _write(output,report)
                print(c["id"],[(x["label"],x["returned"],x["clearance"]["penetration_samples"])
                                for x in replay],flush=True)
        finally:
            cad.close();a.close()
    report["complete"]=True
    _write(output,report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=ROOT/"transfer.json.gz")
    p.add_argument("--output",type=Path,default=ROOT/"audit.json")
    p.add_argument("--ids",type=int,nargs="+",default=[7398,14429])
    a=p.parse_args()
    audit(a.source,a.output,a.ids)
