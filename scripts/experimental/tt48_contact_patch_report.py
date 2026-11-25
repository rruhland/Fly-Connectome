"""Audit center-versus-edge practice and the first same-offer optical rescue."""
import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt40_long_low import same_trace
from .tt48_contact_patch import ROOT
from .tt48_rim_recovery import PRACTICE, STOCK


def replay(ready, card, tape):
    with mpc.centered(ready):
        a, s = mpc.arena(), mpc.arena(detail="cad")
        try:
            guard = CourtClearance(s.model, articulated=True)
            a.launch(tape["shot"], card=card)
            a.partner.reset(seed=card["seed"])
            guarded(a, guard)
            observed = []
            with contact_observer(a, observed):
                for u in np.asarray(tape["actions"]):
                    a.exchange(u, mpc.FAR_COMMAND)
            if a.launches != 1 or digest_json(a.events) != digest_json(tape["events"]):
                raise ValueError("Archived controlled point did not replay exactly")
            if digest_json(audit(a.events, opening_side=1)) != digest_json(tape["score"]):
                raise ValueError("Score changed during audit")
            verify_clearance(a)
            return dict(launches=1, clearance=a.clearance_result(), native_contacts=len(observed),
                        event_sha256=digest_json(a.events))
        finally:
            s.close()
            a.close()


def report(path, out, verify=False):
    old = _read(STOCK)
    live = _read(path)
    practice = _read(PRACTICE)
    if live["control_sha256"] != _hash(STOCK):
        raise ValueError("Different matched control")
    notes = {
        "practice": [
            dict(id=r["id"], height=r["height"], private=r["private"],
                 arms={name: dict(landed=s["landed"], answered=s["answered"],
                                  contact=s["actual"], clearance=s.get("clearance"))
                       for name,s in r["arms"].items()})
            for r in practice["rows"]
        ],
        "held_sha256": _hash(path),
        "control_sha256": _hash(STOCK),
        "complete": live["complete"],
        "held": [],
    }
    for c,a,b in zip(live["cards"],old["rows"],live["rows"]):
        tick=b["first_action_diff"]
        matched=[]
        for sa,sb in zip(a["placements"]["delivered"],b["placements"]["delivered"]):
            if not same_trace(sa["far_contact"],sb["far_contact"]) or not same_trace(sa["near_bounce"],sb["near_bounce"]):
                continue
            if sa["near_landed"]!=sb["near_landed"]:
                matching_corrections=[r for r in b["edge_actions"] if r["issued"]
                and sb["far_contact"]["step"]<=r["tick"]<(
                    (sb["near_contact"] or sb["near_bounce"])["step"]+2)]
                matched.append(dict(leg=sa["leg"], original=sa["near_landed"],
                                    replacement=sb["near_landed"], prior_contact=sa["near_contact"],
                                    new_contact=sb["near_contact"],
                                    correction=matching_corrections))
        r=dict(id=c["id"],first_differing_tick=tick,stock_arm=a["score"]["learner"],
               new_arm=b["score"]["learner"],same_delivered_arm_changes=matched,
               issued=sum(n["issued"] for n in b["edge_actions"]),
               attempts=len(b["edge_actions"]))
        if (verify and "audit_witness" not in notes and
                any(t["replacement"] and t["correction"] for t in matched)):
            notes["audit_witness"] = dict(
                id=c["id"], baseline=replay(old["ready"],c,a), treatment=replay(live["ready"],c,b),
                same_ball_changes=[t for t in matched if t["replacement"] and t["correction"]])
        notes["held"].append(r)
    notes["scores"]=[sum(r[k] for r in notes["held"]) for k in ("stock_arm","new_arm")]
    _write(out,notes)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--held",type=Path,default=ROOT/"held.json.gz")
    p.add_argument("--out",type=Path,default=ROOT/"report.json")
    p.add_argument("--verify",action="store_true")
    a=p.parse_args()
    report(a.held,a.out,a.verify)
