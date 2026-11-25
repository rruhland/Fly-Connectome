"""Verify later handoffs, then allow their new far replies to finish under native physics."""

import argparse
from pathlib import Path

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded
from .tt40_long_low import same_trace
from .tt40_low_feasibility import replay_prefix
from .tt44_paired_credit import State
from .tt48_terminal_repair import ARCHIVE, terminal
from .tt49_terminal_blind import ROOT


def run(path,out):
    d=_read(path)
    if not d.get("complete"):raise ValueError("Need complete timing bracket")
    event,reference=terminal(_read(ARCHIVE))
    rows=[r for r in d["states"] if r["landed"] and not r["changed_tail_clearance"]["penetration_samples"]]
    fork=min(r["tick"] for r in rows)
    source=_read(ARCHIVE)
    doc=dict(schema="tt49-handoff-answers-1",bracket_sha256=_hash(path),source_sha256=_hash(ARCHIVE),rows=[])
    with mpc.centered(source["ready"]):
        a=mpc.arena();cad=mpc.arena(detail="cad")
        check=CourtClearance(cad.model,articulated=True)
        try:
            replay_prefix(a,reference,event["card"],fork)
            state=State(a)
            for v in rows:
                state.restore(a)
                for u in reference["actions"][fork:v["tick"]]: a.exchange(u,mpc.FAR_COMMAND)
                guarded(a,check)
                for u in v["controls_after_cut"]: a.exchange(u,mpc.FAR_COMMAND)
                if digest_json(a.events)!=v["events_sha256"]:
                    raise ValueError("Later retimed motion did not physically replay")
                tail=[];contacts=[]
                with contact_observer(a,contacts):
                    for _ in range(95):
                        if a.terminated or a.truncated or a.rival_returns>=44:break
                        u=[0.]*6
                        a.exchange(u,mpc.FAR_COMMAND);tail.append(u)
                own=a.legal_returns-1
                proof=a.point_placements()["delivered"][42]
                eq=all(same_trace(proof[k],event["original_offer"][k]) for k in ("far_contact","near_bounce"))
                if not eq or own!=43:
                    raise ValueError("Original 43rd offer changed")
                r=dict(tick=v["tick"],body_actuated_legal=True,physical_opponent_answered=a.rival_returns>=44,
                       own=own,opponent=a.rival_returns,expected_offer_exact=eq,
                       events_sha256=digest_json(a.events),additional_commands=tail,
                       terminal_reason=next((e.get("reason") for e in a.events[::-1] if e["kind"]=="point"),None),
                       changed_tail_full_cad=a.clearance_result())
                doc["rows"].append(r)
                _write(out,doc)
                print(v["tick"],r["physical_opponent_answered"],r["terminal_reason"],len(tail),flush=True)
        finally:
            a.close();cad.close()
    doc["complete"]=True
    _write(out,doc)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=ROOT/"handoff.json")
    p.add_argument("--output",type=Path,default=ROOT/"handoff-finish.json")
    a=p.parse_args()
    run(a.source,a.output)
