"""After-cue native teaching only on the frozen training reply draws."""

import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded
from .tt40_long_low import same_trace
from .tt40_low_feasibility import Probe, replay_prefix, trial
from .tt44_paired_credit import State
from .tt48_terminal_repair import ARCHIVE, terminal
from .tt49_terminal_blind import PLAN, ROOT


def run(plan,traffic,out,batch,rounds):
    capture=_read(traffic)
    protocol=_read(plan)
    source=_read(ARCHIVE)
    case,original=terminal(source)
    if _hash(plan)!=capture["plan_sha256"] or not capture.get("complete"):
        raise ValueError("Run only after complete original paired preparation")
    passed={x["stream"] for x in protocol["roster"] if x["split"]=="training"}
    entries=[x for x in capture["rows"] if x["stream"] in passed and x["same_physical_incoming"]]
    if not entries or any(x["split"]!="training" for x in entries):
        raise ValueError("Do not use held draw outcomes for teacher fitting")
    result=dict(schema="tt49-training-only-after-cue-teacher-1",native_sha256=_hash(traffic),
                plan_sha256=_hash(plan),source_sha256=_hash(ARCHIVE),batch=batch,rounds=rounds,rows=[])
    leg=case["leg"]; fork=protocol["anchor"]
    with mpc.centered(source["ready"]):
        a=mpc.arena();cad=mpc.arena(detail="cad")
        checker=CourtClearance(cad.model,articulated=True)
        try:
            result["common_prefix_event_sha256"]=replay_prefix(a,original,case["card"],fork)
            joint=State(a)
            teacher=Probe(a)
            for item in entries:
                c=dict(case["card"],reply_seed=item["stream"])
                result["rows"].append(dict(stream=item["stream"],branches={}))
                for name,r in item["branches"].items():
                    cue=r["first_inbound_tick"]
                    offer=r["new_offer"]
                    if cue is None or offer is None:
                        result["rows"][-1]["branches"][name]=dict(eligible=False)
                        continue
                    issued=r["controls"][:cue-fork]
                    original_outgoing=original["actions"][:fork]
                    def ready(*,clear=False):
                        joint.restore(a)
                        a.card=c
                        if clear:guarded(a,checker)
                        else:a.sample_clearance(False)
                        for u in issued:
                            a.exchange(u,mpc.FAR_COMMAND)
                    ready()
                    pkt=next((x for x in r["observations"] if x[0]==cue),None)
                    if a.velocity[1]>=-.9:
                        result["rows"][-1]["branches"][name]=dict(eligible=False,native_context_not_yet_inbound=True)
                        continue
                    attempt=dict(eligible=True,first_inbound_cue=cue,observed_p_v=None if pkt is None else pkt[1:7],
                                 q=a.joints.tolist(),matches_logged_encoder=same_trace(a.joints.tolist(),r["encoder_at_inbound"]))
                    if not attempt["matches_logged_encoder"]:
                        raise RuntimeError("Physical action prefix did not reproduce the observed body")
                    try:
                        a.clearance_margin=.0006
                        selected,diag=teacher.search([495340,43,item["stream"],int(name=="blind")],batch,rounds)
                    finally:
                        a.clearance_margin=0.
                    attempt.update(private_screen=diag,num_shortlisted=len(selected),replayed=[])
                    result["rows"][-1]["branches"][name]=attempt
                    old=dict(original,actions=original_outgoing+r["controls"],
                             placements=dict(delivered=original["placements"]["delivered"][:leg]+[offer],
                                   requested=original["placements"]["requested"][:leg]+[r["requested"]]))
                    for p in selected[:2]:
                        ready(clear=True)
                        contacts=[]
                        with contact_observer(a,contacts):
                            physical=trial(a,old,c,p,cue,leg,already=True)
                        v=physical["placements"]["delivered"][leg]
                        if not all(same_trace(v[k],offer[k]) for k in ("far_contact","near_bounce")):
                            raise ValueError("Training intervention changed the physical offer")
                        check=a.clearance_result()
                        log=dict(physical=physical,contacts=contacts,changed_tail_full_cad=check,
                                 known_original_outgoing_retained=True)
                        attempt["replayed"].append(log)
                        if physical["answered"] and not check["penetration_samples"]:
                            ready()
                            for u in physical["actions"][cue:]:
                                a.exchange(u,mpc.FAR_COMMAND)
                            log["independent_exact_tail"]=digest_json(a.events)==digest_json(physical["events"])
                            if not log["independent_exact_tail"]:
                                raise ValueError("Teacher tape did not reproduce")
                            break
                    _write(out,result)
                    print(item["stream"],name,diag,
                          [(p["physical"]["legal"],p["physical"]["answered"]) for p in attempt["replayed"]],flush=True)
        finally:
            a.close();cad.close()
    result["complete"]=True
    _write(out,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--native",type=Path,default=ROOT/"native.json.gz")
    p.add_argument("--output",type=Path,default=ROOT/"coach.json.gz")
    p.add_argument("--batch",type=int,default=96)
    p.add_argument("--rounds",type=int,default=2)
    a=p.parse_args()
    run(a.plan,a.native,a.output,a.batch,a.rounds)
