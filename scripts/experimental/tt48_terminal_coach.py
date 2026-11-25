"""Bound the terminal's motion ceiling once from its actual observed-time body state."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt40_low_feasibility import Probe, replay_prefix, trial
from .tt48_terminal_repair import ARCHIVE, ROOT, terminal

# First credible optical inbound fit from the independent camera replay.
CUE = 4062


def assess(output,batch,rounds):
    d=_read(ARCHIVE)
    case,stock=terminal(d)
    tick=CUE
    doc=dict(role="Privileged finite native diagnostic, not a deployed or scored learner",
             source_sha256=_hash(ARCHIVE),case=case,clock=tick,batch=batch,rounds=rounds,rows=[])
    ready=np.asarray(d["ready"])
    with mpc.centered(ready):
        a=mpc.arena()
        try:
            replay_prefix(a,stock,case["card"],tick)
            before=process_time()
            p=Probe(a)
            try:
                a.clearance_margin=.0006
                shortlist,notes=p.search([481614,43],batch,rounds)
            finally:
                a.clearance_margin=0.
            doc.update(motor_search=notes,cpu_s=process_time()-before,
                       body_q=a.joints.tolist(),found=len(shortlist))
            _write(output,doc)
            print("43",notes,flush=True)
            if shortlist:
                cad=mpc.arena(detail="cad")
                check=CourtClearance(cad.model,articulated=True)
                try:
                    for chosen in shortlist[:2]:
                        c=case["card"]
                        a.launch(stock["shot"],card=c)
                        a.partner.reset(seed=c["seed"])
                        guarded(a,check)
                        hits=[]
                        with contact_observer(a,hits):
                            for u in stock["actions"][:tick]:
                                a.exchange(u,mpc.FAR_COMMAND)
                            events=[e for e in stock["events"] if e["time"]<=a.data.time+1e-8]
                            if digest_json(a.events)!=digest_json(events):
                                raise ValueError("Original measured launch history changed")
                            result=trial(a,stock,c,chosen,tick,case["leg"],already=True)
                        verify_clearance(a)
                        record=dict(physical=result,clearance=a.clearance_result(),native=hits[-4:])
                        doc["rows"].append(record)
                        _write(output,doc)
                        print(result["legal"],result["answered"],record["clearance"]["penetration_samples"],flush=True)
                        if result["answered"] and record["clearance"]["penetration_samples"]==0:
                            break
                finally:
                    cad.close()
        finally:
            a.close()
    doc["complete"]=True
    _write(output,doc)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"terminal-native-coach.json.gz")
    p.add_argument("--batch",type=int,default=96)
    p.add_argument("--rounds",type=int,default=3)
    a=p.parse_args()
    assess(a.output,a.batch,a.rounds)
