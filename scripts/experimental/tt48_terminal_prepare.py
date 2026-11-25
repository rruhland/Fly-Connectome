"""Test whether earlier preparation can return the record's exact final physical offer."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded
from .tt40_long_low import (Workspace, indexed, late_search, physical_finish, plain,
                            preparation, same_trace, select)
from .tt40_low_feasibility import replay_prefix
from .tt44_paired_credit import State
from .tt48_terminal_repair import ARCHIVE, ROOT, terminal


def run(output,batch,rounds,ports):
    archive=_read(ARCHIVE)
    fixed,old=terminal(archive)
    card,offer,leg=(fixed[k] for k in ("card","original_offer","leg"))
    home=np.asarray(archive["ready"])
    doc=dict(role="Privileged earlier-preparation coach on the existing 43rd offer, not a deployable actor",
             source_sha256=_hash(ARCHIVE),case=fixed,batch=batch,rounds=rounds,ports=ports,rows=[])
    started=process_time()
    with mpc.centered(home):
        a=mpc.arena();cad=mpc.arena(detail="cad")
        check=CourtClearance(cad.model,articulated=True)
        try:
            dt=a.control_dt
            fork,late,table=indexed(old["events"],offer,dt)
            event_sha=replay_prefix(a,old,card,fork)
            saved=State(a)
            axes=a._arm_ids[1][2]
            stride=a._joint_increments[axes]*(dt/a.model.opt.timestep)
            c0=a.data.ctrl[axes].copy()
            model=Workspace(cad,home)
            phases=model.windows(table)
            bases,removed=model.support(phases,c0[:5],(late-fork)*dt)
            for q in bases:q["eta"]=float(q["time"]+table["time"]-late*dt)
            got=select([b for b in bases if b["p"][1]>-1.49],ports)
            doc.update(fork_tick=fork,search_tick=late,original_prefix_event_sha256=event_sha,
                       preparation_s=(late-fork)*dt,original_q_at_fork=a.joints.tolist(),
                       routed_poses={str(s):sum(b["face"]==s for b in bases) for s in (-1,1)},
                       failed_pose_filters=len(removed))
            _write(output,doc)
            print("prep",round(doc["preparation_s"],3),len(bases),[(x["face"],round(x["p"][1],3)) for x in got],flush=True)
            expect=[e for e in old["events"] if e["time"]<=late*dt+1e-8]
            engine=Rollout(a,1)
            for k,w in enumerate(got):
                saved.restore(a)
                guarded(a,check)
                q=w["approach"]
                target=preparation(c0[:5],w["waypoint"],q,late-fork,c0[-1],brisk=w["face"]<0)
                tape=list(old["actions"][:fork])
                hits=[]
                item=dict(port=plain(w))
                with contact_observer(a,hits):
                    for v in target:
                        current=a.data.ctrl[axes]
                        u=np.clip((v-current)/stride,-.80,.80)
                        a.exchange(u,mpc.FAR_COMMAND)
                        tape.append(u.tolist())
                        if a.terminated or a.truncated:
                            break
                    same=same_trace(a.events,expect) and not (a.terminated or a.truncated)
                    item.update(preserved_actual_delivery=bool(same),q_actual=a.joints.tolist(),
                                q_error_max_rad=float(np.max(abs(a.joints[:5]-q))))
                    if same:
                        a.clearance_margin=.0006
                        try:
                            selections,diagnostic=late_search(a,engine,w,bases,batch,rounds,[480922,43,k])
                        finally:
                            a.clearance_margin=0.
                        item["native_attempts"]=diagnostic
                        if selections:
                            result=physical_finish(a,old,card,leg,tape,selections[0])
                            item.update(physical=result,contacts_after_fork=hits,changed_tail_clearance=a.clearance_result())
                item.setdefault("changed_tail_clearance",a.clearance_result())
                item["changed_tail_clearance_complete"]=(item["changed_tail_clearance"]["samples"]==
                                                         round((a.data.time-fork*dt)/a.model.opt.timestep))
                doc["rows"].append(item)
                _write(output,doc)
                print(k,same,item.get("native_attempts"),item.get("physical",{}).get("legal"),
                      item["changed_tail_clearance"]["penetration_samples"],flush=True)
                physical=item.get("physical")
                if physical and physical["answered"] and not item["changed_tail_clearance"]["penetration_samples"]:
                    saved.restore(a)
                    a.sample_clearance(False)
                    for u in physical["actions"][fork:]:
                        a.exchange(u,mpc.FAR_COMMAND)
                    item["exact_native_replay"]=(digest_json(a.events)==digest_json(physical["events"])
                                              and digest_json(a.point_placements())==digest_json(physical["placements"]))
                    break
        finally:
            a.close();cad.close()
    doc["cpu_s"]=process_time()-started
    doc["complete"]=True
    _write(output,doc)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"terminal-earlier-prepare.json.gz")
    p.add_argument("--batch",type=int,default=96)
    p.add_argument("--rounds",type=int,default=2)
    p.add_argument("--ports",type=int,default=2)
    a=p.parse_args()
    run(a.output,a.batch,a.rounds,a.ports)
