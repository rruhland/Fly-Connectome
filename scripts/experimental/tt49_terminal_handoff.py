"""Use native retiming to bracket the preparation deadline and export short outside RGB snippets."""

import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance,contact_observer,guarded
from .tt40_long_low import preparation,same_trace
from .tt40_low_feasibility import replay_prefix
from .tt44_paired_credit import State
from .tt48_terminal_repair import ARCHIVE,terminal
from .tt49_terminal_blind import PROOF,ROOT

CUTS=(4016,4030,4040,4048,4056,4062)
STREAMS=(1062985728,191026792)
FRAME_START,FRAME_END=4024,4064


def pass_action(a,tape,first,stop):
    for cmd in tape[first:stop]:
        a.exchange(cmd,mpc.FAR_COMMAND)


def case_for(d,k):
    return next(r for r in d["rows"] if r["stream"]==k)


def run(output,clips,native,cuts):
    doc=_read(ARCHIVE); capture=_read(native)
    target,old=terminal(doc)
    p=_read(PROOF)
    lesson=next(r for r in p["rows"] if r["physical"]["answered"] and r["exact_native_replay"])
    lesson_tape=lesson["physical"]["actions"]
    fork=capture["anchor"]
    t0=p["fork_tick"]; late=p["search_tick"]
    dt=mpc.DT; leg=target["leg"]
    result=dict(schema="tt49-observer-ready-deadline-1",source_sha256=_hash(ARCHIVE),
                known_teaching_sha256=_hash(PROOF),blind_source_sha256=_hash(native),
                clocks=dict(far=target["original_offer"]["far_contact"]["time"],first_inbound_tick=4062,
                            first_table_significant_rebound=late*dt,
                            proved_contact=lesson["contacts_after_fork"][1]["time_s"]),
                cuts=list(cuts),states=[],camera_clips=[])
    with mpc.centered(doc["ready"]):
        a=mpc.arena();cam,rig,stride,lo,hi=mpc.resources(a)
        cad=mpc.arena(detail="cad")
        check=CourtClearance(cad.model,articulated=True)
        try:
            axes=a._arm_ids[1][2]
            result["original_prefix_event_sha256"]=replay_prefix(a,old,target["card"],fork)
            initial=State(a)
            frames_left=[]; frames_right=[]; keys=[]
            for source in STREAMS:
                pcap=case_for(capture,source)
                baseline=pcap["branches"]["stock"]
                commands=old["actions"][:fork]+baseline["controls"]
                initial.restore(a); a.card=dict(target["card"],reply_seed=source)
                received=baseline["new_offer"]
                far_tick=int(np.ceil(received["far_contact"]["time"]/dt-1e-7))
                last=FRAME_END if FRAME_END<len(commands) else len(commands)
                take=[]
                for tick in range(fork,last):
                    if FRAME_START<=tick and tick%2==0:
                        images=cam.capture()
                        frames_left.append(images["left"].copy())
                        frames_right.append(images["right"].copy())
                        keys.append([source,tick])
                        take.append(tick)
                    a.exchange(commands[tick],mpc.FAR_COMMAND)
                result["camera_clips"].append(dict(stream=source,ticks=take,last_frame=last,
                        native_far_contact_tick=far_tick,height_at_deep=pcap["actual_projected_m"],
                        actual_lateral_m=received["receive_line"]["xyz"][0],
                        native_reached_lane=received["observed_lane"],actual_legal=baseline["legal"]))
            np.savez_compressed(clips,left=np.asarray(frames_left),right=np.asarray(frames_right),key=np.asarray(keys))
            result["camera_file_sha256"]=_hash(clips)
            _write(output,result)
            print("saved clips",len(keys),flush=True)

            # One native replay snapshots every handoff clock, unchanged before
            # its designated transition away from the recorded stock action.
            initial.restore(a)
            pass_action(a,old["actions"],fork,t0)
            baseline=State(a)
            stock_states={}
            prior=t0
            for tick in sorted(cuts):
                pass_action(a,old["actions"],prior,tick)
                stock_states[tick]=State(a)
                prior=tick

            # Exact original teacher reference suffix before every motor tick.
            baseline.restore(a)
            for u in lesson_tape[t0:late]:
                a.exchange(u,mpc.FAR_COMMAND)
            expected_q=a.joints.copy()
            ref_suffix=[]
            for u in lesson_tape[late:]:
                a.exchange(u,mpc.FAR_COMMAND)
                ref_suffix.append(a.data.ctrl[axes].copy())

            offer=target["original_offer"]
            for tick in cuts:
                state=stock_states[tick]
                state.restore(a)
                guarded(a,check)
                q=a.data.ctrl[axes].copy()
                prep=preparation(q[:5],np.asarray(lesson["port"]["waypoint"]),np.asarray(lesson["port"]["approach"]),
                                 late-tick,q[-1],brisk=True)
                tape=list(old["actions"][:tick])
                contacts=[]
                velocity_clock=(late-tick)*dt
                # All conditions are diagnostically privilege scheduled, not a
                # real-time detector. The known stroke motors stay saturated.
                with contact_observer(a,contacts):
                    for ref in prep:
                        u=np.clip((ref-a.data.ctrl[axes])/stride,-.8,.8)
                        a.exchange(u,mpc.FAR_COMMAND);tape.append(u.tolist())
                        if a.terminated or a.truncated: break
                    error=float(np.max(abs(a.joints-expected_q)))
                    if not (a.terminated or a.truncated):
                        for ref in ref_suffix:
                            u=np.clip((ref-a.data.ctrl[axes])/stride,-.8,.8)
                            a.exchange(u,mpc.FAR_COMMAND);tape.append(u.tolist())
                            if a.terminated or a.truncated or a.rival_returns>=leg+2:break
                got=a.point_placements()["delivered"]
                found=got[leg] if len(got)>leg else None
                eq=bool(found and all(same_trace(found[k],offer[k]) for k in ("far_contact","near_bounce")))
                passed=bool(found and found["near_landed"] and eq)
                read=dict(tick=tick,seconds=tick*dt,available_before_taught_strike_start_s=velocity_clock,
                          preparation_encoder_max_error_rad=error,kept_same_offer=eq,
                          landed=passed,answered=bool(passed and len(got)>leg+1),changed_tail_clearance=a.clearance_result(),
                          contacts=contacts,events_sha256=digest_json(a.events))
                if passed:
                    read["controls_after_cut"]=tape[tick:]
                result["states"].append(read)
                _write(output,result)
                print(tick,passed,read["answered"],round(error,3),read["changed_tail_clearance"]["penetration_samples"],
                      flush=True)
        finally:
            cam.close();a.close();cad.close()
    result["complete"]=True
    _write(output,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"handoff.json")
    p.add_argument("--clips",type=Path,default=ROOT/"outside-left-right.npz")
    p.add_argument("--native",type=Path,default=ROOT/"native.json.gz")
    p.add_argument("--cuts",nargs="+",type=int,default=CUTS)
    a=p.parse_args()
    run(a.output,a.clips,a.native,a.cuts)
