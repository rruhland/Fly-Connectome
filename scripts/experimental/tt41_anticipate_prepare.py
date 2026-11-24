"""Can the known low-ball movement still work when a causal short-outbound cue starts it?"""

import argparse
from pathlib import Path

import numpy as np

from . import tt41_deadline as limits
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_cue_vision import quality
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, guarded
from .tt40_long_low import same_trace
from .tt41_anticipate import OUT, outbound_projection
from .tt41_anticipate_reveal import STOCK, WITNESS
from .tt41_deadline_warm import received, reference_tape


def interpolate(refs,time_in_ticks):
    i = int(np.floor(time_in_ticks))
    j = min(len(refs)-1,i+1)
    return refs[i]+(time_in_ticks-i)*(refs[j]-refs[i])


def control(a,s,rig,old,c,leg,fork,deadline,ref_pack,clearance=None):
    """Practice timing assay. The one physically earned motor example is assigned in advance."""
    refs,stride,lo,hi = ref_pack
    axis = a._arm_ids[1][2]
    dt = a.control_dt
    a.sample_clearance(False)
    a.launch(old["shot"],card=c)
    a.partner.reset(seed=c["seed"])
    track=Vision(rig,dt,raw=True)
    seen_inward=False
    active=None
    audit_time=None
    tape, observations = [],[]
    motor_boundary = max(0,fork-18)
    for k in range(len(refs)-1+40):
        if active is None and k%2==0 and motor_boundary<=k<deadline:
            frames=s.capture()
            track.update(k,frames,a.joints)
            f=track.eye.latest
            if quality(f,k*dt):
                p=f.position
                if p[1]<-.76 and f.velocity[1]<-1.15:
                    seen_inward=True
                state=np.r_[p,f.velocity]
                b=outbound_projection(state,f.bounced)
                if b is not None and -.16>p[1]:
                    observations.append(dict(tick=k,estimated_far_bounce_y=b[1],outgoing_y=p[1],
                                             tracked_prior_inward=seen_inward))
                    if seen_inward and .05<b[1]<.65:
                        active=k
                        if active<fork:
                            # The camera did not wait for a privileged fork, so never substitute a
                            # known pattern before this pilot's already-issued near contact is over.
                            return dict(active=active,unexpected_early=True,observations=observations)
                        if clearance is not None:
                            audit_time=float(a.data.time)
                            guarded(a,clearance)
        if active is None:
            u=np.asarray(old["actions"][k]) if k<len(old["actions"]) else np.zeros_like(stride)
        else:
            t=k+1
            if t<=deadline:
                source_ticks=fork+(deadline-fork)*(t-active)/max(1,deadline-active)
            else:
                source_ticks=min(len(refs)-1,t)
            target=interpolate(refs,source_ticks)
            q=a.data.ctrl[axis].copy()
            r=limits.bound(target[None,None,:],q,np.asarray(tape[-1]),stride,lo,hi)[0,0]
            u=np.clip((r-q)/stride,-limits.RATE,limits.RATE)
        tape.append(u.tolist())
        _,_,end,cap,_=a.exchange(u,FAR_COMMAND)
        if end or cap or a.rival_returns>=leg+2:
            break
    measure=received(a,leg)
    expected=old["placements"]["delivered"][leg]
    incoming=same_trace(measure.get("far"),expected["far_contact"]) and same_trace(
        measure.get("bounce"),expected["near_bounce"])
    report=dict(active=active,original_prep_tick=fork,desired_ready_tick=deadline,
                issued_delay_s=None if active is None else (active-fork)*dt,
                observations=observations,incoming_unchanged=incoming,outcomes=measure,
                events_sha256=digest_json(a.events),actions=tape,events=a.events,
                placements=a.point_placements(),clearance=a.clearance_result(),end_s=float(a.data.time))
    if audit_time is not None:
        samples=round((a.data.time-audit_time)/a.model.opt.timestep)
        if report["clearance"]["samples"]!=samples:
            raise ValueError("CAD missed some states after the causal intervention")
        report["cad_start_s"]=audit_time
    return report


def run(output,witness_path=WITNESS):
    witness, native = _read(witness_path),_read(STOCK)
    docs={g["origin"]:_read(witness_path.with_name(g["origin"])) for g in witness["groups"]}
    data={(c["id"],c["seed"],c["reply_seed"]):(c,r) for c,r in zip(native["cards"],native["rows"],strict=True)}
    report=dict(role="Per-case recalled motor timing window, not a release-held controller",
                sources={str(p):_hash(p) for p in (witness_path,STOCK)},rows=[],complete=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with centered(native["ready"]):
        a=arena()
        cad=arena(detail="cad")
        clearance=CourtClearance(cad.model,articulated=True)
        camera,rig,*_=resources(a)
        seen=set()
        try:
            for g in witness["groups"]:
                key=tuple(g["card"][k] for k in ("id","seed","reply_seed"))
                c, old = data[key]
                leg = g["leg"]
                if (key,leg) in seen:
                    continue
                seen.add((key,leg))
                d=next(r for r in docs[g["origin"]]["rows"] if r["card"]==g["card"] and r["leg"]==leg)
                win=next(x for x in d["wins"] if digest_json(x["events"])==g["events_sha256"])
                pack=reference_tape(a,win,c)
                result=control(a,camera,rig,old,c,leg,g["command_prefix_end_tick"],
                               g["preparation_end_tick"],pack,clearance)
                report["rows"].append(dict(id=c["id"],card=c,leg=leg,original_tape=g["origin"],
                                           original_answered=win["answered"],result=result))
                print(c["id"],result["active"],result.get("issued_delay_s"),result.get("outcomes",{}).get("legal"),
                      result.get("outcomes",{}).get("answered"),flush=True)
                _write(output,report)
        finally:
            camera.close()
            a.close()
            cad.close()
    report["complete"]=True
    _write(output,report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUT/"identified-preparation.json.gz")
    p.add_argument("--witness",type=Path,default=WITNESS)
    args=p.parse_args()
    run(args.output,args.witness)
