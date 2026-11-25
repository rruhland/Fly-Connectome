"""Test a single achieved ready hub in native continuous points, with a privileged coach."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_motor_clinic import landing_error
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt38_motorcoach import AIM, SIZES, answerable, schedules
from .tt40_long_low import physical_finish
from .tt40_low_feasibility import Faces, SOURCE, replay_prefix
from .tt43_stance import OUT, RECORD
from .tt41_shallow_model import impulse

# One central short-notice normal and two low arrivals. No treatment-score selection.
CASES = ((14000, 13), (4322, 5), (14000, 1), (8302, 3), (14843, 1))


def fork_ticks(old, w, dt):
    outgoing = next(e for e in reversed(old["events"]) if e["time"] < w["far"]["time"]
                    and e.get("collider") == "learner_racket")
    last_out = int(np.ceil(outgoing["time"] / dt - 1e-8)) + 3
    # 83 ms after far separation. The recorded low cameras needed 58-77 ms to
    # expose an inbound fit. This clock is preassigned solely to test the plant.
    start = int(np.ceil((w["far"]["time"] + .0833) / dt - 1e-8))
    return last_out, start


def preload(a, old, card, hub, fork, start, guard=None):
    tape = list(old["actions"][:fork])
    if guard is not None:
        a.launch(old["shot"], card=card)
        a.partner.reset(seed=card["seed"])
        guarded(a, guard)
        for cmd in tape:
            a.exchange(cmd, FAR_COMMAND)
    else:
        replay_prefix(a, old, card, fork)
    axes = a._arm_ids[1][2]
    stride = a._joint_increments[axes] * (a.control_dt/a.model.opt.timestep)
    for _ in range(fork, start):
        cmd = np.clip((hub - a.data.ctrl[axes]) / stride, -.8, .8)
        a.exchange(cmd, FAR_COMMAND)
        tape.append(cmd.tolist())
    expected = [e for e in old["events"] if e["time"] <= float(a.data.time)+1e-8]
    if digest_json(a.events) != digest_json(expected):
        raise ValueError("Preparation changed the prefix events")
    return tape


def direct_swing(poses, velocities, etas, params, mode, dt, frames):
    """Finish most displacement first and add the contact drive only near impact."""
    elapsed = (1+np.arange(frames)) * dt
    lead = .040 + params[:,10]
    phase = elapsed[None,:] - etas[:,None] + lead[:,None]
    # The old shallow suggestions pull back with the full strike velocity for
    # 150 ms. Do not spend scarce reach travel on an entire backswing.
    span = np.clip(phase, -np.array([.025,.055,.090])[mode][:,None], .087)
    speed = (velocities + params[:,5:10]) * np.array([.20,.63,1.])[mode][:,None]
    q = poses[:,None,:] + params[:,None,:5] + span[:,:,None]*speed[:,None,:]
    kick = np.clip(phase / .08, 0., 1.)
    q[:,:,3:5] += kick[:,:,None]*params[:,None,11:]
    return np.concatenate((q,np.full((*q.shape[:2],1),-.1)),axis=2)


def dynamic_options(a, engine, ik, poses, w, batch, rounds, seed, *, memories=(), schedule="direct"):
    options = []
    now = a.data.time
    for row in poses:
        eta = row["eta"] if w is None else row["far"] + w["far"]["time"] - now
        q = np.clip(row["q"], ik.low, ik.high)
        if not .045 < eta < .61 or np.max(abs(q-a.joints[:5]))/4.16 > eta-.012:
            continue
        p = row["p"]
        if "v" in row:
            v = row["v"]
        else:
            v = (w["table"]["velocity"] if schedule=="original"
                 else np.r_[w["table"]["velocity"][:2], 0.])
        n, u = impulse(np.asarray(p),np.asarray(v),4.4,.13)
        _, normal, J, N = ik.pose(q, jacobian=True)
        # The signed face is already in q. Joint rates and small pose
        # corrections now determine the physical departure and landing.
        dq = np.linalg.lstsq(np.r_[J,.065*N,.023*np.eye(5)], np.r_[u,np.zeros(8)],rcond=None)[0].clip(-5.8,5.8)
        options.append((row,q,dq,eta))
    if not options:
        return None, []
    # Keep both contact families while limiting search cost.
    options.sort(key=lambda z:np.max(abs(z[1]-a.joints[:5]))/4.16-z[-1])
    groups = [[z for z in options if z[0]["face"]==face][:10] for face in (1,-1)]
    options = [z for g in groups for z in g]
    q, dq, eta = [np.array([z[k] for z in options]) for k in (1,2,3)]
    dt=a.control_dt
    cutoff=float(eta.max()+.11)
    cap=round((cutoff+1.0)/dt)
    rng=np.random.default_rng(seed)
    key=np.resize(np.arange(len(q)),batch)
    size=SIZES*np.r_[[.53]*5,[.63]*5,[.61,.65,.65]]
    z=(2*qmc.Sobol(13,scramble=True,seed=int(rng.integers(2**31))).random_base2(int(np.ceil(np.log2(batch))))[:batch]-1.)*size
    z[:len(q)*(2 if schedule=="direct" else 1)]=0.
    mode=(np.arange(batch)//len(q))%3
    if schedule=="original":
        mode[:]=3
    if memories:
        for i in range(min(len(q),batch//2)):
            row=options[key[i]][0]
            ref=np.r_[row["p"],row["v"]]
            eligible=[(np.linalg.norm((np.r_[r["p"],r["v"]]-ref)/[.16,.20,.10,1.,1.,1.]),r)
                      for r in memories if row["face"]==r["face"]]
            if eligible:
                score, learned=min(eligible,key=lambda p:p[0])
                if score<2.:
                    z[i]=learned["parameters"]; mode[i]=learned["drive_mode"]
    found=[]; archive=[]; trace=[]
    for stage in range(rounds):
        if schedule=="original":
            goals=schedules(q[key],dq[key],eta[key],z,dt,cap)
        else:
            goals=direct_swing(q[key],dq[key],eta[key],z,np.minimum(mode,2),dt,cap)
            deep=mode==3
            if deep.any():
                goals[deep]=schedules(q[key[deep]],dq[key[deep]],eta[key[deep]],z[deep],dt,cap)
        f=np.zeros((batch,9))
        try:
            a.clearance_margin=.0006 # private candidate screen only
            rates,res=engine.trajectories(goals,AIM,cutoff,feedback=f)
        finally:
            a.clearance_margin=0.
        legal=np.isfinite(res[:,0])
        valid=legal & (res[:,5]>.012) & (answerable(res)<5.8)
        rank=np.where(legal,answerable(res)-11.,np.where(f[:,1]>.5,landing_error(f)+3.*f[:,8],19.+39.*f[:,0]))
        for j in np.flatnonzero(valid):
            n=min(len(rates[j]),round((res[j,1]+.10)/dt))
            found.append(dict(tape=rates[j,:n].tolist(),cost=float(answerable(res[j])),quality=res[j].tolist(),
                       feedback=f[j].tolist(),seed=options[key[j]][0],parameters=z[j].tolist(),stage=stage,
                       goals=goals[j,:n].tolist(), drive_mode=int(mode[j])))
        trace.append(dict(stage=stage,tried=batch,contact=int(sum(f[:,1]>.5)),legal=int(sum(legal)),
                    usable=int(sum(valid)),closest_m=float(np.min(f[:,0]))))
        if found:
            break
        archive.extend(zip(rank,key.copy(),z.copy(),mode.copy(),strict=True))
        top=sorted(archive,key=lambda p:p[0])[:12]
        i=rng.integers(len(top),size=batch)
        key=np.array([top[k][1] for k in i]);z=np.array([top[k][2] for k in i])+rng.normal(size=z.shape)*size*.53
        mode=np.array([top[k][3] for k in i])
        z=np.clip(z,-2.7*size,2.7*size)
        key[:len(top)]=[t[1] for t in top];z[:len(top)]=np.array([t[2] for t in top]);mode[:len(top)]=[t[3] for t in top]
    return min(found,key=lambda x:x["cost"]) if found else None,trace


def study(output, chart_path, witnesses_path, batch, rounds, cases, schedule):
    d=_read(witnesses_path);chart=_read(chart_path)
    old=_read(SOURCE); record=_read(RECORD.with_name("random-38-tape.json.gz"))
    hub=np.r_[chart["routes"]["either_far"][1]["q"],d["boot"][-1]]
    w=d["regular"]+[r for r in d["low"] if r["face"]=="negative"]
    r=dict(schema="tt43-common-hub-physical-forks-1",boot=d["boot"],hub=hub.tolist(),rows=[],
           schedule=schedule,chart_sha256=_hash(chart_path),witness_sha256=_hash(witnesses_path),source_sha256=_hash(SOURCE),
           record_sha256=_hash(RECORD.with_name("random-38-tape.json.gz")),
           scope="Scheduled matched native feasibility on known development shots. Not a visual policy or qualification.")
    with centered(d["boot"]):
        a=arena()
        cad=arena(detail="cad")
        engine=Rollout(a,1)
        ik=Faces(a)
        guard=CourtClearance(cad.model,articulated=True)
        try:
            for ident,leg in cases:
                if ident==14000:
                    card=record["cards"][0];orig=record["rows"][0]
                else:
                    i=next(i for i,c in enumerate(old["cards"]) if c["id"]==ident)
                    card=old["cards"][i];orig=old["rows"][i]
                idx=next(i for i,x in enumerate(w) if x["id"]==ident and x["leg"]==leg)
                witness=w[idx]
                fork,tick=fork_ticks(orig,witness,a.control_dt)
                started=process_time()
                preload(a,orig,card,hub,fork,tick)
                posture=a.joints.copy()
                chosen,tried=dynamic_options(a,engine,ik,chart["banks"][idx],witness,batch,rounds,[430114,ident,leg],schedule=schedule)
                item=dict(id=ident,leg=leg,card=card,fork=fork,start=tick,q=posture.tolist(),
                          hub_distance_rad=float(np.max(abs(posture-hub))),stock_legal=orig["placements"]["delivered"][leg]["near_landed"],
                          trials=tried)
                if chosen is not None:
                    contact=[]
                    with contact_observer(a,contact):
                        prefix=preload(a,orig,card,hub,fork,tick,guard)
                        result=physical_finish(a,orig,card,leg,prefix,chosen)
                    verify_clearance(a)
                    item.update(physical=result,clearance=a.clearance_result(),contacts=contact)
                    saved=(digest_json(result["events"]),digest_json(result["placements"]))
                    a.launch(orig["shot"],card=card)
                    a.partner.reset(seed=card["seed"])
                    for cmd in result["actions"]:
                        a.exchange(cmd,FAR_COMMAND)
                    item["independent_event_and_placement_replay"]=(saved==(digest_json(a.events),digest_json(a.point_placements())))
                item["cpu_s"]=process_time()-started
                r["rows"].append(item)
                _write(output,r)
                print(ident,leg,item["hub_distance_rad"],tried,
                        None if chosen is None else (result["legal"],result["answered"],item["clearance"]["penetration_samples"]),
                        flush=True)
        finally:
            a.close();cad.close()
    r["complete"]=True
    _write(output,r)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chart",type=Path,default=OUT/"chart.json.gz")
    p.add_argument("--witnesses",type=Path,default=OUT/"witness-envelope.json.gz")
    p.add_argument("--output",type=Path,default=OUT/"native.json.gz")
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--limit",type=int,default=len(CASES))
    p.add_argument("--schedule",choices=("direct","original"),default="original")
    p.add_argument("--cases",nargs="+",type=int,help="zero-based listed development cases")
    args=p.parse_args()
    cases=CASES[:args.limit] if args.cases is None else [CASES[i] for i in args.cases]
    study(args.output,args.chart,args.witnesses,args.batch,args.rounds,cases,args.schedule)
