"""Identify local command sensitivity on scored strokes, then transport to fresh physical replies."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt48_contact_patch as evaluator
from . import tt57_motor_school as school
from . import tt56_central_training as physical
from . import tt59_fewshot_joints as fewshot
from .tt29_rally_physics import BladeIK, Rollout
from .tt37_opponent import draw
from .tt4_transfer_trial import _hash, _read, _write
from .tt44_paired_credit import State, prefix

mpc = school.mpc
ROOT = Path("artifacts/tt67_action_sensitivity")
PLAN, MODEL, SCOUT, OUT = (ROOT / x for x in ("protocol.json", "fit.json.gz", "scouts.json.gz", "held.json.gz"))
SOURCES = tuple(Path("artifacts") / (name + ".json.gz") for name in (
    "tt44_combined/combined", "tt48_transfer/continued-64", "tt42/after48", "tt42/diverse-learning"))
H = school.H
SCAN = 16
MOVES = 3
CAL_WORLDS = 19
RNG = 670751
RATE = .037
GATE_Y = -.88
NEAR = ("easy", "low")


def state4(o):
    raw = o["receive_line"]
    return np.asarray([raw["xyz"][0], school.projected_height(raw), -raw["velocity"][1], raw["velocity"][2]])


def freeze():
    forbidden = {r["id"] for r in _read(school.PLAN)["held"]} | {9060, 8595, 12866}
    old = _read(fewshot.PLAN)
    forbidden.update(r["id"] for r in old["cases"])
    options = []
    for p in SOURCES:
        d = _read(p)
        for i, (c, r) in enumerate(zip(d["cards"], d["rows"])):
            if c["id"] in forbidden or c["profile"] != "reach":
                continue
            for o in r["placements"]["delivered"][1:5]:
                if not o["near_landed"] or not all(o.get(k) for k in ("near_contact", "far_contact", "near_bounce", "receive_line")):
                    continue
                leg = o["leg"]
                if len(r["placements"]["delivered"]) <= leg+1:
                    continue
                near = o["near_contact"]
                radius = float(np.linalg.norm(np.asarray(near["racket"])-near["location"]))
                x, z, *_ = state4(o)
                phase = "low" if .991 <= z <= 1.056 else "easy" if 1.063 <= z <= 1.139 else None
                if phase is None or not .018 < radius < .066 or abs(x)>.205 or near["step"]-o["near_bounce"]["step"] < 11:
                    continue
                g = r["placements"]["requested"][leg]
                if g["draw"] != "random" or g["time_s"] > 9.:
                    continue
                options.append(dict(source=str(p), index=i, id=c["id"], leg=leg, phase=phase, radius=radius,
                                    height=z, x=x, card=c, original=g["target"],
                                    split_tick=int(np.floor(g["time_s"]/mpc.DT+1e-7)), state=state4(o).tolist()))
    rng = np.random.default_rng(RNG)
    pool = [options[j] for j in rng.permutation(len(options))]
    chosen, used = [], set()
    for phase in NEAR * 3:
        possible = [x for x in pool if x["phase"] == phase and x["id"] not in used]
        if not possible:
            continue
        s = min(possible, key=lambda r: (r["split_tick"]//360, r["radius"]//.012))
        used.add(s["id"])
        targets = []
        while len(targets) < SCAN:
            seed = int(rng.integers(1, 2**30))
            target = draw(physical.changed(s["card"], s["leg"], seed), s["leg"], "random", None)[0]
            dx, _, dz, dp = np.array(target)-s["original"]
            if abs(dx) <= .11 and abs(dz) <= .063 and abs(dp) <= .46:
                targets.append(dict(seed=seed, target=list(target)))
        chosen.append(dict(s, targets=targets))
    if len(chosen) != 6:
        raise ValueError("Need three independent central-low and three ordinary interior-score sources")
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(PLAN, dict(schema="tt67-source-action-sensitivity-1", sources={str(p):_hash(p) for p in SOURCES},
                      cases=chosen, scan=SCAN, retained=MOVES, modeled_worlds=CAL_WORLDS,
                      scope="Start from previously scored, answered interior strikes, select ordinary/easier and comparatively low. Source position, body, command effects, and private simulation may enter source-only system identification. Scout natural, independently seeded continuously randomized opponent replies before trying held movements. Holdout inference has only RGB/encoders and the previously learned template. Same physical contact and bounce across methods. Source shift test distinguishes blind phase-warped replay from the estimated direct response law. Tests local transfer on similar deliveries, not novel-release retrieval."))
    print("source strokes", [(s["id"],s["phase"],round(s["height"],3),round(s["radius"],3)) for s in chosen], flush=True)


def design():
    p = _read(PLAN)
    if any(_hash(Path(n)) != v for n,v in p["sources"].items()):
        raise ValueError("Changed source movement tapes")
    return p


def acquire(a, cam, rig, row, start, quit, take):
    q_ref = a._arm_ids[1][2]
    eye = mpc.Vision(rig, a.control_dt, raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye, school.course.OPTIC)
    # All commanded behavior up to the chosen held fork is identical across
    # conditions. The independent opponent request is the only plant change.
    for tick in range(start, quit):
        q = a.joints.copy()
        obs = mpc.observed(eye, tick, cam.capture() if tick%2 == 0 else None, q)
        if take(tick, obs, eye):
            return dict(tick=tick, x=obs, q=q, ref=a.data.ctrl[q_ref].copy())
        if tick >= len(row["actions"]):
            break
        _, _, done, cap, _ = a.exchange(row["actions"][tick], mpc.FAR_COMMAND)
        if done or cap:
            break
    return None


def armed(a, cam, rig, row, case, card):
    root = case["split_tick"]
    prefix(a, case["card"], row, root, None, a._arm_ids[1][2], window=0)
    a.card = card
    # The same clock starts observations for original and held goals. The
    # gate uses only the optical stream; no scout bounce tick is passed in.
    return acquire(a,cam,rig,row,root,root+115,
        lambda tick,x,eye: x is not None and x[4] < -1.2 and x[5] > .48 and -1.04 < x[1] <= GATE_Y)


def do_scout():
    p = design()
    result = dict(protocol=_hash(PLAN), rows=[], complete=False)
    docs = {str(f): _read(f) for f in SOURCES}
    with school.enabled(cameras=False):
        for c in p["cases"]:
            d = docs[c["source"]]
            r = d["rows"][c["index"]]
            leg = c["leg"]
            found = []
            with mpc.centered(d["ready"]):
                a = mpc.arena(width=64,height=48)
                try:
                    cut = c["split_tick"]
                    prefix(a,c["card"],r,cut,None,a._arm_ids[1][2],window=0)
                    if len(a.partner.requests) != leg:
                        raise ValueError("Reply already selected before intervention")
                    root = State(a)
                    for v in c["targets"]:
                        root.restore(a)
                        a.card = physical.changed(c["card"], leg, v["seed"])
                        for u in r["actions"][cut:cut+104]:
                            _,_,done,capped,_ = a.exchange(u,mpc.FAR_COMMAND)
                            if done or capped or leg+1 in a.crossings:
                                break
                        place = a.point_placements()
                        requests = place["requested"]
                        if len(requests)>leg and requests[leg]["target"] != v["target"]:
                            raise ValueError("The original opponent draw law changed")
                        deliveries = place["delivered"]
                        o = deliveries[leg] if len(deliveries)>leg else None
                        valid = bool(o and o.get("far_contact") and o.get("near_bounce") and o.get("receive_line"))
                        # Retain only the incoming half. Do not use or export
                        # the outcome under these unmodified archived strokes.
                        observed = {k:o[k] for k in ("leg","far_contact","near_bounce","receive_line")} if valid else None
                        found.append(dict(v, offered=observed, state=state4(o).tolist() if valid else None))
                finally:
                    a.close()
            scores = []
            origin = np.asarray(c["state"])
            for j,v in enumerate(found):
                if v["state"] is None:
                    continue
                shift = np.asarray(v["state"])-origin
                distance = float(np.linalg.norm(shift[:2]))
                if .012 <= distance <= .073 and abs(shift[0])<.060 and abs(shift[1])<.058 and abs(shift[2])<.65:
                    scores.append((j,shift,distance))
            # Select two materially displaced directions and one easier
            # independent small variation, without performing a receiving trial.
            selected = []
            if scores:
                selected.append(min(scores,key=lambda q: abs(q[2]-.018)))
                left = [s for s in scores if s[0] != selected[0][0]]
                if left:
                    selected.append(min(left,key=lambda q: abs(q[2]-.044)))
                while len(selected)<MOVES and left:
                    left = [s for s in left if s[0] not in {r0[0] for r0 in selected}]
                    if left:
                        selected.append(max(left,key=lambda q: min(np.linalg.norm(q[1][:2]-s[1][:2]) for s in selected)))
            result["rows"].append(dict(source=c["id"], candidates=found, retained=[s[0] for s in selected]))
            result["complete"] = len(result["rows"]) == len(p["cases"])
            _write(SCOUT,result)
            print(c["id"], "held shifts", [(found[j]["seed"],np.round(s[:2],3).tolist()) for j,s,_ in selected],flush=True)


def projected(x, y):
    t = np.clip((x[1]-y)/max(1.25,-x[4]), .023, .39)
    return np.array([x[0]+t*x[3], x[2]+t*x[5]-4.905*t*t, t])


def smooth(ticks):
    p = np.clip(school.PHASE/max(3.,ticks*.85),0.,1.)
    return (p*p*(3.-2.*p))[:,None]


def estimate(a, plant, goals, basis, tick, ref, rng):
    # At a pre-impact clock, blade center and normal remain smooth even
    # when later contact either misses or goes out. All public transport
    # uses this pre-impact calibration, not an oracle scoring gradient.
    snapshot = State(a)
    axes, dofs = a._arm_ids[1][2], a._near_dofs
    pos = a.model.jnt_qposadr[a.model.actuator_trnid[axes,0]]
    kin = BladeIK(a)
    pairs = []
    value0 = None
    motors = np.r_[np.eye(5)*RATE,-np.eye(5)*RATE,np.zeros((1,5))]
    motions = np.asarray([goals + basis*np.r_[d,0.] for d in motors])
    for k in range(CAL_WORLDS):
        snapshot.restore(a)
        b = np.zeros(5) if k==0 else rng.normal(size=5)*np.array([.013,.014,.017,.017,.020])
        w = np.zeros(5) if k==0 else rng.normal(size=5)*.17
        a.data.qpos[pos[:5]] += b
        a.data.qvel[dofs[:5]] += w
        # Only manipulator kinematics at the pre-impact time constrain J.
        # All original input perturbations leave the actual training test
        # branch unchanged. Exact state mutations never reach the actor.
        tr = np.full((*motions.shape[:2],17),np.nan)
        plant.trajectories(motions,school.AIM,.45,traces=tr)
        features = []
        for q in tr[:,tick,6:11]:
            if not np.isfinite(q).all():
                features.append(np.full(5,np.nan))
                continue
            center, n, *_ = kin.pose(q)
            features.append(np.r_[center,n[[0,2]]])
        f = np.asarray(features)
        if k==0:
            value0 = f[-1]
        if np.isfinite(f).all():
            matrix = ((f[:5]-f[5:10])/(2*RATE)).T
            pairs.append((matrix,b,w,f[-1]))
    snapshot.restore(a)
    if len(pairs)<8 or not np.isfinite(value0).all():
        raise ValueError("Could not identify a pre-contact response from original trials")
    j = np.median([z[0] for z in pairs],axis=0)
    features = np.asarray([np.r_[z[1],z[2]] for z in pairs])
    Y = np.asarray([z[3] for z in pairs])-value0
    ridge = np.linalg.solve(features.T@features+.006*np.eye(10),features.T@Y).T
    return j, ridge, len(pairs), value0


def fit():
    d=design()
    docs={str(f):_read(f) for f in SOURCES}
    result=dict(protocol=_hash(PLAN), rows=[], complete=False)
    with school.enabled():
        for c in d["cases"]:
            doc=docs[c["source"]]
            row=doc["rows"][c["index"]]
            expected=row["placements"]["delivered"][c["leg"]]
            with mpc.centered(doc["ready"]):
                a=mpc.arena()
                cam,rig,stride,lo,hi=mpc.resources(a)
                try:
                    obs=armed(a,cam,rig,row,c,c["card"])
                    if obs is None:
                        print("No public optical source gate",c["id"],flush=True)
                        result["rows"].append(dict(id=c["id"], skipped="source optical gate"))
                        _write(MODEL,result)
                        continue
                    time=(expected["near_contact"]["time"]-a.data.time)/mpc.DT
                    pre=int(np.clip(np.floor(time)-2,3,16))
                    ref=obs["ref"]
                    rates=mpc.tail(np.asarray(row["actions"][obs["tick"]:obs["tick"]+H]),H)
                    goals=school.atlas.reference_path(ref,rates,stride,lo,hi)[1:]
                    rng=np.random.default_rng([RNG,c["id"]])
                    j,dyn,n,blade=estimate(a,Rollout(a,1),goals,smooth(pre),pre,ref,rng)
                    # The contact line comes from the observed incoming
                    # history extrapolated from the fixed public gate.
                    # Use source exact contact Y only for fitting its local
                    # waveform. The number is frozen per prior case.
                    y=float(expected["near_contact"]["ball"][1])
                    row0=dict(id=c["id"], obs={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in obs.items()},
                              actual_contacts_elapsed_ticks=float(time), probe=pre, contact_line=y,
                              shape=(goals-ref).tolist(), J=j.tolist(), body_jac=dyn.tolist(), worlds=n,
                              baseline_blade=blade.tolist(),
                              action_singular_values=np.linalg.svd(j[:3],compute_uv=False).tolist())
                    result["rows"].append(row0)
                    print(c["id"], pre, n, np.round(row0["action_singular_values"],4),flush=True)
                finally:
                    cam.close()
                    a.close()
            result["complete"]=len(result["rows"])==len(d["cases"])
            _write(MODEL,result)


def policy(m, obs, mode):
    old=m["obs"]
    a,b=projected(np.asarray(old["x"]),m["contact_line"]),projected(obs["x"],m["contact_line"])
    timing=np.clip(a[2]/b[2],.76,1.28)
    intent=school.warp(np.asarray(m["shape"]),timing,0.)
    eta=m["probe"]/timing
    filt=smooth(eta)
    if mode == "forward":
        delta=b-a
        # Current encoder and controller lag, not the exact qvel.
        q=np.asarray(obs["q"]); ref=np.asarray(obs["ref"])
        match=np.asarray(old["q"]); r0=np.asarray(old["ref"])
        difference=np.r_[q[:5]-match[:5],5.*((ref-q)-(r0-match))[:5]]
        d=np.array([delta[0],0.,delta[1],0.,0.])
        J=np.asarray(m["J"])
        body=np.asarray(m["body_jac"])@difference
        scale=np.array([1.,.75,1.,.14,.14])
        A=scale[:,None]*J
        gain=np.linalg.solve(A.T@A+.013*np.eye(5),A.T@(scale*(d-body)))
        intent+=filt*np.r_[np.clip(gain,-.145,.145),0.]
    else:
        gain=np.zeros(5)
    return intent,dict(observed_displacement=(b-a).tolist(), timing=float(timing), gain=gain.tolist())


def held():
    d,s,m=design(),_read(SCOUT),_read(MODEL)
    if not s["complete"] or not m["complete"] or s["protocol"] != _hash(PLAN) or m["protocol"] != _hash(PLAN):
        raise ValueError("Unsealed held scout or fit")
    docs={str(f):_read(f) for f in SOURCES}
    result=dict(protocol=_hash(PLAN), fit=_hash(MODEL), scouts=_hash(SCOUT), rows=[],complete=False)
    if OUT.exists():
        previous = _read(OUT)
        if any(previous[k] != result[k] for k in ("protocol", "fit", "scouts")):
            raise ValueError("Cannot resume a different frozen local sensitivity experiment")
        result = previous
    with school.enabled():
        cad=mpc.arena(width=64,height=48,detail="cad")
        guard=school.CourtClearance(cad.model,articulated=True)
        try:
            for ci,(c,saved,lesson) in enumerate(zip(d["cases"],s["rows"],m["rows"],strict=True)):
                if ci < len(result["rows"]):
                    if result["rows"][ci]["id"] != c["id"]:
                        raise ValueError("Held receipt is not an ordered prefix of this plan")
                    continue
                doc=docs[c["source"]]; row=doc["rows"][c["index"]]; leg=c["leg"]
                source=dict(seed=None,target=c["original"],offered=row["placements"]["delivered"][leg],state=c["state"])
                orders=[source]+[saved["candidates"][j] for j in saved["retained"]]
                trials=[]
                if "skipped" not in lesson:
                    with mpc.centered(doc["ready"]):
                        a=mpc.arena()
                        cam,rig,stride,lo,hi=mpc.resources(a)
                        try:
                            for order in orders:
                                card=c["card"] if order["seed"] is None else physical.changed(c["card"],leg,order["seed"])
                                target=order["offered"]
                                a.sample_clearance(False)
                                obs=armed(a,cam,rig,row,c,card)
                                if obs is None:
                                    trials.append(dict(order=order,no_gate=True))
                                    continue
                                snapshot=State(a)
                                outcomes={}
                                physical_row=dict(row,placements=dict(row["placements"]))
                                actual_leg=row["placements"]["delivered"].copy()
                                actual_leg[leg]=target
                                physical_row["placements"]["delivered"]=actual_leg
                                for mode in ("replay","forward"):
                                    relative,meta=policy(lesson,obs,mode)
                                    ref=obs["ref"]
                                    targets=np.clip(ref+relative,lo,hi)
                                    sent=ref.copy()
                                    u=[]
                                    for dest in targets:
                                        rate=np.clip((dest-sent)/stride,-.8,.8)
                                        u.append(rate)
                                        sent=np.clip(sent+rate*stride,lo,hi)
                                    tape=np.vstack((u,np.zeros((154-H,6))))
                                    spec=dict(tick=obs["tick"],leg=leg)
                                    # Far requests before the physical fork retain
                                    # their historical RNG seed. The intervened and
                                    # later draws use the new seed. This is solely
                                    # the verification routine, not the arena law.
                                    def expected_draw(card_at, j, *args):
                                        return draw(c["card"] if j < leg else card_at, j, *args)
                                    with patch.object(evaluator, "draw", expected_draw):
                                        grade=school.physical(a,snapshot,card,physical_row,spec,tape)
                                        if grade["landed"]:
                                            audit=school.physical(a,snapshot,card,physical_row,spec,tape,guard)
                                            if audit["event_sha256"]!=grade["event_sha256"]:
                                                raise ValueError("Independent native CAD replay diverged")
                                            grade["clearance"]=audit["clearance"]
                                    outcomes[mode]=dict(grade,**meta)
                                trials.append(dict(order=order, optical={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in obs.items()},
                                                   results=outcomes))
                                print(c["id"],order["seed"],[(mode,x["landed"],x["answered"],
                                    None if x.get("actual") is None else round(x["actual"]["radius"],3)) for mode,x in outcomes.items()],flush=True)
                        finally:
                            cam.close(); a.close()
                result["rows"].append(dict(id=c["id"], phase=c["phase"], trials=trials, skipped=lesson.get("skipped")))
                result["complete"]=ci+1==len(d["cases"])
                _write(OUT,result)
        finally:
            cad.close()


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("mode", choices=("plan","scout","fit","held"))
    task = cmd.parse_args()
    {"plan":freeze,"scout":do_scout,"fit":fit,"held":held}[task.mode]()
