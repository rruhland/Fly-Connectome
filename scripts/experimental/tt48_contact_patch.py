"""Same-delivery exact-state physical probes of low rim contact versus the inner face."""
import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, Rollout
from .tt37_opponent import draw
from .tt37_opponent_audit import CourtClearance, guarded
from .tt39_opportunity_map import projected_height
from .tt40_long_low import same_trace
from .tt41_shallow_model import impulse, unit
from .tt44_paired_credit import State

ROOT = Path("artifacts/tt48_contact_patch")
SOURCE = Path("artifacts/tt46_season/learning-24.json.gz")
# A left floor miss, a right fixture fault, a right low floor miss, and
# a central own-half fault. Pick brief native prefixes for these probes.
INDICES = (9, 11, 3, 6)
PATCHES = (0.023, .040, .055, .072)
SHIFT = (-6, -4, -2, 0, 2)
FRAMES = 80


def ease(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


def suffix(a, state, card, row, info, tape, *, clearance=None):
    state.restore(a)
    mark = len(a.events)
    start = float(a.data.time)
    if clearance is not None:
        guarded(a, clearance)
    else:
        a.sample_clearance(False)
    actual = []
    controls = []
    callback = a._racket_contact

    def hit(side, location, incoming):
        if side == 1 and not actual:
            blade, ball = a._racket_ids[1], a._geom_ids["ball"]
            found = [h for h in a.data.contact if h.efc_address >= 0 and set(h.geom) == {blade, ball}]
            if not found:
                raise ValueError("The arm event did not come from native collision")
            normal = a.data.geom_xmat[blade].reshape(3,3)[:,2].copy()
            if normal[1] < 0.:
                normal *= -1.
            center = a.data.geom_xpos[blade].copy()
            relative = np.asarray(location) - center
            tangent = relative - normal*(normal@relative)
            touch = min(found, key=lambda h: np.linalg.norm(h.pos-location))
            face_normal = touch.frame[:3] * (1. if int(touch.geom[0]) == blade else -1.)
            actual.append(dict(
                when=float(a.data.time-start), step=int(a.steps), location=np.asarray(location).tolist(),
                center=center.tolist(), normal=normal.tolist(), tangent=tangent.tolist(),
                radius=float(np.linalg.norm(tangent)), alignment=abs(float(face_normal@normal)),
                joints=a.joints[:5].tolist(),
            ))
        return callback(side, location, incoming)

    with patch.object(a, "_racket_contact", hit):
        for u in tape:
            _, _, done, capped, _ = a.exchange(np.asarray(u), mpc.FAR_COMMAND)
            controls.append(np.asarray(u).tolist())
            if done or capped or a.rival_returns > info["leg"]+1:
                break
    offers = a.point_placements()
    expected = row["placements"]["delivered"][info["leg"]]
    landed = offers["delivered"][info["leg"]] if len(offers["delivered"])>info["leg"] else None
    if (landed is None or not same_trace(landed["far_contact"], expected["far_contact"])
            or not same_trace(landed["near_bounce"], expected["near_bounce"])):
        raise ValueError("The branch did not receive the same physical delivery")
    for k, x in enumerate(offers["requested"]):
        if x["target"] != list(draw(card,k,"random",None)[0]):
            raise ValueError("The broad partner draw changed")
    answer = dict(
        controls=controls, matched_far_hit=True, matched_near_bounce=True,
        landed=bool(landed["near_landed"]), answered=len(offers["delivered"])>info["leg"]+1,
        actual=actual[0] if actual else None, stock_prefix_events=int(mark),
        events=a.events[mark:], event_sha256=digest_json(a.events),
        delivered=landed, score=audit(a.events,opening_side=1),
    )
    if clearance is not None:
        check=a.clearance_result()
        if check["samples"]!=round((a.data.time-start)/a.model.opt.timestep):
            raise ValueError("A changed native step was not checked")
        answer["clearance"]=check
    return answer


def ready_goals(a, old, start):
    axes=a._arm_ids[1][2]
    stride=a._joint_increments[axes]*a.control_dt/a.model.opt.timestep
    lo,hi=a._joint_low[axes],a._joint_high[axes]
    q=a.data.ctrl[axes].copy()
    goals=[]
    for j in range(FRAMES):
        if start+j<len(old["actions"]):
            q=np.clip(q+stride*np.asarray(old["actions"][start+j]),lo,hi)
        goals.append(q)
    return np.asarray(goals),lo,hi


def planning(a, old, info, stock):
    """Equal patch-grid budgets at matched observed incoming physical phases in the evaluator."""
    p0=a.ball.copy()
    v0=a.velocity.copy()
    arc=a.partner.flight.path(p0,v0,a.spin.copy(),1,bounced=a._received_bounce,horizon=.57)
    if not len(arc):
        return {}, {"why": "no nominal incoming arc"}
    planner, kin=Rollout(a,1),BladeIK(a)
    ref,lo,hi=ready_goals(a,old,info["fork"])
    times=DT*(1+np.arange(FRAMES))
    impact=stock["actual"]
    q0=np.clip(impact["joints"],kin.low,kin.high)
    n0=np.asarray(impact["normal"])
    radial=unit(np.asarray(impact["tangent"]))
    current=a.joints[:5].copy()
    moves=[ref]
    meta=[dict(anchor="original reference")]
    eta0=impact["when"]
    for shift in SHIFT:
        approx=float(eta0+shift*DT)
        if not .07<approx<.49:
            continue
        close=arc[np.argmin(abs(arc[:,0]-approx))]
        t,p,v=close[0],close[1:4],close[4:7]
        if abs(t-approx)>.018 or not -1.70<p[1]<-.96:
            continue
        for pace in (3.55,4.35):
            want,_=impulse(p,v,pace,.045)
            n=unit(.42*n0+.58*want)
            for radius in PATCHES:
                # Move the disk under the same incoming air path, holding
                # desired face orientation fixed across this matched grid.
                tang=unit(radial-n*(radial@n))
                target=p-.019*n-radius*tang
                q,(center,normal,*_)=kin.fit(target,n,q0)
                if abs(normal@n)<.78 or np.max(abs(q-current))>6.0*t+.30:
                    continue
                change=np.clip(q-q0,-.52,.52)
                for gain in (.92,1.23):
                    m=ref.copy()
                    m[:,:5]+=ease(times/max(.06,.92*t))[:,None]*(gain*change)
                    moves.append(np.clip(m,lo,hi))
                    meta.append(dict(delay_frames=shift, requested_radius=radius, pace=pace,
                                     gain=gain, contact_seed_s=float(t),
                                     fit_gap_m=float(np.linalg.norm(center-target)),
                                     face_dot=float(normal@n)))
    def run_batch(goals):
        fb=np.zeros((len(goals),9))
        at=np.full((len(goals),19),np.nan)
        rates,y=planner.trajectories(goals,mpc.AIM,.52,feedback=fb,approaches=at)
        return rates,y,fb,at
    rates,y,fb,at=run_batch(moves)
    good=np.isfinite(y[:,0])
    contacts=fb[:,1]>.5
    detail={"trials":len(moves),"legal":int(good.sum()),"separated":int(contacts.sum()),
            "radii":np.quantile(fb[contacts,8],[0,.25,.50,.75,1]).tolist() if contacts.any() else [],
            "stock_separation_radius":float(fb[0,8]) if contacts[0] else None,
            "stock_contact_in_model":bool(at[0,18]>.5)}
    # Keep non-scoring contacts as physical contrasts rather than filtering
    # on the desired result of the experiment.
    selected={}
    for label,bounds in (("inner_face",(.0,.059)),("edge",(.063,.082))):
        in_band=contacts & (fb[:,8]>=bounds[0]) & (fb[:,8]<bounds[1])
        in_band[0]=False
        eligible=np.flatnonzero(in_band)
        detail[label]={"separated":len(eligible),"legal":int(np.sum(in_band&good))}
        if not len(eligible):
            continue
        loss,_=mpc.cost(y,fb)
        ix=int(eligible[np.argmin(loss[eligible])])
        chosen=dict(meta[ix],nominal_radius=float(fb[ix,8]), nominal_legal=bool(good[ix]),
                    nominal_departure=fb[ix,2:8].tolist(),cost=float(loss[ix]),
                    nominal_ball_xyz=at[ix,1:4].tolist() if np.isfinite(at[ix,1:4]).all() else None)
        selected[label]=(rates[ix].tolist(), chosen)
    return selected,detail


DT=mpc.DT


def run(output,source,indices):
    d=_read(source)
    ready=d["ready"]
    initial=dict(schema="tt48-same-delivery-contact-patch-1",source_sha256=_hash(source),
                 code_sha256=_hash(Path(__file__)),indices=list(indices),rows=[],complete=False,
                 boundary="Privileged snapshot/arc is used to diagnose actuator reach in controlled physical practice only. No modified base, limits, racket, ball or opponent; not a deployed learner.")
    if output.exists():
        old=_read(output)
        if any(old[k]!=v for k,v in initial.items() if k not in ("rows","complete")):
            raise ValueError("Changed causal docket")
        initial=old
    output.parent.mkdir(exist_ok=True,parents=True)
    with mpc.centered(ready):
        a=mpc.arena()
        shadow=mpc.arena(detail="cad")
        guard=CourtClearance(shadow.model,articulated=True)
        try:
            for i in indices[len(initial["rows"]):]:
                start=process_time()
                c,row=d["cards"][i],d["rows"][i]
                leg=len(row["placements"]["delivered"])-1
                b=row["placements"]["delivered"][leg]
                if b["near_landed"] or not b.get("near_contact"):
                    raise ValueError("Not an archived touched terminal miss")
                fork=max(b["far_contact"]["step"]+3,b["near_contact"]["step"]-18)
                info=dict(index=i,id=c["id"],leg=leg,fork=fork,card=c,shot=row["shot"],
                          height=projected_height(b["receive_line"]))
                a.sample_clearance(False)
                a.launch(row["shot"],card=c)
                a.partner.reset(seed=c["seed"])
                for u in row["actions"][:fork]:
                    a.exchange(np.asarray(u),mpc.FAR_COMMAND)
                pre=[e for e in row["events"] if e["time"]<=a.data.time+1e-8]
                if digest_json(a.events)!=digest_json(pre):
                    raise ValueError("Original physical prefix did not replay exactly")
                state=State(a)
                stock=suffix(a,state,c,row,info,row["actions"][fork:])
                if stock["event_sha256"]!=digest_json(row["events"]):
                    raise ValueError("Stock suffix did not replay all original physical events")
                state.restore(a)
                recipes,private=planning(a,row,info,stock)
                arms={"stock":stock}
                for tag,(tape,meta) in recipes.items():
                    # Keep exact branch motion up to incoming/outgoing score,
                    # then release any original tape after the planned horizon.
                    zero=np.zeros((95,6)).tolist()
                    v=suffix(a,state,c,row,info,tape+zero,clearance=guard)
                    v["proposed"]=meta
                    arms[tag]=v
                initial["rows"].append(dict(info,arms=arms,private=private,cpu_s=process_time()-start))
                initial["complete"]=len(initial["rows"])==len(indices)
                _write(output,initial)
                print(c["id"],round(info["height"],3), private,
                      [(name, arm["landed"], None if arm["actual"] is None else round(arm["actual"]["radius"],3),
                        arm.get("clearance",{}).get("penetration_samples")) for name,arm in arms.items()],flush=True)
        finally:
            shadow.close()
            a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--output",type=Path,default=ROOT/"causal.json.gz")
    p.add_argument("--indices",nargs="+",type=int,default=INDICES)
    a=p.parse_args()
    run(a.output,a.source,a.indices)
