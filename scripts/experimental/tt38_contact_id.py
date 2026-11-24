"""Identify an effective contact-state correction from issued actions and stereo outcomes."""

import argparse
from collections import Counter
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from time import process_time
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt33_joint as joint
from . import tt37_optical_clinic as clinic
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt34_results import paired
from .tt35_error import passage
from .tt36_impact_refine import Frames
from .tt37_opponent import arena
from .tt37_rebound import fit as optical_fit
from .tt37_visual_inertia import LearnedOptics

ARTIFACTS = Path(__file__).resolve().parents[2]/"artifacts"
OLD = ARTIFACTS/"tt37"
OUT = ARTIFACTS/"tt38"
VISION = OLD/"flight-fit.npz"
IMPACT = ARTIFACTS/"tt36"/"impact-model.json"
TRAIN = (("risk-recheck",0,6),("reuse-cold",0,4))
HELD = (("stream-learn",0,6),("reuse-warm-frozen",6,9))
CONTACT_HORIZON = 25
LIVE_GAMES = 4
AIM = np.array([0.,1.29,1.05,3.60])
SCALES = np.array([.16,.33,.16,.80,1.2,1.65,.75,.8,.85,.8,.8,1.2,1.2,1.5,.7])
# A discrete bank makes it possible to estimate a conditional action
# consequence without averaging physically distinct contact hypotheses.
# Dimensions are [dy, dz, dvy, dvz, dwx, dwy, dwz], SI units.
DOMAIN = np.array([.014,.009,.19,.27,39.,33.,39.])
HYPOTHESES = np.vstack((np.zeros((1,7)), np.diag(DOMAIN), -np.diag(DOMAIN)))
VELOCITY_SCALE = np.array([1.2,1.65,1.65])


def bounce(track):
    """Measure an already visible bounce using 3 actual samples on each side."""
    vals=list(track)
    for i in range(len(vals)-4,1,-1):
        seg=vals[i-2:i+4]
        dates=np.array([v[0] for v in seg])
        p=np.array([v[1] for v in seg])
        if (not np.isfinite(p).all() or not .72 < p[2,2] < .89
                or not (-1.34<p[2,1]<.01) or max(np.diff(dates))>.042):
            continue
        t0=.5*(dates[2]+dates[3])
        left,incoming,_=optical_fit(dates[:3]-t0,p[:3])
        right,outgoing,_=optical_fit(dates[3:]-t0,p[3:])
        if incoming[2]<-.80 and outgoing[2]>.7 and incoming[1]<-.8 and outgoing[1]<-.8:
            if np.linalg.norm(left-right)<.12 and vals[-1][0]-t0<.36:
                return np.clip(outgoing - incoming*np.array([1,1,-1]),-2.,2.),1.
    return np.zeros(3),0.


def context(engine):
    """Read only the present sensor-initialized private state and prior optical images."""
    t=engine.twin
    track=getattr(getattr(engine.tracker,"filter",None),"track",())
    change,seen=bounce(track)
    return np.r_[t.ball,t.velocity,engine.previous[:5],change,seen]


def shifted(engine, value):
    t=engine.twin
    delta=np.asarray(value)
    t.ball[[1,2]]+=delta[:2]
    t.velocity[[1,2]]+=delta[2:4]
    i=t._ball_dof+3
    t.data.qvel[i:i+3]+=delta[4:7]


def state_rollout(engine, stored, bounced, goals, eta):
    twin=engine.twin
    feedback=np.zeros((len(HYPOTHESES),9))
    winning=np.zeros(len(HYPOTHESES),dtype=bool)
    for i,h in enumerate(HYPOTHESES):
        mujoco.mj_copyData(twin.data,twin.model,stored)
        twin._received_bounce=bounced
        shifted(engine,h)
        fb=feedback[i:i+1]
        _,r=engine.trajectories(goals[None],AIM,eta+.13,feedback=fb)
        winning[i]=np.isfinite(r[0,0])
    return feedback,winning


def outcome_cost(feedback, actual, miss):
    hit=feedback[:,1]>.5
    if actual is not None:
        loss=np.linalg.norm((feedback[:,2:5]-actual["outgoing"])/VELOCITY_SCALE,axis=1)
        location=np.linalg.norm((feedback[:,5:8]-actual["center"])/[.10,.12,.09],axis=1)
        return np.where(hit,np.minimum(loss,5.)+.14*np.minimum(location,4.),3.2)
    if miss is None:
        raise ValueError("An invisible outcome cannot label a hypothesis")
    return np.where(hit,2.4,.16*np.minimum(1.0,abs(feedback[:,0]-miss)/.25))


def collect_point(physical, renderer, rig, stride, lo, hi, engine, card, row, ready, impact,
                  horizon=CONTACT_HORIZON):
    physical.launch(row["shot"],card=card)
    physical.partner.reset(seed=card["seed"])
    engine.previous=None
    engine.bounced=False
    vision=joint.Vision(rig,physical.control_dt)
    vision.eye=LearnedOptics(vision.eye,VISION)
    view=Frames(rig)
    commands=np.asarray(row["actions"])
    dt=physical.control_dt
    qref=np.asarray(ready)
    refs=[]
    snapshots=[]
    last_snapshot=-5.
    last_inbound=-10.
    started=process_time()
    for tick,command in enumerate(commands):
        now=tick*dt
        q=physical.joints.copy()
        dense=now-last_inbound<.27
        frame=renderer.capture() if tick%2==0 or dense else None
        obs=vision.update(tick,frame if tick%2==0 else None,q)
        if frame is not None and (dense or obs is not None and obs[0]<.30):
            view.observe(frame,now,q)
        if obs is not None and -.04<obs[0]<.32:
            last_inbound=now
        engine.observe(tick,vision.eye,q,qref.copy())
        if (obs is not None and .095<obs[0]<.164 and now-last_snapshot>.48
                and engine.latest is not None and engine.latest.rms<2.4
                and engine.latest.depth_sigma<.16 and now-engine.latest.time<.075 and engine.sync()):
            t=engine.twin
            shadow=mujoco.MjData(t.model)
            mujoco.mj_copyData(shadow,t.model,t.data)
            snapshots.append((tick,float(obs[0]),shadow,bool(t._received_bounce),context(engine)))
            last_snapshot=now
        physical.exchange(command,joint.FAR_COMMAND)
        qref=np.clip(qref+command*stride,lo,hi)
        refs.append(qref.copy())
    # Digest verification does not select, grade or time a training example.
    if digest_json(physical.events)!=digest_json(row["events"]):
        raise ValueError("Recorded commands did not reproduce the source events")
    detections=view.estimates(impact)
    opt=np.asarray(refs)
    stereo=[(round(t/dt),p,b) for t,p,b in view.stereo if b is not None]
    records=[]
    for i,eta,stored,seen,x in snapshots:
        t=i*dt
        near=[d for d in detections if .029<d["time"]-t<.34 and abs(d["time"]-t-eta)<.135]
        actual=min(near,key=lambda d:abs(d["time"]-t-eta)) if near else None
        miss=None
        if actual is None:
            p=passage([(k,b,a) for k,b,a in stereo if i<k<=i+22],i)
            if p is not None and np.linalg.norm(p["gap_xz_m"])>.13:
                miss=float(np.linalg.norm(p["gap_xz_m"]))
            else:
                continue
        # No need to simulate most of the outgoing flight. The identified
        # observation is the signed departure, formed before this prefix ends.
        goals=opt[i:i+horizon]
        if len(goals)<5:
            continue
        y,legal=state_rollout(engine,stored,seen,goals,eta)
        loss=outcome_cost(y,actual,miss)
        records.append(dict(context=x.tolist(), loss=loss.tolist(), predicted=y.tolist(),
                            horizon_s=len(goals)*dt, measured=None if actual is None else actual["outgoing"],
                            camera_only_miss_gap_m=miss, tick=i, observer=None if actual is None else actual["estimator"]))
    return records,dict(snapshots=len(snapshots),scored=len(records),process_cpu_s=process_time()-started)


def collect(*, resume=False):
    impact=_read(IMPACT)["models"]["60"]
    occupied=set()
    for role,split in (("fit",TRAIN),("held",HELD)):
        output=OUT/f"contact-id-{role}-observations.json"
        digests={s:_hash(OLD/f"{s}.json.gz") for s,_,_ in split}
        existing=_read(output) if resume and output.exists() else None
        bank=existing["points"] if existing else []
        if existing is not None and (existing["sources"]!=digests
                or existing["vision_sha256"]!=_hash(VISION) or existing["impact_sha256"]!=_hash(IMPACT)):
            raise ValueError("Cannot mix resumed source tapes or sensor calibration")
        complete={(r["source"],r["row"]) for r in bank}
        scheduled={(source,i) for source,start,end in split for i in range(start,end)}
        if not complete<=scheduled:
            raise ValueError("The resume inputs are not a subset of the prescribed physical replays")
        for name,start,end in split:
            path=OLD/f"{name}.json.gz"
            d=_read(path)
            cards=d["cards"][start:end];rows=d["rows"][start:end]
            if len(cards)!=end-start or len(rows)!=len(cards):
                raise ValueError("Incomplete source")
            if role=="fit":
                occupied.update(c["id"] for c in cards)
            elif occupied&{c["id"] for c in cards}:
                raise ValueError("One calibration release appears in held observations")
            scheduled_rows=[(i,c,r) for i,(c,r) in enumerate(zip(cards,rows,strict=True),start)
                            if (name,i) not in complete]
            if not scheduled_rows:
                continue
            joint.RECEIVE_Y=float(d["depth"])
            with centered(d["ready"]):
                env=arena(protocol="random")
                engine=clinic.OpticalClinic(body=True)
                camera,rig,stride,lo,hi=joint.resources(env)
                try:
                    for index,c,r in scheduled_rows:
                        legacy=name=="risk-recheck"
                        observed,stats=collect_point(env,camera,rig,stride,lo,hi,engine,c,r,d["ready"],impact,
                                                     horizon=90 if legacy else CONTACT_HORIZON)
                        bank.append(dict(source=name,row=index,**stats,records=observed))
                        _write(output,
                               dict(sources=digests,
                                    vision_sha256=_hash(VISION),impact_sha256=_hash(IMPACT),points=bank))
                        print(role,name,index,len(observed),stats["snapshots"],flush=True)
                finally:
                    camera.close();engine.twin.close();env.close()


class Correction:
    def __init__(self, rows, *, mode="local"):
        self.x=np.array([r["context"] for r in rows])
        self.loss=np.array([r["loss"] for r in rows])
        self.mode=mode

    def choose(self,x):
        d=np.linalg.norm((self.x-x)/SCALES,axis=1)
        if self.mode=="global":
            selected=np.arange(len(d))
            w=np.ones(len(selected))
        else:
            selected=np.argsort(d)[:18]
            if not len(selected) or d[selected[0]]>3.15:
                return 0
            w=np.exp(-.55*(d[selected]**2-d[selected[0]]**2))
        # Contact perturbations need measurable support before overriding zero.
        rank=(w@self.loss[selected])/w.sum()
        best=int(np.argmin(rank+.045*np.arange(len(rank)).astype(bool)))
        return best if rank[0]-rank[best]>.09 else 0


def all_records(doc):
    return [r for point in doc["points"] for r in point["records"]]


def score(records, correction=None, *, oracle=False):
    fixed=correction is None
    idx=[int(np.argmin(r["loss"])) if oracle else 0 if fixed else correction.choose(r["context"])
         for r in records]
    observed=[(r,i) for r,i in zip(records,idx,strict=True) if r["measured"] is not None]
    hit=[(r,i) for r,i in observed if r["predicted"][i][1]>.5]
    misses=[(r,i) for r,i in zip(records,idx,strict=True) if r["measured"] is None]
    diff=np.asarray([np.asarray(r["predicted"][i][2:5])-r["measured"] for r,i in hit])
    return dict(outcomes=len(records), chosen=dict(Counter(idx)), observed_touches=len(observed),
                predicted_touches=len(hit), observed_clear_misses=len(misses),
                false_touches=sum(r["predicted"][i][1]>.5 for r,i in misses),
                all_loss=float(np.mean([r["loss"][i] for r,i in zip(records,idx,strict=True)])),
                mean_velocity_mps_error=None if not len(diff) else float(np.mean(np.linalg.norm(diff,axis=1))),
                mean_forward_bias_mps=None if not len(diff) else float(np.mean(diff[:,1])),
                mean_vertical_bias_mps=None if not len(diff) else float(np.mean(diff[:,2])))


def shared_contacts(rows, correction):
    matched=[]
    for r in rows:
        if r["measured"] is None:
            continue
        chosen=correction.choose(r["context"])
        a,b=np.asarray(r["predicted"])[[0,chosen]]
        if a[1]>.5 and b[1]>.5:
            matched.append(np.linalg.norm(np.array([a[2:5],b[2:5]])-r["measured"],axis=1))
    if not matched:
        return dict(pairs=0)
    errors=np.array(matched)
    wins=int(np.sum(errors[:,1]<errors[:,0]-1e-10))
    losses=int(np.sum(errors[:,1]>errors[:,0]+1e-10))
    return dict(pairs=len(matched), nominal_mps=float(errors[:,0].mean()),
                adjusted_mps=float(errors[:,1].mean()),wins=wins,losses=losses,
                ties=len(matched)-wins-losses)


def fit():
    train,held=(OUT/f"contact-id-{role}-observations.json" for role in ("fit","held"))
    source,control=_read(train),_read(held)
    if any(d["vision_sha256"]!=_hash(VISION) or d["impact_sha256"]!=_hash(IMPACT) for d in (source,control)):
        raise ValueError("The two replays must share their frozen optical calibration")
    rows=all_records(source);test=all_records(control)
    if not rows or not test:
        raise ValueError("Both replay roles need a matched visible outcome")
    _write(OUT/"contact-id-checkpoint.json",dict(
        rows=[{k:r[k] for k in ("context","loss")} for r in rows],train_sha256=_hash(train),
        observations=len(rows),hypotheses=HYPOTHESES.tolist(),vision_sha256=_hash(VISION)))
    local=Correction(rows);global_model=Correction(rows,mode="global")
    receipt=dict(fit_sha256=_hash(train),held_sha256=_hash(held),train=score(rows,local),
                 held=dict(nominal=score(test),local=score(test,local),global_offset=score(test,global_model),
                           diagnostic_outcome_aware_best=score(test,oracle=True),
                           on_shared_detected_contacts=shared_contacts(test,local)))
    _write(OUT/"contact-id-sensor-results.json",receipt)
    print(receipt["held"],flush=True)


def live_plan():
    selected=_read(OLD/"stream-learn.json.gz")["cards"][:LIVE_GAMES]
    cards=[dict(c,reply_seed=381609+17*i,key=f"contact-id-held-{i}")
           for i,c in enumerate(selected)]
    _write(OUT/"contact-id-plan.json",dict(held=cards))


class SensorCalibrated(clinic.OpticalClinic):
    """Apply an inferred disturbance only to the private twin at sync."""
    def __init__(self, correction, record, **settings):
        super().__init__(**settings)
        self.contact_correction=correction
        self.calibration_calls=record

    def sync(self):
        valid=super().sync()
        if not valid:
            return False
        x=context(self)
        if self.bounced and -1.47<x[1]<-.62 and x[4]<-.9:
            action=self.contact_correction.choose(x) if self.contact_correction is not None else 0
            if action:
                shifted(self,HYPOTHESES[action])
                mujoco.mj_forward(self.twin.model,self.twin.data)
            self.calibration_calls.append(action)
        return valid


def live(mode):
    snapshot=OUT/"contact-id-checkpoint.json"
    reset=_read(snapshot)
    if reset["vision_sha256"]!=_hash(VISION):
        raise ValueError("Holding a different sensor model would change the contact prior")
    model=Correction(reset["rows"],mode=mode) if mode!="nominal" else None
    logged=[]
    result=OUT/f"contact-id-live-{mode}.json.gz"
    options=["tt37_motor_clinic","optical-fixed",str(result),"--plan",str(OUT/"contact-id-plan.json"),
             "--group","held","--n",str(LIVE_GAMES),"--depth","-1.56","--cut",".18","--samples","35","--rounds","3",
             "--guided","--refine","--body","--cap","24","--visual",str(VISION),
             "--memory",str(OLD/"reuse-warm-frozen.json.gz")]
    def new_engine(**settings):
        return SensorCalibrated(model,logged,**settings)
    with patch.object(clinic,"OpticalClinic",new_engine),patch.object(sys,"argv",options):
        runpy.run_module("scripts.experimental.tt37_motor_clinic",run_name="__main__")
    d=_read(result)
    _write(result,d|dict(contact_calibration=mode,model_sha256=_hash(snapshot),sync_calls=dict(Counter(logged))))


def classify_for_report(doc):
    """Read the referee independently of fitting to show which strokes were represented."""
    sources={}
    counts=Counter()
    for point in doc["points"]:
        name=point["source"]
        if name not in sources:
            path=OLD/f"{name}.json.gz"
            if doc["sources"][name]!=_hash(path):
                raise ValueError("The independent referee tape changed")
            sources[name]=_read(path)
        events=sources[name]["rows"][point["row"]]["events"]
        for outcome in point["records"]:
            if outcome["measured"] is None:
                counts["clear_visible_miss"]+=1
                continue
            origin=outcome["tick"]/60.
            match=next((j for j,event in enumerate(events)
                        if event["kind"]=="contact" and event["side"]==1
                        and origin+.029<event["time"]<origin+.34),None)
            if match is None:
                counts["unmatched_optical_deflection"]+=1
                continue
            stop=next((j for j in range(match+1,len(events)) if events[j]["kind"]=="contact"),len(events))
            legal=any(e["kind"]=="legal_return" and e["side"]==1 for e in events[match+1:stop])
            counts["legal_touch" if legal else "nonlegal_touch"]+=1
    return dict(counts)


def summary():
    modes=[v for v in ("nominal","local","global")
           if v!="global" or (OUT/"contact-id-live-global.json.gz").is_file()]
    tapes={v:_read(OUT/f"contact-id-live-{v}.json.gz") for v in modes}
    before=tapes["nominal"]
    for after in tapes.values():
        if before["cards"]!=after["cards"] or len(after["rows"])!=len(after["cards"]):
            raise ValueError("Incomplete or unpaired held test")
    picked=[i for i,row in enumerate(before["rows"]) if row["placements"]["delivered"]]
    for after in tapes.values():
        if picked!=[i for i,row in enumerate(after["rows"]) if row["placements"]["delivered"]]:
            raise ValueError("Actual first physical offer missing in one condition")
    eligible={v:dict(d,rows=[d["rows"][i] for i in picked],cards=[d["cards"][i] for i in picked])
              for v,d in tapes.items()}
    labels={k:classify_for_report(_read(OUT/f"contact-id-{k}-observations.json")) for k in ("fit","held")}
    s=dict(initial_opponent_faults=len(before["cards"])-len(picked),
           games={v:paired(eligible["nominal"],eligible[v]) for v in modes if v!="nominal"},
           label_categories_evaluator_only=labels,
           sync_calls={v:d["sync_calls"] for v,d in tapes.items()})
    _write(OUT/"contact-id-live-results.json",s)
    print(s,flush=True)


def audit():
    before=_read(OUT/"contact-id-live-nominal.json.gz")
    checked={}
    with TemporaryDirectory(prefix="tt38-contact-replay-") as folder:
        corpus,report=(Path(folder)/x for x in ("subset.json.gz","report.json"))
        for mode in ("local","global"):
            path=OUT/f"contact-id-live-{mode}.json.gz"
            if not path.exists():
                continue
            after=_read(path)
            if before["cards"]!=after["cards"] or len(before["rows"])!=len(before["cards"]):
                raise ValueError("Invalid held replay comparison")
            difference=[abs(a["score"]["learner"]-b["score"]["learner"])
                        for a,b in zip(before["rows"],after["rows"],strict=True)]
            i=max(range(len(difference)),key=lambda i:(difference[i],after["rows"][i]["score"]["learner"]))
            selected=dict(after,cards=[after["cards"][i]],rows=[after["rows"][i]])
            _write(corpus,selected)
            with patch.object(joint,"replay_arena",lambda cards,**settings:
                              arena(cards,protocol="random",**settings)):
                joint.verify(corpus,report,cad=True)
            checked[mode]=dict(independent_replay=_read(report)["points"],index=i,source_sha256=_hash(path))
    _write(OUT/"contact-id-audit.json",checked)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("collect","fit","plan","nominal","local","global","summary","audit"))
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    fn={"collect":lambda:collect(resume=a.resume),"fit":fit,"plan":live_plan,
        "summary":summary,"audit":audit}.get(a.phase)
    fn() if fn else live(a.phase)
