"""Compare causal camera states on the same three archived physical rim contrasts."""
import argparse
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt42_camera_precision as precision
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt41_early_optics import EarlyReadout
from .tt42_optical_uncertainty import ForecastErrors, NORM
from .tt48_contact_optical_check import CHECKS
from .tt48_contact_patch import ROOT

FLIGHT = Path("artifacts/tt37/flight-fit.npz")
HZ60 = Path("artifacts/tt37/hz60-rebound.npz")
ERRORS = Path("artifacts/tt42/optical-uncertainty.npz")
DT = mpc.DT
RNG=480221


def hypotheses():
    # No measurement, outcome, target, or field from the three test scenes
    # sets any ensemble width. Position/velocity use the archived optical
    # error standardization, with a bounded spin nuisance prior around the
    # ordinary two-model MPC center.
    z=np.clip(ndtri(qmc.Sobol(d=9,scramble=True,seed=RNG).random_base2(6)),-1.8,1.8)
    return z*np.r_[NORM,19.,11.,13.]+np.r_[np.zeros(6),16.,0.,0.]


def outcome(engine, goals):
    f=np.zeros((2,9))
    _,y=engine.predict.trajectories(goals,mpc.AIM,.48,feedback=f)
    return dict(legal=np.isfinite(y[:,0]).tolist(), departure=f[:,2:5].tolist(),
                contact=(f[:,1]>.5).tolist(), radii=f[:,8].tolist())


def tally(samples):
    y=np.asarray([x["legal"] for x in samples],dtype=bool).reshape(-1,2)
    return dict(n=len(y),legal_votes=y.sum(0).tolist(),
                pair_correct=int(np.sum(~y[:,0]&y[:,1])),
                spurious_stock=int(y[:,0].sum()),
                opposite=int(np.sum(y[:,0]&~y[:,1])),
                both_legal=int(np.sum(y[:,0]&y[:,1])),
                missed_both=int(np.sum(~y[:,0]&~y[:,1])),
                earned_wins_votes=bool(y[:,1].sum()>y[:,0].sum()))


def check(ident, original, witness, residual):
    archive, study=_read(original),_read(witness)
    if study["source_sha256"]!=_hash(original):
        raise ValueError("Changed fixed test command tape")
    r=next(x for x in study["rows"] if x["id"]==ident)
    row,card=archive["rows"][r["index"]],r["card"]
    ready=np.asarray(archive["ready"])
    result=dict(id=ident,decision_tick=r["fork"],original_tapes=list(("stock","edge")),
                actual_legal=[r["arms"][key]["landed"] for key in ("stock","edge")])
    with mpc.centered(ready):
        a=mpc.arena()
        image,rig,stride,lo,hi=mpc.resources(a)
        high,highrig=precision.zoomed(a)
        dense=precision.Shutter(high,highrig)
        engine=mpc.OpticalClinic(body=True)
        try:
            eyes={}
            for name,model in (("standard",course.OPTIC),("flight_fit",FLIGHT),
                              ("early_bounce",EarlyReadout(course.OPTIC)),("rebound_60",HZ60)):
                eye=mpc.Vision(rig,a.control_dt,raw=True)
                eye.eye=mpc.LearnedOptics(eye.eye,model)
                eyes[name]=eye
            bounce={tag:False for tag in eyes}
            at=None
            ref=ready.copy()
            a.launch(row["shot"],card=card)
            a.partner.reset(seed=card["seed"])
            for tick in range(r["fork"]+1):
                q=a.joints
                frame=image.capture()
                for tag,eye in eyes.items():
                    pv=mpc.observed(eye,tick,frame if tick%2==0 or tag=="rebound_60" else None,q)
                    bounce[tag]=precision.bounce_flag(bounce[tag],tick*DT,eye.eye.latest)
                    if tag=="standard":
                        at=pv
                dense.step(tick*DT,at)
                engine.observe(tick,eyes["standard"].eye,q,ref.copy())
                if tick==r["fork"]:
                    break
                u=np.asarray(row["actions"][tick])
                ref=np.clip(ref+u*stride,lo,hi)
                a.exchange(u,mpc.FAR_COMMAND)
            # The physical state is not read into the private plant. Its
            # events only validate the preceding executed tape.
            prior=[e for e in row["events"] if e["time"]<=a.data.time+1e-8]
            if digest_json(a.events)!=digest_json(prior):
                raise ValueError("Camera replay changed the prior physical prefix")
            goals=[]
            for key in ("stock","edge"):
                control=ref.copy()
                issued=[]
                for u in r["arms"][key]["controls"]:
                    control=np.clip(control+stride*np.asarray(u),lo,hi)
                    issued.append(control)
                goals.append(mpc.tail(np.asarray(issued),mpc.HORIZON+mpc.TAIL))
            point={}
            standard=None
            twin=engine.twin
            spin=twin.data.qvel[twin._ball_dof+3:twin._ball_dof+6]
            for tag,eye in eyes.items():
                fit=eye.eye.latest
                if precision.stock_at(tick*DT,fit) is None:
                    point[tag]=None
                    continue
                engine.latest=fit
                engine.tracker=eye.eye
                engine.bounced=bounce[tag]
                assert engine.sync()
                point[tag]=outcome(engine,goals)|dict(state=np.r_[twin.ball,twin.velocity,spin].tolist(),
                                                       bounced=bool(twin._received_bounce))
                if tag=="standard":
                    standard=np.r_[twin.ball,twin.velocity,spin]
            if standard is None:
                return result|dict(point=point,note="None of the plausible proposals replaces no-sighting")
            if at is not None:
                update=dense.accepted(tick*DT,at,bounce["standard"])
                # The original dense model abstains prebounce. Also test the
                # identical smooth-suffix fit with a descending prebounce
                # phase check instead, with no future RGB or native bounce.
                extension=None
                if update is None and not bounce["standard"] and at[5]<-.55:
                    extension=dense.sensor._estimate(tick*DT,at,True)
                for name,est in (("dense_published",update),("dense_prebounce_extension",extension)):
                    if est is None:
                        point[name]=None
                        continue
                    eye=eyes["standard"].eye
                    engine.latest=replace(eye.latest,time=tick*DT,position=est["pv"][:3],velocity=est["pv"][3:])
                    engine.tracker=eye
                    engine.bounced=bounce["standard"]
                    assert engine.sync()
                    point[name]=outcome(engine,goals)|dict(state=np.r_[twin.ball,twin.velocity,spin].tolist(),
                        bounced=bool(twin._received_bounce),img_error=est["rms_base_pixel"],camera_pairs=est["pairs"])
            engine.latest=eyes["standard"].eye.latest
            engine.tracker=eyes["standard"].eye
            engine.bounced=bounce["standard"]
            assert engine.sync()
            sets={"bounded_64":hypotheses(),
                  "stock_two":np.asarray([np.r_[s*np.array([.0018,-.0023,.0028,.023,.043,.073]),
                                             16.+17.*s,0.,0.] for s in (-1.,1.)])}
            learned=residual.scenarios(standard[:6])
            if learned is not None:
                sets["archived_pose_velocity_error_two"]=np.c_[learned,[-1.,33.],np.zeros((2,2))]
            logs={}
            for name,delta in sets.items():
                variants=[]
                for perturbation in delta:
                    v=standard+perturbation
                    twin.ball[:],twin.velocity[:],spin[:]=v[:3],v[3:6],v[6:]
                    mujoco.mj_forward(twin.model,twin.data)
                    variants.append(outcome(engine,goals))
                logs[name]=tally(variants)|dict(outputs=variants)
            return result|dict(point=point,ensembles=logs,archived_error_support=learned is not None)
        finally:
            engine.twin.close()
            high.close()
            image.close()
            a.close()


def run(out):
    with np.load(ERRORS) as d:
        practiced=set(map(int,d["practice_ids"]))
        if practiced & {c for c,*_ in CHECKS}:
            raise ValueError("Optical uncertainty practice overlaps the physical test")
        residual=ForecastErrors(d["states"],d["errors"])
    receipts={str(x):_hash(x) for x in (FLIGHT,HZ60,ERRORS,course.OPTIC,*[p for _,a,b in CHECKS for p in (a,b)])}
    answer=dict(sources=receipts, code_sha256=_hash(Path(__file__)), rows=[],complete=False,
                 boundary="Only original RGB through the original decision tick and own encoders initialize any variant. Same achieved physical bounded controls for both command choices. Body stays optical/encoder initialized. Recorded actual outcomes only grade the pair, and do not pick a plausible hypothesis.")
    for ident,source,auth in CHECKS:
        r=check(ident,source,auth,residual)
        answer["rows"].append(r)
        answer["complete"]=len(answer["rows"])==len(CHECKS)
        _write(out,answer)
        print(ident,{name:None if v is None else v["legal"] for name,v in r["point"].items()},
              {name:(v["legal_votes"],v["pair_correct"]) for name,v in r["ensembles"].items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path,default=ROOT/"optical-alternatives.json",nargs="?")
    run(p.parse_args().out)
