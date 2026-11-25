"""Measure release-disjoint sim-to-real edge departure discrepancy from actual RGB."""
import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt36_impact_refine import Frames
from .tt40_real_contact import CAMERA
from .tt44_paired_credit import State
from .tt48_contact_patch import ROOT, SOURCE
from .tt48_rim_recovery import alignment

PRACTICES=((SOURCE,ROOT/"causal.json.gz"),
           (Path("artifacts/tt46_season/learning-48.json.gz"),ROOT/"new-offers.json.gz"))
CAPTURE=ROOT/"discrepancy-rgb.json.gz"


def packet(arena, room, engine, row, old, ready, calibration):
    image,rig,stride,lo,hi=room
    c=row["card"]
    arena.launch(old["shot"],card=c)
    arena.partner.reset(seed=c["seed"])
    eye=mpc.Vision(rig,arena.control_dt,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
    ref=np.asarray(ready).copy()
    for tick in range(row["fork"]+1):
        q=arena.joints
        x=mpc.observed(eye,tick,image.capture() if tick%2==0 else None,q)
        engine.observe(tick,eye.eye,q,ref.copy())
        if tick==row["fork"]:
            break
        u=np.asarray(old["actions"][tick])
        ref=np.clip(ref+u*stride,lo,hi)
        arena.exchange(u,mpc.FAR_COMMAND)
    expect=[e for e in old["events"] if e["time"]<=arena.data.time+1e-8]
    if digest_json(expect)!=digest_json(arena.events):
        raise ValueError("Archived preparation not reproduced")
    result=dict(id=row["id"], fork=row["fork"], arms={}, valid=x is not None and engine.sync())
    if not result["valid"]:
        return result
    # This is the only shared sensor context used in training; no actual
    # future placement, missed trajectory, or native contact record in it.
    result["context"]=np.r_[x,q[:5],ref[:5]-q[:5]].tolist()
    mark=State(arena)
    for mode,arm in row["arms"].items():
        mark.restore(arena)
        optical=Frames(rig)
        issued=[]
        reference=ref.copy()
        for j,u in enumerate(arm["controls"]):
            now=row["fork"]+j
            if j<36:
                optical.observe(image.capture(),now*mpc.DT,arena.joints)
            arr=np.asarray(u)
            reference=np.clip(reference+arr*stride,lo,hi)
            issued.append(reference)
            arena.exchange(arr,mpc.FAR_COMMAND)
        if digest_json(arena.events)!=arm["event_sha256"]:
            raise ValueError(f"{row['id']} {mode}: RGB sampling changed archived physics")
        choices=optical.estimates(calibration)
        # Default first causal contact estimate in this cropped physical
        # window. No native event timing/velocity chooses the fitted label.
        visible=next((v for v in choices if .07<v["time"]-row["fork"]*mpc.DT<.58),None)
        goal=[mpc.tail(np.asarray(issued),mpc.HORIZON+mpc.TAIL)]
        observed=np.zeros((1,9))
        _,y=engine.predict.trajectories(goal,mpc.AIM,.48,feedback=observed)
        angle=alignment(engine,[goal[0][:mpc.HORIZON]])[0]
        result["arms"][mode]=dict(
            issued=np.asarray(issued[:mpc.HORIZON]).tolist(), native_legal=bool(np.isfinite(y[0,0])),
            virtual_feedback=observed[0].tolist(),
            virtual_alignment=float(angle) if np.isfinite(angle) else None,
            measured=visible, landed=arm["landed"], answered=arm["answered"],
            camera_points=len(optical.stereo), event_sha256=arm["event_sha256"])
    return result


def collect(out):
    receipts={str(p):_hash(p) for refs in PRACTICES for p in refs}
    calibration=_read(CAMERA)["models"]["60"]
    work=[]
    ready=None
    for archive_path,practice_path in PRACTICES:
        a,p=_read(archive_path),_read(practice_path)
        if not p["complete"] or p["source_sha256"]!=_hash(archive_path):
            raise ValueError("Changed source for real optical labels")
        if ready is None:
            ready=a["ready"]
        if ready!=a["ready"]:
            raise ValueError("Changed clamp or ready posture")
        work.extend((r,a["rows"][r["index"]]) for r in p["rows"])
    out.parent.mkdir(exist_ok=True,parents=True)
    source=dict(schema="tt48-camera-rim-discrepancy-1",inputs=receipts,calibration_sha256=_hash(CAMERA),
                code_sha256=_hash(Path(__file__)),rows=[],ready=ready,complete=False)
    if out.exists():
        old=_read(out)
        if any(old[k]!=v for k,v in source.items() if k not in ("rows","complete")):
            raise ValueError("RGB packet provenance changed")
        source=old
    with mpc.centered(ready):
        arena=mpc.arena()
        room=mpc.resources(arena)
        engine=mpc.OpticalClinic(body=True)
        try:
            for row,old in work[len(source["rows"]):]:
                v=packet(arena,room,engine,row,old,ready,calibration)
                source["rows"].append(v)
                source["complete"]=len(source["rows"])==len(work)
                _write(out,source)
                print(v["id"],[(name,a["landed"],a["native_legal"],
                                 a["measured"]["outgoing"] if a["measured"] else None)
                                for name,a in v["arms"].items()],flush=True)
        finally:
            engine.twin.close()
            room[0].close()
            arena.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path,default=CAPTURE,nargs="?")
    collect(p.parse_args().out)
