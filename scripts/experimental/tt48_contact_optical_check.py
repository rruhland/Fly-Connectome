"""Can the camera twin see the earned same-ball rim contrast at all?"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt48_contact_patch import ROOT, SOURCE
from .tt48_rim_recovery import alignment

CHECKS = ((9230, SOURCE, ROOT/"causal.json.gz"),
          (11809, SOURCE, ROOT/"causal.json.gz"),
          (9114, Path("artifacts/tt46_season/learning-48.json.gz"), ROOT/"new-offers.json.gz"))


def check(ident, original, witness, factors=False):
    archive, paper = _read(original), _read(witness)
    if paper["source_sha256"]!=_hash(original):
        raise ValueError("Changed physical source")
    row=next(r for r in paper["rows"] if r["id"]==ident)
    old=archive["rows"][row["index"]]
    c, ready=row["card"], np.asarray(archive["ready"])
    with mpc.centered(ready):
        table=mpc.arena()
        render,rig,stride,lo,hi=mpc.resources(table)
        engine=mpc.OpticalClinic(body=True)
        try:
            table.launch(old["shot"],card=c)
            table.partner.reset(seed=c["seed"])
            eye=mpc.Vision(rig,table.control_dt,raw=True)
            eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
            ref=ready.copy()
            for tick in range(row["fork"]+1):
                q=table.joints
                x=mpc.observed(eye,tick,render.capture() if tick%2==0 else None,q)
                engine.observe(tick,eye.eye,q,ref.copy())
                if tick==row["fork"]:
                    break
                u=np.asarray(old["actions"][tick])
                ref=np.clip(ref+u*stride,lo,hi)
                table.exchange(u,mpc.FAR_COMMAND)
            expected=[e for e in old["events"] if e["time"]<=table.data.time+1e-8]
            if digest_json(expected)!=digest_json(table.events):
                raise ValueError("Original command/camera prefix changed")
            if not engine.sync():
                return dict(id=ident,causal_optical_fit=False)
            names=["stock","edge"]
            goals=[]
            for kind in names:
                r=ref.copy()
                stream=[]
                for u in row["arms"][kind]["controls"]:
                    r=np.clip(r+np.asarray(u)*stride,lo,hi)
                    stream.append(r)
                goals.append(mpc.tail(np.asarray(stream),mpc.HORIZON+mpc.TAIL))
            def trial():
                f=np.zeros((2,9))
                _,y=engine.predict.trajectories(goals,mpc.AIM,.48,feedback=f)
                face=alignment(engine,[g[:mpc.HORIZON] for g in goals])
                return {kind:dict(camera_nominal_legal=bool(np.isfinite(y[i,0])),
                                  camera_separation=bool(f[i,1]>.5),camera_departure=f[i,2:5].tolist(),
                                  camera_radius_m=float(f[i,8]),camera_alignment=float(face[i]) if np.isfinite(face[i]) else None,
                                  actual_legal=row["arms"][kind]["landed"],
                                  actual_alignment=row["arms"][kind]["actual"]["alignment"])
                        for i,kind in enumerate(names)}
            result=trial()
            error=np.r_[engine.twin.ball-table.ball,engine.twin.velocity-table.velocity]
            perturbations={}
            if factors:
                twin=engine.twin
                assert twin.model.nq==table.model.nq and twin.model.nv==table.model.nv
                camera_data=mujoco.MjData(twin.model)
                mujoco.mj_copyData(camera_data,twin.model,twin.data)
                p,w=twin._ball_joint,twin._ball_dof
                bounced=twin._received_bounce
                try:
                    for tag, body, ball in (("ball_rotation",False,"rotation"),
                                             ("ball",False,"exact"),
                                             ("body",True,"camera"),
                                             ("both",True,"exact")):
                        mujoco.mj_copyData(twin.data,twin.model,table.data if body else camera_data)
                        if ball=="camera":
                            twin.data.qpos[p:p+7]=camera_data.qpos[p:p+7]
                            twin.data.qvel[w:w+6]=camera_data.qvel[w:w+6]
                        elif ball=="exact":
                            twin.data.qpos[p:p+7]=table.data.qpos[p:p+7]
                            twin.data.qvel[w:w+6]=table.data.qvel[w:w+6]
                        else:
                            twin.data.qpos[p+3:p+7]=table.data.qpos[p+3:p+7]
                            twin.data.qvel[w+3:w+6]=table.data.qvel[w+3:w+6]
                        twin._received_bounce=table._received_bounce if ball=="exact" else bounced
                        mujoco.mj_forward(twin.model,twin.data)
                        perturbations[tag]=trial()
                finally:
                    mujoco.mj_copyData(twin.data,twin.model,camera_data)
                    twin._received_bounce=bounced
            return dict(id=ident,causal_optical_fit=True,sensed=None if x is None else x.tolist(),
                match_prefix_events=len(expected),bounced=bool(engine.bounced),
                evaluator_initial_pv_error=error.tolist(), arms=result, evaluator_factors=perturbations)
        finally:
            engine.twin.close()
            render.close()
            table.close()


def run(output,factors):
    doc=dict(rows=[],sources={str(p):_hash(p) for _,a,b in CHECKS for p in (a,b)},
             scope="Causal camera twin after exact archived control prefix. Physical ball/body state is used only for error reporting and optional labeled evaluator ablations, after scoring the optically initialized proposed motions. Not a held policy test.")
    for ix,a,b in CHECKS:
        new=check(ix,a,b,factors)
        doc["rows"].append(new)
        _write(output,doc)
        print(ix,new,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",nargs="?",type=Path,default=ROOT/"optical-practice-check.json")
    p.add_argument("--factors",action="store_true")
    a=p.parse_args()
    run(a.out,a.factors)
