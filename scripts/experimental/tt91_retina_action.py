"""One fixed ordinary TT75 live point to verify a faster crop renderer's actions."""
import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt91_retina_benchmark as bench
from . import tt75_observer_adapter as observer
from . import tt83_playable_course as playable
from .tt39_carry_joints import Carried
from .tt4_transfer_trial import _hash,_read,_write

school,mpc,course,season=bench.school,bench.mpc,bench.school.course,bench.school.season
ROOT=bench.ROOT
PLAN=ROOT/"action.protocol.json"
OUT=ROOT/"action.json.gz"


def freeze():
    p=_read(playable.PLAN)
    item=p["games"][0]
    required=(playable.PLAN,bench.PLAN,mpc.BANK,course.PRACTICE,course.OPTIC,season.CARRY,
              season.CARRY.with_suffix("").with_suffix(".npz"),season.fresh.RISK)
    _write(PLAN,dict(item=item,ready=p["ready"],settings=_read(bench.PLAN)["settings"],
                    sources={str(f):_hash(f) for f in required},
                    role="One preassigned known physically playable ordinary start (first from existing TT83 supplemental plan). New full point pair uses identical TT75 actor with original vs optimized renderer, identical launch, continuous independent partner requests and frozen stock motor checkpoint. No alternative-action development or broad game expansion."))


def trial(mode):
    p=_read(PLAN)
    if any(_hash(Path(f))!=h for f,h in p["sources"].items()):
        raise ValueError("Changed fixed causal action parity")
    evidence=_read(bench.OUT)
    compared=evidence["rows"]
    if not compared or any(row["parity"][mode]["optical_point_exact"]!=row["sensors"]["original"]["exposures"]
                           for row in compared):
        raise ValueError("Optical parity was not exact for the intended renderer")
    if not sum(x["sensors"][mode]["sensor_cpu_s"] for x in compared)<sum(
        x["sensors"]["original"]["sensor_cpu_s"] for x in compared):
        raise ValueError("There is no demonstrated sensor speedup")
    c,shot=(p["item"][k] for k in ("card","shot"))
    rank=season.fresh.shot.Coverage(_read(season.fresh.RISK))
    cp=_read(season.CARRY)
    actor=observer.actor_class(season.Player)
    ready=np.array(p["ready"])
    d=dict(plan=_hash(PLAN),
           optical_benchmark_prefix=school.digest_json(compared),
           optical_calibration_ids=[x["case"]["id"] for x in compared],
           modes=("original",mode),rows={})
    with np.load(mpc.BANK) as f:raw=mpc.DeepField(f,"local")
    with mpc.centered(ready),school.enabled():
        for name in d["modes"]:
            a=mpc.arena()
            wide,rig,stride,lo,hi=mpc.resources(a)
            x=None
            try:
                with patch.object(mpc,"Fragments",lambda f,e:Carried(f,e,practice=[course.PRACTICE],held={c["id"]})):
                    x=actor(mpc.BANK,raw,coverage=rank,variant="elastic",budget=28)
                course.resume(x,season.CARRY.with_suffix("").with_suffix(".npz"),cp)
                x.observer_home,x.observer_raw=ready,raw
                x.observation_enabled=True
                with patch.object(observer,"Retina",bench.FACTORIES[name]):
                    x.exposures=observer.ExposureSession(a,rig,wide,p["settings"])
                began=process_time()
                r=mpc.play(a,wide,rig,stride,lo,hi,ready,c,shot,raw,x,course.OPTIC,"frozen",240)
                r.update(cpu_s=process_time()-began,native_seconds=float(a.data.time),
                         observer=dict(x.exposures.stats(),selection=x.public.counts,
                                       decisions=x.observer_decisions),complete=r["events"][-1]["kind"]=="point")
                d["rows"][name]=r
                _write(OUT,d)
                print(name,c["id"],r["score"]["learner"],len(r["actions"]),r["observer"]["selection"],
                      round(r["cpu_s"],2),flush=True)
            finally:
                if x is not None:
                    if getattr(x,"exposures",None) is not None:
                        x.exposures.retina.crop.renderer._gl_context.make_current()
                        x.exposures.close()
                    x.close()
                wide.renderer._gl_context.make_current()
                wide.close();a.close()
    u,v=d["rows"].values()
    d["parity"]=dict(actions=u["actions"]==v["actions"],
                     referee=school.digest_json(u["events"])==school.digest_json(v["events"]),
                     placements=school.digest_json(u["placements"])==school.digest_json(v["placements"]),
                     dense_decisions=school.digest_json(u["observer"]["decisions"])==school.digest_json(v["observer"]["decisions"]),
                     complete_points=all(r["complete"] for r in d["rows"].values()))
    _write(OUT,d)
    print(d["parity"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","packed","culled","mono"))
    a=p.parse_args()
    freeze() if a.mode=="plan" else trial(a.mode)
