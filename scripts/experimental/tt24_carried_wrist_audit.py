"""Replay a whole mixed-body wrist cohort using the portable policy bundle as the live controller."""
import gzip
import json
from time import perf_counter

import mujoco
import numpy as np

from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_posture import WidePosture
from .tt24_ready import prepare_step


def main():
    policy=CarriedWristPolicy(BUNDLE)
    with gzip.open(ROOT/"ready-body-combined.json.gz","rt") as f:
        expected=json.load(f)
    elapsed=[]
    audited=[]
    arenas=[]
    try:
        for body in ("stock","alternate"):
            arenas.append((WidePosture(view="wrist",width=192,height=144,max_steps=360,arm=body),
                           BounceArena(view="none",arm=body)))
        for row in expected:
            body=row["body"]
            seed=row["seed"]
            regime="changed" if body else "standard"
            arena,server=arenas[body]
            server.reset(seed=seed,split="held",regime=regime,render=False)
            location=server._ball_position.copy()
            if not np.array_equal(location,row["release_xyz"]):
                raise ValueError("launcher position differs")
            arena.reset(seed=seed,split="held",regime=regime,render=False)
            arena._ball_position[:]=location
            arena._ball_velocity[:]=0.
            mujoco.mj_forward(arena._model,arena._data)
            obs=arena._observation(True)
            policy.begin_release(obs)
            if policy.calibration_index!=row["selected"]:
                raise ValueError("bundle recognized another calibration")
            lead=row["preparation_ticks"]
            for tick,recorded in enumerate(row["actions"]):
                start=perf_counter()
                a=policy.act(obs)
                elapsed.append(perf_counter()-start)
                if not np.array_equal(a,recorded):
                    raise ValueError((seed,tick,"action mismatch",np.max(abs(a-recorded))))
                if tick<lead:
                    prepare_step(arena,a,location)
                    if tick+1==lead:
                        arena._ball_velocity[:]=server._ball_velocity
                        arena._ball_spin[:]=server._ball_spin
                        mujoco.mj_forward(arena._model,arena._data)
                    obs=arena._observation(True)
                else:
                    obs,_,done=arena.step(a)
                    if done!=(tick+1==len(row["actions"])):
                        raise ValueError("terminal tick differs")
            metrics=arena.judge_metrics()
            if metrics["longest_consecutive_rebounds"]!=row["run"] or metrics["rebounds"]!=row["rebounds"]:
                raise ValueError("events differ")
            audited.append(dict(seed=seed,body=body,frames=len(row["actions"]),streak=row["run"]))
    finally:
        for arena,server in arenas:
            arena.close()
            server.close()
    receipt=dict(cases=audited,exact_commands=len(elapsed),inference_ms_percentiles=dict(zip(
                 ("p50","p95","p99"),(1000*np.percentile(elapsed,[50,95,99])).tolist())),
                 inputs="wrist RGB and joint encoders",measured="optical observer and motor inference, excluding simulation and rendering")
    (ROOT/"carried-wrist-audit.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt["exact_commands"],receipt["inference_ms_percentiles"])


if __name__=="__main__":
    main()
