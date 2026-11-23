"""An untuned two-minute stress test of wrist-only control with repeated hidden gravity changes."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import mujoco
import numpy as np

from .tt4_transfer_trial import _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm, schedule
from .tt24_ready import prepare_step
from .tt24_tool_equilibrium import ToolEquilibrium


CAP=7200


def play(arm,policy,plan,launch,mode,tape=None):
    location,vel,spin=launch
    render=tape is None and mode=="live"
    env=arm.arena
    obs=arm.reset(plan,location,render)
    if render:
        policy.begin_release(obs)
    lead=plan["prep_ticks"]
    changes=dict(plan["changes"])
    records=[]
    events=[]
    for t in range(CAP+lead):
        if t-lead in changes:
            events.append(arm.switch_gravity(t-lead,changes[t-lead]))
        action=(tape[t] if tape is not None else policy.act(obs) if render else np.zeros(5))
        records.append(np.asarray(action).tolist())
        if t<lead:
            prepare_step(env,np.asarray(action),location)
            if t+1==lead:
                env._ball_velocity[:]=vel
                env._ball_spin[:]=spin
                mujoco.mj_forward(env._model,env._data)
            obs=env._observation(render)
        else:
            obs,_,done=env.step(action,render=render)
            if done:
                break
    return dict(metrics=env.judge_metrics(),actions=records,events=events)


def run(i):
    original=schedule(1571916104,6)
    plan=original[i]
    rng=np.random.default_rng(9461715+i)
    t=int(rng.integers(190,300))
    switches=[]
    while t<CAP:
        g=float(rng.uniform(.78,.92) if len(switches)%2==i%2 else rng.uniform(1.11,1.25))
        switches.append((t,g))
        t+=int(rng.integers(270,491))
    plan["changes"]=switches
    arm=InstalledArm(plan["body"])
    arm.arena.max_steps=CAP
    server=BounceArena(view="none")
    policy=ToolEquilibrium(BUNDLE,.0557)
    policy.mode="workspace_local"
    try:
        server.reset(seed=plan["launcher_seed"],split="held",render=False)
        launch=[x.copy() for x in (server._ball_position,server._ball_velocity,server._ball_spin)]
        results={}
        for mode in ("live","passive"):
            row=play(arm,policy,plan,launch,mode)
            repeat=play(arm,policy,plan,launch,mode,row["actions"])
            if digest_json(row)!=digest_json(repeat):
                raise ValueError("long course replay mismatch")
            results[mode]=row
        _write(ROOT/f"long-wrist-{i}.json.gz",dict(plan=plan,results=results))
    finally:
        arm.arena.close()
        server.close()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(run,[i for i in range(12) if not (ROOT/f"long-wrist-{i}.json.gz").exists()]))
