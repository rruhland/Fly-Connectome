"""Intervene mid-flight on a mixed-body release and compare the bundle to its own original commands."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import mujoco
import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_posture import WidePosture
from .tt24_ready import prepare_step

KICK=72


def simulate(arena,server,case,velocity,variant,policy,tape=None):
    index=case["body"]
    regime="changed" if index else "standard"
    seed=case["seed"]
    lead=case["preparation_ticks"]
    server.reset(seed=seed,split="held",regime=regime,render=False)
    arena.reset(seed=seed,split="held",regime=regime,render=False)
    loc=server._ball_position.copy()
    arena._ball_position[:]=loc
    arena._ball_velocity[:]=0.
    mujoco.mj_forward(arena._model,arena._data)
    obs=arena._observation(variant=="live")
    if variant=="live":
        policy.begin_release(obs)
    records=[]
    prefix_contacts=None
    for t in range(lead+arena.max_steps):
        if t==lead+KICK:
            prefix_contacts=arena._paddle_contacts
            arena._ball_velocity[:2]+=velocity
            mujoco.mj_forward(arena._model,arena._data)
        if variant=="live":
            action=policy.act(obs)
        elif variant=="check":
            action=tape[t]
        elif t<len(case["actions"]):
            action=case["actions"][t]
        else:
            action=np.zeros(5)
        if t<lead+KICK and not np.array_equal(case["actions"][t],action):
            raise ValueError("the counterfactual changed a command before the intervention")
        records.append(np.asarray(action).tolist())
        if t<lead:
            prepare_step(arena,np.array(action),loc)
            if t+1==lead:
                arena._ball_velocity[:]=server._ball_velocity
                arena._ball_spin[:]=server._ball_spin
                mujoco.mj_forward(arena._model,arena._data)
            obs=arena._observation(variant=="live")
        else:
            obs,_,done=arena.step(action,render=variant=="live")
            if done:
                break
    metrics=arena.judge_metrics()
    score=0
    if prefix_contacts is not None:
        ids=[e["impact_index"] for e in metrics["rebound_events"] if e["impact_index"]>prefix_contacts]
        for index,event in enumerate(ids):
            if event!=prefix_contacts+index+1:
                break
            score+=1
    return dict(actions=records,after=score,reached=prefix_contacts is not None,
                terminal_tick=metrics["steps"],events=digest_json(metrics))


def group(body):
    policy=CarriedWristPolicy()
    which="alternate" if body else "stock"
    arena=WidePosture(view="wrist",width=192,height=144,arm=which,max_steps=360)
    server=BounceArena(view="none",arm=which)
    with gzip.open(ROOT/"ready-body-combined.json.gz","rt") as f:
        all_cases=json.load(f)
    directions=np.random.default_rng(439053515).permutation(np.tile(np.eye(2).tolist()+(-np.eye(2)).tolist(),(8,1)))
    output=[]
    try:
        for i,case in enumerate(all_cases):
            if case["body"]!=body:
                continue
            velocity=directions[i]*.22
            result={}
            for variant in ("live","original_tape"):
                trial=simulate(arena,server,case,velocity,variant,policy)
                replay=simulate(arena,server,case,velocity,"check",policy,tape=trial["actions"])
                if trial!=replay:
                    raise ValueError("counterfactual failed replay")
                result[variant]=trial
            output.append(dict(seed=case["seed"],velocity=velocity.tolist(),conditions=result))
        with gzip.open(ROOT/f"carry-disturbance-{which}.json.gz","wt") as f:
            json.dump(dict(kick_tick=KICK,body=which,rows=output),f,separators=(",",":"))
    finally:
        arena.close()
        server.close()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(group,(0,1)))
