"""Play from arbitrary joint angles on a fixed external release, without a ready-motion tape."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import mujoco
import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_broad_donors import fixed_seeds
from .tt24_donor_trial import ROOT, Native, FUSED_OUTPUT, load_public
from .tt24_donor_assess import CHECKPOINT
from .tt24_launch_context import draw
from .tt24_posture import WidePosture


def prepare_step(arena,command,location):
    """A fixed launcher holds the ball while the five ordinary actuators move."""
    for _ in range(8):
        arena._control[:5]=np.clip(arena._control[:5]+command*arena._increments[:5],
                                   arena._control_low[:5],arena._control_high[:5])
        arena._ball_position[:]=location
        arena._ball_velocity[:]=0.
        arena._ball_spin[:]=0.
        mujoco.mj_step(arena._model,arena._data)
    arena._ball_position[:]=location
    arena._ball_velocity[:]=0.
    mujoco.mj_forward(arena._model,arena._data)


def episode(arena,server,seed,native,w,mu,mode,regime,*,alpha=None):
    server.reset(seed=seed,split="held",regime=regime,render=False)
    location=server._ball_position.copy()
    vel,spin=server._ball_velocity.copy(),server._ball_spin.copy()
    lead=int(np.random.default_rng(seed ^ 11940).integers(14,39))
    def reset():
        arena.reset(seed=seed,split="held",regime=regime,render=False)
        arena._ball_position[:]=location
        arena._ball_velocity[:]=0.
        mujoco.mj_forward(arena._model,arena._data)
    reset()
    sensor=load_public(FUSED_OUTPUT)
    obs=arena._observation(True)
    q=obs["joints"]
    ordinary=mode=="as_started"
    target=q if ordinary else mu
    if alpha is not None:
        target=q+alpha*(mu-q)
    if alpha is not None or mode in ("coherent","unweighted"):
        sensor.home=sensor.decoder.body(target)[12:15].copy()
    issued=(q-target).astype(float)
    tape=[]
    for tick in range(lead+arena.max_steps):
        action=draw(native,sensor.observe(obs),w,issued)
        tape.append(action.tolist())
        if tick<lead:
            prepare_step(arena,action,location)
            if tick+1==lead:
                arena._ball_velocity[:]=vel
                arena._ball_spin[:]=spin
                mujoco.mj_forward(arena._model,arena._data)
            obs=arena._observation(True)
        else:
            obs,_,done=arena.step(action)
            if done:
                break
    got=arena.judge_metrics()
    reset()
    for tick,a in enumerate(tape):
        if tick<lead:
            prepare_step(arena,np.asarray(a),location)
            if tick+1==lead:
                arena._ball_velocity[:]=vel
                arena._ball_spin[:]=spin
                mujoco.mj_forward(arena._model,arena._data)
        else:
            _,_,done=arena.step(a,render=False)
            if done != (tick==len(tape)-1):
                raise ValueError("held release ended at another tick")
    if digest_json(got)!=digest_json(arena.judge_metrics()):
        raise ValueError("physical referee mismatch")
    return dict(seed=seed,preparation_ticks=lead,release_xyz=location.tolist(),actions=tape,
                run=got["longest_consecutive_rebounds"],rebounds=got["rebounds"])


def qualify(mode):
    native=Native()
    try:
        with np.load(ROOT/"posture-source.npz",allow_pickle=False) as data:
            mu=data["initial_positions"].mean(axis=0) if mode=="unweighted" else data["center"]
        with np.load(CHECKPOINT,allow_pickle=False) as data:
            w=data["weights"]
        groups={}
        for i,(label,body,regime,n) in enumerate((("stock","stock","standard",16),("changed","alternate","changed",12))):
            server=BounceArena(view="none",arm=body)
            arena=WidePosture(view="wrist",width=192,height=144,max_steps=360,arm=body)
            try:
                groups[label]=[episode(arena,server,s,native,w,mu,mode,regime) for s in fixed_seeds(n,18211236+i)]
                print(label,mode,[x["run"] for x in groups[label]],flush=True)
            finally:
                server.close()
                arena.close()
        with gzip.open(ROOT/f"ready-{mode}.json.gz","wt") as f:
            json.dump(groups,f,separators=(",",":"))
    finally:
        native.working.cleanup()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(qualify,("coherent","as_started","motor_only","unweighted")))
