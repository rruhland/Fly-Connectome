"""Apply a previously learned camera timing model to a new reward-trained donor policy."""
from collections import deque
import gzip
import json
import numpy as np
from .tt24_bounce import BounceArena
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import Native, ROOT
from .tt24_latency import MAX_AGE, MODEL, SceneClock, LateOptics
from .tt24_latency_trial import latency_at, timing
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_wrist_egomotion import CALIBRATION, MAP


def play(arena,native,weights,seed,mode,build):
    primary=load_public(FUSED_OUTPUT)
    late=build()
    history=deque(maxlen=MAX_AGE+1)
    obs=arena.reset(seed=seed,split="held")
    issued=np.zeros(5)
    rng=np.random.default_rng(seed+813)
    ages,tape=[],[]
    while True:
        t=len(tape)
        history.appendleft(obs["rgb"])
        base=latency_at(seed,t)
        jitter=int(rng.choice((-1,0,1),p=(.12,.76,.12))) if base else 0
        age=0 if mode in ("intact","intact_clock") else min(t,len(history)-1,max(0,min(MAX_AGE,base+jitter)))
        public=dict(rgb=history[age],joints=obs["joints"])
        state=primary.observe(public)
        estimated=0
        if mode in ("clock","oracle","intact_clock"):
            pred=late.observe(public,age if mode=="oracle" else None)
            estimated=late.age
            if estimated:
                state=pred
        state=np.ascontiguousarray(state,dtype=np.float64)
        command=np.zeros(5)
        native.command(state.ctypes.data,weights.ctypes.data,issued.ctypes.data,command.ctypes.data)
        ages.append([age,estimated])
        tape.append(command.tolist())
        obs,_,done=arena.step(command)
        if done:
            break
    judge=arena.judge_metrics()
    arena.reset(seed=seed,split="held",render=False)
    for command in tape:
        arena.step(command,render=False)
    if arena.judge_metrics()!=judge:
        raise ValueError("physical replay mismatch")
    return dict(seed=seed,run=judge["longest_consecutive_rebounds"],score=judge["rebounds"],ages=ages,actions=tape)


def main():
    with np.load(MAP,allow_pickle=False) as src:
        theta,points=src["source_theta"],src["points"]
    with np.load(MODEL,allow_pickle=False) as src:
        params={k:src[k] for k in src.files}
    with np.load(CHECKPOINT,allow_pickle=False) as src:
        weights=np.ascontiguousarray(src["weights"],dtype=float)
    geometry=load_public(CALIBRATION).decoder
    def build():
        clock=SceneClock(geometry,theta,points,params["center"],params["offset"],params["scale"])
        return LateOptics(load_public(FUSED_OUTPUT),clock,{i:params[f"h{i}"] for i in range(1,MAX_AGE+1)})
    seeds=np.random.default_rng(171038922).integers(1,2**31-1,12).tolist()
    native=Native()
    arena=BounceArena(view="wrist",width=192,height=144,max_steps=300)
    outcomes,stats={},{}
    try:
        for mode in ("naive","clock","intact","oracle","intact_clock"):
            rows=[play(arena,native,weights,s,mode,build) for s in seeds]
            outcomes[mode]=rows
            if mode in ("clock","intact_clock"):
                stats[mode]=timing(rows)
            print(mode,[r["run"] for r in rows],flush=True)
    finally:
        native.working.cleanup()
        arena.close()
    with gzip.open(ROOT/"donor-latency.json.gz","wt") as dest:
        json.dump(dict(seeds=seeds,verified=True,results=outcomes,timing=stats,jitter_probability=.24),
                  dest,separators=(",",":"))


if __name__=="__main__":
    main()
