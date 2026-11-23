"""Change gravity several times without restarting the five-joint wrist controller."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_broad_donors import fixed_seeds
from .tt24_context_residual import WINDOW, flight
from .tt24_donor_trial import ROOT, Native, FUSED_OUTPUT, load_public
from .tt24_donor_assess import CHECKPOINT
from .tt24_launch_context import draw

# One unannounced change every four seconds, then retain the last gravity.
GRAVITIES=(1.,.76,1.25,.86,1.12,1.)
CAP=1800
SEGMENT=240
SEEDS=fixed_seeds(12,482119045)


def episode(arena,native,weights,seed,mode,nominal):
    arena._model.opt.gravity[:]=nominal
    obs=arena.reset(seed=seed,split="held",render=True)
    sensor=load_public(FUSED_OUTPUT)
    issued=np.zeros(5)
    positions,visible=deque(maxlen=WINDOW),deque(maxlen=WINDOW)
    recent=deque(maxlen=5)
    estimate=1.
    tape,trace=[],[]
    for tick in range(CAP):
        ratio=GRAVITIES[min(tick//SEGMENT,len(GRAVITIES)-1)] if mode!="constant" else 1.
        arena._model.opt.gravity[:]=nominal*ratio
        gaps=sensor.gaps
        state=sensor.observe(obs).copy()
        positions.append(state[:3].copy())
        visible.append(gaps==sensor.gaps)
        if tick%2==0 and mode=="optical":
            measured=flight(positions,visible)
            if measured is not None:
                # Reference is the measured median from prior stock optical trials.
                recent.append(np.clip(-measured[0]/9.28,.62,1.40))
                estimate=.65*estimate+.35*np.median(recent)
        if mode=="oracle":
            estimate=ratio
        if tick%15==0:
            trace.append((tick,ratio,estimate))
        # Ballistic similarity under a change of acceleration: velocities scale as sqrt(g).
        # The readout still chooses all motor commands. The inferred factor has no task labels.
        state[3:6]/=np.sqrt(estimate)
        a=np.zeros(5) if mode=="stationary" else draw(native,state,weights,issued)
        tape.append(a.tolist())
        obs,_,done=arena.step(a)
        if done:
            break
    metrics=arena.judge_metrics()
    arena._model.opt.gravity[:]=nominal
    arena.reset(seed=seed,split="held",render=False)
    for tick,a in enumerate(tape):
        ratio=GRAVITIES[min(tick//SEGMENT,len(GRAVITIES)-1)] if mode!="constant" else 1.
        arena._model.opt.gravity[:]=nominal*ratio
        _,_,done=arena.step(a,render=False)
        if done != (tick+1==len(tape)):
            raise ValueError("live and replay end on different ticks")
    if digest_json(metrics)!=digest_json(arena.judge_metrics()):
        raise ValueError("physical event mismatch")
    arena._model.opt.gravity[:]=nominal
    return dict(seed=seed,actions=tape,measurements=trace,score=metrics["rebounds"],
                run=metrics["longest_consecutive_rebounds"],ticks=len(tape),
                impact_times=[a["impact_time"] if "impact_time" in a else a["contact_time"] for a in metrics["rebound_events"]])


def qualify(mode):
    arena=BounceArena(view="wrist",width=192,height=144,max_steps=CAP)
    nominal=arena._model.opt.gravity.copy()
    native=Native()
    try:
        with np.load(CHECKPOINT,allow_pickle=False) as d:
            weights=np.ascontiguousarray(d["weights"])
        rows=[episode(arena,native,weights,s,mode,nominal) for s in SEEDS]
        with gzip.open(ROOT/f"continuous-gravity-{mode}.json.gz","wt") as f:
            json.dump(dict(gravities=GRAVITIES,segment=SEGMENT,cap=CAP,rows=rows),f,separators=(",",":"))
        print(mode,[r["run"] for r in rows],flush=True)
    finally:
        arena.close()
        native.working.cleanup()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(qualify,("frozen","optical","oracle","constant","stationary")))
