"""Carry learned visual calibrations and an online effective-motor observer across body switches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil
from .tt24_effects import transmission
from .tt24_motor_recall import MotorRecall, STOCK, calibrate
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_public_morph import load_morph
from .tt24_view_recall import descriptor
from .tt24_world_arm import DIR, SOURCE


# New geometry and unfamiliar motor wiring arrive separately. Both optical
# calibrations came from earlier self-issued probes. Wiring is new for each stream.
BODIES = (0,0,1,1,1,1,1,0,0,0,1,1,0,1,0)
MODES = ("combined","visual","frozen","familiar")
SEED = 136701974


def altered_wiring(index, tick):
    return (index==3 and tick>=24) or 4<=index<=9 or (index==10 and tick<24)


def templates(arenas, archived=False):
    rows = []
    pairs = (("public-body.json.gz","standard"),("public-morph-calibration.json.gz","changed"))
    for arena,(path,regime) in zip(arenas,pairs,strict=True):
        with gzip.open(DIR/path,"rt") as f:
            rec = json.load(f)
        seeds = [row["seed"] for row in rec["data"][:4]]
        split="practice" if archived and regime=="standard" else "held"
        rows.append([descriptor(arena.reset(seed=seed,split=split,regime=regime)["rgb"]) for seed in seeds])
    return np.array(rows)


def evaluate(arenas, starts, mix, mode, references, controller, observers=None):
    result = []
    for index,(body,seed) in enumerate(zip(BODIES,starts,strict=True)):
        arena = arenas[body]
        regime = "changed" if body else "standard"
        obs = arena.reset(seed=seed, split="held",regime=regime)
        likelihood = np.max(references @ descriptor(obs["rgb"]),axis=1)
        selected = int(np.argmax(likelihood)) if mode!="frozen" else 0
        pupil = WristPupil.load(SOURCE)
        pupil.sensor = load_morph() if selected else load_public(FUSED_OUTPUT)
        if mode=="local_model":
            controller.observer,controller.scale=observers[selected]
        controller.reset(obs)
        tape,requests = [],[]
        while True:
            desired = pupil.command(obs)
            do_learn = mode in ("combined","local_model")
            command = controller.actuate(desired) if do_learn else desired
            physical = mix@command if altered_wiring(index,len(tape)) and mode!="familiar" else command
            requests.append(command.tolist())
            tape.append(physical.tolist())
            obs, _, done = arena.step(physical)
            if do_learn:
                controller.observe(command,obs,pupil)
            if done:
                break
        judge = arena.judge_metrics()
        arena.reset(seed=seed,split="held",regime=regime,render=False)
        for physical in tape:
            arena.step(physical,render=False)
        if arena.judge_metrics()!=judge:
            raise ValueError("mixed course replay diverged")
        result.append(dict(seed=seed,body=body,observation_model=selected,
                           run=judge["longest_consecutive_rebounds"],total=judge["rebounds"],
                           ticks=len(tape),adaptation=controller.events,requests=requests,actions=tape))
    return result


def group(index,seed,local):
    rng=np.random.default_rng(seed)
    starts=rng.integers(1,2**31-1,len(BODIES)).tolist()
    mix=transmission(rng)
    arenas=[BounceArena(view="wrist",width=192,height=144,max_steps=360,arm=a)
            for a in ("stock","alternate")]
    observer,scale,_=calibrate(STOCK)
    try:
        references=templates(arenas,archived=local)
        observers,charges=None,[]
        if local:
            from .tt24_compound_observer import fit
            for arena in arenas:
                arena.max_steps=450
            observers,charges=fit(arenas)
            for arena in arenas:
                arena.max_steps=360
        modes={}
        for mode in (("local_model","combined","visual","familiar") if local else MODES):
            rows=evaluate(arenas,starts,mix,mode,references,MotorRecall(observer,scale),observers)
            modes[mode]=rows
            print(index,mode,[r["run"] for r in rows],flush=True)
    finally:
        for arena in arenas:
            arena.close()
    stem="compound-observer" if local else "compound-carry"
    with gzip.open(DIR/f"{stem}-{index}.json.gz","wt") as out:
        json.dump(dict(seed=seed, wiring=mix.tolist(), starts=starts, bodies=BODIES, verified=True,
                       nominal_calibration=charges,results=modes),out,separators=(",",":"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--streams",type=int,default=4)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--local",action="store_true")
    args=p.parse_args()
    seeds=np.random.default_rng(SEED+int(args.local)).integers(1,2**31-1,args.streams).tolist()
    with ProcessPoolExecutor(args.workers) as pool:
        for result in [pool.submit(group,i,seed,args.local) for i,seed in enumerate(seeds)]:
            result.result()
