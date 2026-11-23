"""Recalibrate the eyes of a reward-trained donor policy on a new physical arm."""
import gzip
import json
from hashlib import sha256
import numpy as np
from .tt24_donor_recovery import DonorPolicy
from .tt24_donor_assess import ShiftedBounce, CHECKPOINT
from .tt24_donor_trial import Native, ROOT
from .tt24_public_morph import ProbeEyes, calibrate, load_morph

FIT = ROOT/"donor-morph.npz"
FIT_RECEIPT = ROOT/"donor-morph-fit.json.gz"
PROBES=tuple(range(57983101,57983105))


class ShiftedEyes(ShiftedBounce):
    fixed_images=ProbeEyes.fixed_images

    def __init__(self):
        super().__init__()
        self.max_steps=72
        ProbeEyes.configure_reference_cameras(self)


def evaluate(arena,native,weights,seed,kind):
    obs=arena.reset(seed=seed,split="held",regime="changed")
    policy=DonorPolicy(native,weights)
    if kind in ("recalibrated","earlier_body"):
        policy.eye=load_morph(FIT) if kind=="recalibrated" else load_morph()
    actions=[]
    while True:
        command=policy.command(obs) if kind!="hold" else np.zeros(5)
        actions.append(command.tolist())
        obs,_,done=arena.step(command)
        if done:
            break
    judged=arena.judge_metrics()
    arena.reset(seed=seed,split="held",regime="changed",render=False)
    for act in actions:
        arena.step(act,render=False)
    if arena.judge_metrics()!=judged:
        raise ValueError("physical tape differs")
    return dict(seed=seed,run=judged["longest_consecutive_rebounds"],score=judged["rebounds"],
                ticks=judged["steps"],actions=actions)


def main():
    calibrate(FIT,FIT_RECEIPT,seeds=PROBES,arena_factory=ShiftedEyes,arm_name="unseen tilted donor stress body")
    seeds=np.random.default_rng(159746301).integers(1,2**31-1,16).tolist()
    with np.load(CHECKPOINT,allow_pickle=False) as data:
        weights=np.ascontiguousarray(data["weights"],dtype=np.float64)
    native=Native()
    arena=ShiftedBounce()
    arena.max_steps=450
    results={}
    try:
        for mode in ("recalibrated","stock","earlier_body","hold"):
            rows=[evaluate(arena,native,weights,s,mode) for s in seeds]
            print(mode,[r["run"] for r in rows],flush=True)
            results[mode]=rows
    finally:
        arena.close()
        native.working.cleanup()
    with gzip.open(ROOT/"donor-morph.json.gz","wt") as out:
        json.dump(dict(seeds=seeds,probes=PROBES,verified=True,horizon=450,
                       motor_sha256=sha256(CHECKPOINT.read_bytes()).hexdigest(),results=results),out,
                  separators=(",",":"))


if __name__=="__main__":
    main()
