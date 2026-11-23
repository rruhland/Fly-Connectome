"""Paired causal disturbance test for a controller previously learned on donor bodies."""
import argparse
from hashlib import sha256
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import _read, _write
from .tt24_bounce import BounceArena
from .tt24_bounce_perturb import KICKS, _case, _summary
from .tt24_donor_trial import Native, ROOT
from .tt24_public_body import FUSED_OUTPUT, load_public


class DonorPolicy:
    def __init__(self, native, weights):
        self.native = native
        self.weights = weights
        self.issued = np.zeros(5)
        self.eye = load_public(FUSED_OUTPUT)

    def command(self, obs):
        state = np.ascontiguousarray(self.eye.observe(obs),dtype=np.float64)
        action = np.zeros(5)
        self.native.command(state.ctypes.data,self.weights.ctypes.data,self.issued.ctypes.data,action.ctypes.data)
        return action


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,default=624402911)
    p.add_argument("--checkpoint",type=Path,default=ROOT/"tt24_donor-refined.npz")
    p.add_argument("--receipt",type=Path,default=ROOT/"tt24_donor-refined.json.gz")
    args=p.parse_args()
    receipt=_read(args.receipt)
    pin=sha256(args.checkpoint.read_bytes()).hexdigest()
    if "held_stock_summary" not in receipt or receipt["frozen_sha256"] != pin:
        raise ValueError("donor learning and independent stock evaluation must finish before this test")
    with np.load(args.checkpoint,allow_pickle=False) as state:
        weights=np.ascontiguousarray(state["weights"],dtype=np.float64)
    native=Native()
    arena=BounceArena(view="wrist",width=192,height=144,max_steps=450)
    rng=np.random.default_rng(args.seed)
    seeds=rng.integers(1,2**31-1,16).tolist()
    kicks=rng.permutation(np.tile(KICKS,(2,1)))
    cases=[]
    try:
        for seed,(axis,velocity) in zip(seeds,kicks,strict=True):
            delta=np.zeros(2)
            delta[int(axis)]=velocity
            launch=dict(seed=seed,velocity_delta_m_s=delta.tolist())
            cases.append(_case(arena,"stock","standard",launch,lambda:DonorPolicy(native,weights)))
    finally:
        arena.close()
        native.working.cleanup()
    summary=_summary(cases)
    _write(ROOT/"donor-recovery.json.gz",dict(checkpoint_sha256=pin,seed=args.seed,cases=cases,summary=summary))
    print(summary,flush=True)


if __name__=="__main__":
    main()
