"""Learn which joint postures supported source play and transfer on arbitrary starts."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gzip
import json

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena, _sample_drop
from .tt24_broad_donors import fixed_seeds
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import DonorBounce, DESIGNS, Native, ROOT, _live, FUSED_OUTPUT, load_public
from .tt24_launch_context import draw


class WidePosture(BounceArena):
    @staticmethod
    def sample_drop(seed,split,regime):
        drop=_sample_drop(seed,split,regime)
        rng=np.random.default_rng(int(seed)^83510054)
        # Randomize all actuated start angles independently. Ball still launches over the physical blade.
        delta=rng.uniform(-1.,1.,5)*np.deg2rad([13.,9.,12.,11.,13.])
        return replace(drop,pose=tuple(np.r_[np.array(drop.pose[:5])+delta,drop.pose[5]]))


def fit():
    with gzip.open(ROOT/"tt24_donor-refined.json.gz","rt") as f:
        source=json.load(f)["held_donor"]
    examples=[]
    for i,group in enumerate(source):
        arena=DonorBounce(DESIGNS[i])
        try:
            for row in group:
                obs=arena.reset(seed=row["seed"],split="held",regime="changed",render=True)
                streak=row["conditions"]["clean"]["metrics"]["longest_consecutive_rebounds"]
                examples.append((obs["joints"].copy(),streak))
        finally:
            arena.close()
    q=np.stack([entry[0] for entry in examples])
    r=np.array([entry[1] for entry in examples])
    # Prevent a single long source game from deciding the posture distribution.
    importance=.20+.80*np.minimum(r,10.)/10.
    center=importance@q/importance.sum()
    d=q-center
    covariance=d.T@(importance[:,None]*d)/importance.sum()
    np.savez_compressed(ROOT/"posture-source.npz",center=center,covariance=covariance,initial_positions=q,streaks=r)


def correction(mode,q,center,covariance,decoder):
    if mode=="frozen":
        return np.zeros(5)
    if mode=="all_axes":
        return np.clip(center-q,-.24,.24)
    if mode=="shuffled":
        perm=[2,4,1,3,0]
        covariance=covariance[np.ix_(perm,perm)]
    eigen,vectors=np.linalg.eigh(covariance)
    relative=(q-center)@vectors
    # Apply a correction only outside the range of configurations experienced on donors.
    bound=2.25*np.sqrt(np.maximum(eigen,.004**2))
    delta=vectors@(np.clip(relative,-bound,bound)-relative)
    # Compare with a source manifold regularized using public kinematics to preserve starting reach.
    if mode in ("manifold","shuffled"):
        h=.002
        jac=np.stack([(decoder.body(q+h*v)[12:15]-decoder.body(q-h*v)[12:15])/(2*h)
                      for v in np.eye(5)],axis=1)
        delta=np.linalg.solve(np.eye(5)+jac.T@jac/.095**2,delta)
    return np.clip(delta,-.24,.24)


def evaluate(arena,native,w,seed,mode,center,covariance):
    obs=arena.reset(seed=seed,split="held")
    observer=load_public(FUSED_OUTPUT)
    delta=correction(mode,obs["joints"],center,covariance,observer.decoder)
    # issued tracks the actual actuator target relative to the new equilibrium.
    issued=-delta.copy()
    tape=[]
    while True:
        action=draw(native,observer.observe(obs),w,issued)
        tape.append(action.tolist())
        obs,_,done=arena.step(action)
        if done:
            break
    got=arena.judge_metrics()
    replay=_live(native,arena,w,seed,"standard","replay",tape=tape)
    if digest_json(got)!=digest_json(replay["metrics"]):
        raise ValueError("physical action replay mismatch")
    return dict(seed=seed,correction=delta.tolist(),streak=got["longest_consecutive_rebounds"],
                rebounds=got["rebounds"],pose=got["drop"]["pose"],actions=tape)


def test(mode):
    native=Native()
    try:
        with np.load(CHECKPOINT,allow_pickle=False) as d:
            w=np.ascontiguousarray(d["weights"])
        with np.load(ROOT/"posture-source.npz",allow_pickle=False) as d:
            center,cov=d["center"],d["covariance"]
        groups={}
        for name,cls,count,s in (("expanded",WidePosture,24,32515343),("ordinary",BounceArena,12,32515344)):
            arena=cls(view="wrist",width=192,height=144,max_steps=300)
            try:
                groups[name]=[evaluate(arena,native,w,seed,mode,center,cov) for seed in fixed_seeds(count,s)]
                print(mode,name,sum(v["streak"] for v in groups[name]),flush=True)
            finally:
                arena.close()
        with gzip.open(ROOT/f"posture-{mode}.json.gz","wt") as f:
            json.dump(groups,f,separators=(",",":"))
    finally:
        native.working.cleanup()


if __name__=="__main__":
    fit()
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(test,("manifold","shuffled","all_axes","frozen")))
