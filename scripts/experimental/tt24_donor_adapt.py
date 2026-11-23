"""Use differences between self-trained donor policies to guide eight new real-time trials."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import numpy as np
import mujoco

from .tt24_bounce import BounceArena
from .tt24_donor_trial import ROOT, Native, _live
from .tt19_view_full_game import digest_json

CHECKPOINT=ROOT/"tt24_donor-refined.npz"
ARCHIVE=ROOT/"tt24_donor-refined.json.gz"
PLAN=ROOT/"donor-adapt-plan.json"
SCHEDULE_SEED=1249462219


def historical_basis(checkpoint, shuffle=False):
    with gzip.open(ARCHIVE,"rt") as f:
        history=json.load(f)
    x=np.concatenate([np.asarray(gen["development_weights"]) for gen in history["updates"]]).reshape(-1,100)
    # Each column is actual repeated success on one donor launch. Geometry,
    # ball parameters and reward information from the new arm stay out of this step.
    y=np.concatenate([np.asarray(gen["development"])[:,:,:,1].transpose(1,0,2).reshape(3,-1)
                      for gen in history["updates"]])
    x-=checkpoint.ravel()
    dist=np.linalg.norm(x,axis=1)
    radius=float(np.median(dist))
    attention=np.exp(-dist**2/(2*(2*radius)**2))
    if shuffle:
        rng=np.random.default_rng(5139012)
        y=y[rng.permutation(len(y))]
    n=attention.sum()
    x=x-attention@x/n
    y=y-attention@y/n
    covariance=x.T@(attention[:,None]*x)
    slopes=np.linalg.solve(covariance+.05*np.trace(covariance)/100*np.eye(100),
                           x.T@(attention[:,None]*y))
    # Donor worlds differed in which parameter updates improved their reward.
    slopes-=slopes.mean(axis=1,keepdims=True)
    u,s,_=np.linalg.svd(slopes,full_matrices=False)
    return u[:,:2].T.reshape(2,5,20)*radius,dict(radius=radius,
           fraction_of_directional_variance=float((s[:2]@s[:2])/(s@s)),examples=len(x))


def launch(arena,native,w,seed,verify):
    row=_live(native,arena,w,seed,"standard","public")
    if verify:
        other=_live(native,arena,w,seed,"standard","replay",tape=row["actions"])
        if digest_json(row)!=digest_json(other):
            raise ValueError("unreproduced closed-loop episode")
    judge=row["metrics"]
    return dict(seed=seed,run=judge["longest_consecutive_rebounds"],score=judge["rebounds"],
                ticks=judge["steps"],actions=row["actions"] if verify else None)


def register():
    rng=np.random.default_rng(SCHEDULE_SEED)
    groups=[]
    # Different wiring is handled by a separate experiment. Here the camera
    # works normally and each ballast/physics configuration stays unannounced.
    for gravity,ball_mass,regime in [(1.,1.,0),(.78,1.24,1),(1.26,.83,0),(.87,.78,1)]:
        schedule=np.tile(np.arange(4),2)
        rng.shuffle(schedule)
        groups.append(dict(gravity=gravity,mass=ball_mass,weak_drive=regime,
                           practice=rng.integers(1,2**31-1,8).tolist(),
                           held=rng.integers(1,2**31-1,12).tolist(),order=schedule.tolist()))
    doc=dict(seed=SCHEDULE_SEED,groups=groups,primary="mean longest consecutive rebounds on 12 fresh drops after 8 distinct online practice drops per learner and setting",
             selection="bounded reward-weighted interpolation of native donor parameter perturbations",
             controls=["label_shuffle","isotropic","frozen","stationary"])
    text=json.dumps(doc,indent=2)+"\n"
    if PLAN.exists() and PLAN.read_text()!=text:
        raise RuntimeError("registration changed")
    PLAN.write_text(text)
    return groups


def assess(index,group):
    with np.load(CHECKPOINT,allow_pickle=False) as archive:
        weights=archive["weights"]
    learned,meta=historical_basis(weights)
    sham,sham_meta=historical_basis(weights,True)
    rand=np.linalg.qr(np.random.default_rng(index+730251).normal(size=(100,2)))[0].T.reshape(2,5,20)*meta["radius"]
    arena=BounceArena(view="wrist",width=192,height=144,max_steps=360)
    arena._model.opt.gravity[:]*=group["gravity"]
    arena._model.body_mass[arena._ball_body]*=group["mass"]
    arena._model.body_inertia[arena._ball_body]*=group["mass"]
    if group["weak_drive"]:
        arena._standard_gain*=.80
        arena._standard_bias*=.80
        arena._standard_force*=.80
    mujoco.mj_setConst(arena._model,arena._data)
    native=Native()
    try:
        outcomes={}
        for condition,basis in (("learned",learned),("label_shuffle",sham),("isotropic",rand),
                                ("frozen",None),("stationary",None)):
            practice=[]
            alpha=np.zeros(2)
            if basis is not None:
                moves=np.array(((1.,0.),(-1.,0.),(0.,1.),(0.,-1.)))
                for i,seed in enumerate(group["practice"]):
                    which=group["order"][i]
                    active=weights+np.tensordot(moves[which],basis,axes=1)
                    practice.append(launch(arena,native,active,seed,True))
                score=np.array([r["score"] for r in practice])
                order=np.array(group["order"])
                means=np.array([score[order==k].mean() for k in range(4)])
                pref=np.exp((means-means.max())/max(2.,.65*score.std()))
                alpha=pref@moves/pref.sum()
                active=weights+np.tensordot(alpha,basis,axes=1)
            else:
                active=weights if condition=="frozen" else np.zeros_like(weights)
            held=[launch(arena,native,active,seed,True) for seed in group["held"]]
            outcomes[condition]=dict(practice=practice,coefficients=alpha.tolist(),held=held,weights=active.tolist())
            print(index,condition,sum(r["run"] for r in held),flush=True)
    finally:
        native.working.cleanup()
        arena.close()
    with gzip.open(ROOT/f"donor-adapt-{index}.json.gz","wt") as f:
        json.dump(dict(group=index, meta=meta, sham_meta=sham_meta, verified=True, outcomes=outcomes),f,separators=(",",":"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register",action="store_true")
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    groups=register()
    if not args.register:
        with ProcessPoolExecutor(args.workers) as pool:
            for future in [pool.submit(assess,i,g) for i,g in enumerate(groups)]:
                future.result()
