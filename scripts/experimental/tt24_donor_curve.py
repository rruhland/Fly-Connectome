"""Evaluate earlier donor-only checkpoints on the same untouched stock wrist launches."""
import gzip
import json
import numpy as np

from .tt24_bounce import BounceArena
from .tt24_donor_trial import ROOT, Native
from .tt24_donor_adapt import launch


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train=np.array([r["training_drops"] for r in rows])
    streak=np.array([[v["run"] for v in r["held"]] for r in rows])
    rng=np.random.default_rng(224109)
    indices=rng.integers(streak.shape[1],size=(3000,streak.shape[1]))
    low,high=np.quantile(streak[:,indices].mean(axis=-1),(.025,.975),axis=1)
    fig,(left,right)=plt.subplots(1,2,figsize=(9.8,3.8),layout="constrained")
    for ax in (left,right):
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y",color="#dce2e5")
        ax.set_axisbelow(True)
        ax.set_xlabel("Scored practice drops on the three donor arms")
        ax.tick_params(axis="both",labelsize=9)
    left.fill_between(train,low,high,color="#87cbb6",alpha=.45)
    left.plot(train,streak.mean(axis=1),color="#126b58",marker="o")
    left.set_ylabel("Mean longest streak on the new arm")
    left.set_ylim(bottom=0)
    right.plot(train,(streak>=10).mean(axis=1)*100,color="#4d5fa3",marker="o")
    right.set_ylabel("New-arm drops reaching ten rebounds (%)")
    right.set_ylim(-3,103)
    right.set_yticks([0,25,50,75,100])
    fig.suptitle("Transfer to wrist-only play before any motor practice on the target arm",fontweight="bold")
    fig.savefig(ROOT/"donor-curve.png",dpi=170)
    plt.close(fig)


def main():
    seeds=np.random.default_rng(656301798).integers(1,2**31-1,12).tolist()
    arena=BounceArena(view="wrist",width=192,height=144,max_steps=300)
    native=Native()
    rows=[]
    try:
        with np.load(ROOT/"tt24_donor-development-weights.npz",allow_pickle=False) as archive:
            for budget in (0,150,450,900,1800,3600,5400):
                i=int(np.flatnonzero(archive["training_drops"]<=budget)[-1])
                episodes=[launch(arena,native,archive["weights"][i],s,True) for s in seeds]
                row={k:int(archive[k][i]) for k in ("generations","training_drops","training_frames","validation_drops","validation_frames")}
                rows.append(row|{"budget":budget,"held":episodes})
                print(row["training_drops"],[s["run"] for s in episodes],flush=True)
    finally:
        arena.close()
        native.working.cleanup()
    with gzip.open(ROOT/"donor-curve.json.gz","wt") as out:
        json.dump(dict(seeds=seeds,verified=True,horizon=300,rows=rows),out,separators=(",",":"))
    figure(rows)


if __name__=="__main__":
    main()
