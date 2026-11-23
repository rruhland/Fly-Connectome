"""Plot the native-source replication results using physical wrist trajectories."""
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tt24_donor_trial import ROOT


def main():
    banks=[]
    for replica in range(6):
        with gzip.open(ROOT/f"reference-bank-{replica}.json.gz","rt") as f:
            banks.append(json.load(f))
    fig,axes=plt.subplots(1,2,figsize=(9,4.1),layout="constrained")
    for ax,body,label in zip(axes,("stock","alternate"),("Stock arm · 48 drops","Changed arm · 24 drops")):
        scores={mode:np.array([[r["run"] for r in d["groups"][body]["by_alpha"][str(d["selections"][mode])]]
                               for d in banks]) for mode in banks[0]["selections"]}
        y=np.stack([scores[m].mean(axis=1) for m in ("shuffled","native")])
        for a,b in y.T:
            ax.plot([0,1],[a,b],lw=1.1,color="#739690",alpha=.68,zorder=0)
        for x,c in ((0,"#67737c"),(1,"#087b66")):
            ax.scatter(np.full(6,x),y[x],s=31,color=c,alpha=.84)
        ax.plot([0,1],y.mean(axis=1),color="#192c34",lw=2,marker="D",markersize=7,label="Six-replica mean")
        ax.set_xticks([0,1],["Shuffled Pong credit","Intact Pong credit"])
        ax.set_xlim(-.25,1.25)
        ax.set_ylim(bottom=-.5,top=18)
        ax.set_title(label,loc="left")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.19)
    axes[0].set_ylabel("Mean longest streak on new wrist-controlled releases")
    axes[1].legend(frameon=False,loc="upper right")
    fig.suptitle("Transferring a coordinate reference learned in Pong",fontweight="bold")
    fig.savefig(ROOT/"reference-transfer.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main()
