"""Plot actual paddle contacts and legal landings from a verified single point."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def draw(receipt, output):
    replay = json.loads(receipt.read_text())
    shots = replay["score"]["shots"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.8), sharex=True, sharey=True)
    fig.patch.set_facecolor("#fcfaf4")
    fig.suptitle(f"{len(shots)} consecutive legal returns", fontsize=20, y=0.97)
    for axis, field, title in zip(axes, ("contact_xyz", "landing_xyz"),
                                  ("Where each paddle hit the ball", "Where each return landed")):
        axis.set_facecolor("#fcfaf4")
        axis.add_patch(Rectangle((-.7625, -1.37), 1.525, 2.74,
                                 facecolor="#d8e7df", edgecolor="#47665e", linewidth=1.2))
        axis.plot([0, 0], [-1.37, 1.37], color="white", linewidth=1)
        axis.plot([-.915, .915], [0, 0], color="#253831", linewidth=2)
        for side, name, marker, color in ((1, "SO-101", "o", "#a23617"),
                                          (-1, "Practice partner", "+", "#164e82")):
            xy = np.array([shot[field][:2] for shot in shots if shot["side"] == side])
            axis.scatter(*xy.T, label=name, marker=marker, s=16 if side == 1 else 26,
                         c=color, alpha=.8, linewidth=.85)
        axis.set_title(title, fontsize=12)
        axis.set_aspect("equal")
        axis.set(xlim=(-.98, .98), ylim=(-1.65, 1.75), xlabel="Across table (m)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Along table (m) · arm at near end")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.receipt, args.output)
