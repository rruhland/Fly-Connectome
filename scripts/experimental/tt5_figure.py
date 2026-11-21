"""Plot the scheduled hall points from the three matched conditions."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .tt4_transfer_trial import ROOT, _read


def figure(output):
    folder = ROOT / "artifacts" / "tt5"
    old = _read(folder / "temporal-baseline.json.gz")
    new = _read(folder / "hall-qualification.json.gz")
    series = [
        ("Previous controller · mini", old, "temporal:legacy_mini", "#767e8d"),
        ("New controller · Pong only", new, "rally:pong_only", "#547bb5"),
        ("New controller · Pong + mini", new, "rally:legacy_mini", "#12796b"),
    ]
    fig, panels = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True)
    fig.subplots_adjust(top=.78, bottom=.12, hspace=.50, left=.09, right=.98)
    fig.suptitle("First-person hall returns on five unseen installations", x=.09, y=.97,
                 ha="left", fontsize=17, weight="bold")
    for axis, required, title in zip(panels, (1, 2),
                                     ("Points with an opening return", "Points with a second legal return")):
        for j, (label, doc, key, color) in enumerate(series):
            counts = [sum(p["player_returns"] >= required
                          for p in station["runs"][key]["points"])
                      for station in doc["courses"]["first_person"]]
            bars = axis.bar(np.arange(len(counts)) + (j - 1) * .24, counts, .21, color=color, label=label)
            axis.bar_label(bars, padding=2, fontsize=8)
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_ylim(0, 11.8)
        axis.set_yticks([0, 5, 10])
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e0e3e6")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(length=0)
    panels[0].legend(loc="lower left", bbox_to_anchor=(-.01, 1.22), ncol=3,
                     frameon=False, fontsize=9, handlelength=1, columnspacing=1.6)
    panels[1].set_xticks(range(5), [str(i + 1) for i in range(5)])
    panels[1].set_xlabel("Installation  ·  10 matched serves each", labelpad=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    figure(parser.parse_args().output)
