"""Plot the matched receiving-table tests with the denominator on each panel."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCES = {
    "legacy_mini": ("Pong → original mini", "#257b68", "o"),
    "first_person_mini": ("Pong → first-person mini", "#3984c6", "s"),
    "pong_only": ("Pong only", "#d18631", "D"),
    "fresh": ("Fresh", "#6b6c7d", "^"),
}


def fraction(path, name):
    with gzip.open(path, "rt") as f:
        doc = json.load(f)
    vals = [r["player_returns"] for c in doc["courses"]["first_person"] for r in c["runs"][name]["points"]]
    return np.count_nonzero(vals), len(vals)


def figure(directory, output):
    groups = [("native-study-projected-control", "Projected RGB"),
              ("native-studio-transfer", "Shaded studio"),
              ("native-hall-transfer", "Shaded hall")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in axes:
        ax.set_ylim(0, 106)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color="#e0e4e8")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    for idx, (source, (label, color, marker)) in enumerate(SOURCES.items()):
        pts = [fraction(directory / (key + ".json.gz"), source) for key, _ in groups]
        x = np.arange(len(groups)) + (idx - 1.5) * .055
        y = [100 * n / den for n, den in pts]
        axes[0].plot(x, y, "-"+marker, color=color, lw=1.6, ms=6, label=label)
    axes[0].set_title("Same 12 points and motor wirings", loc="left", weight="bold")
    axes[0].set_xticks(range(3), [name for _, name in groups])
    axes[0].set_ylabel("Points with a legal return")
    for idx, source in enumerate(("legacy_mini", "first_person_mini", "pong_only")):
        n, den = fraction(directory / "native-hall-confirmation.json.gz", source)
        value = 100 * n / den
        axes[1].bar(idx, value, width=.59, color=SOURCES[source][1])
        axes[1].text(idx, value + 3, f"{n}/{den}", ha="center", weight="bold")
    axes[1].set_xlim(-.65, 2.65)
    axes[1].set_title("40 new points in the shaded hall", loc="left", weight="bold")
    axes[1].set_xticks(range(3), ["Original\nmini", "First-person\nmini", "Pong\nonly"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Legal returns from the player's first-person cameras", ha="left", x=.07,
                 size=16, weight="bold")
    fig.subplots_adjust(left=.08, right=.985, bottom=.23, top=.78, wspace=.30)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    figure(args.directory, args.output)
