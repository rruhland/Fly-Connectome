"""Plot held-out table learning and paired Pong retention from a cycle receipt."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def draw(receipt, output):
    with gzip.open(receipt, "rt") as stream:
        record = json.load(stream)
    plot, (curve, retention) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    for axis in (curve, retention):
        axis.spines[["right", "top", "left"]].set_visible(False)
        axis.tick_params(length=0, pad=6)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e5e7eb", lw=0.8)
    cold = "cold_clock" if "cold_clock" in record["stats"] else "scratch"
    palette = {"trained": "#007968", "fresh_goal": "#4074a4", cold: "#7d737f"}
    labels = {
        "trained": "Pong roles and geometry",
        "fresh_goal": "Pong roles only",
        cold: "From scratch",
    }
    matches = record["matches"]
    for condition in palette:
        values = np.array(
            [
                [point["legal_returns"] > 0 for point in match[condition]["points"]]
                for match in matches
            ]
        )
        total = values.cumsum(axis=1)
        n, points = total.shape
        x = np.arange(points + 1)
        boot = np.random.default_rng(76395).integers(n, size=(4000, n))
        lo, hi = np.quantile(total[boot].mean(axis=1), [0.025, 0.975], axis=0)
        curve.fill_between(
            x,
            np.r_[0, lo],
            np.r_[0, hi],
            color=palette[condition],
            alpha=0.11,
            linewidth=0,
        )
        curve.plot(
            x,
            np.r_[0, total.mean(axis=0)],
            color=palette[condition],
            lw=2.4,
            label=labels[condition],
        )
    curve.set(
        title="Learning at a new table",
        xlabel="Points played at this installation",
        ylabel="Cumulative points with a legal return",
        xlim=(0, points),
        ylim=(0, points),
    )
    curve.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    curve.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    curve.legend(frameon=False, loc="upper left", fontsize=9)
    native = record["native"]["comparison"]
    categories = ["First returns", "Later returns", "Earned points"]
    metrics = [
        "first_return_fraction",
        "later_return_fraction",
        "earned_point_fraction",
    ]
    x = np.arange(3)
    for side, label, offset, color in (
        ("before", "Before table tennis", -0.18, "#a3b5c6"),
        ("after", "After table tennis", 0.18, "#007968"),
    ):
        rates = [100 * native[side][key] for key in metrics]
        bars = retention.bar(x + offset, rates, 0.34, label=label, color=color)
        retention.bar_label(
            bars, labels=[f"{rate:.0f}%" for rate in rates], fontsize=8, padding=2
        )
    retention.set(
        title="Back in native Pong",
        ylabel="Rate (%)",
        ylim=(0, 110),
        xticks=x,
        xticklabels=categories,
    )
    retention.legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12)
    )
    plot.subplots_adjust(left=0.078, right=0.975, top=0.87, bottom=0.25, wspace=0.29)
    plot.text(
        0.078,
        0.04,
        f"Left: mean across {len(matches)} independent table installations, with 95% match-bootstrap intervals. "
        f"Right: {native['before']['starts']} paired Pong points.",
        fontsize=9,
        color="#494e52",
    )
    plot.savefig(output, dpi=170, facecolor="white")
    plt.close(plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    draw(options.receipt, options.output)
