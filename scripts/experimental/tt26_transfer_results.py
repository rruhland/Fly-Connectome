"""Summarize paired receiving, continuous wrist play, and cross-grip learning speed."""


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt26_grip_course import OUT


def read(name):
    return _read(OUT / (name + ".json.gz"))


def scores(rows):
    v = np.asarray(rows)
    return dict(
        n=len(v),
        total=int(v.sum()),
        median=float(np.median(v)),
        maximum=int(v.max()),
        ten=int((v >= 10).sum()),
    )


def summarize():
    speed = read("receiving-speed")["records"]
    optical = read("visual-acquisition")["records"]
    live = read("live-points")["cases"]
    solo = read("tool-recall")["rows"]
    if (len(speed), len(optical), len(live), len(solo)) != (8, 4, 48, 24):
        raise ValueError("Incomplete paired runs")
    summary = {"online": {}, "visual": {}, "live": {}, "solo": {}}
    series = {}
    for name in ("bounce", "cold", "shuffled"):
        v = np.array([[q["results"][name]["legal"] for q in r["history"]] for r in speed])
        if np.any(v.sum(axis=1) < 5):
            raise ValueError("The fifth-return milestone is censored")
        fifth = np.argmax(np.cumsum(v, axis=1) >= 5, axis=1) + 1
        summary["online"][name] = dict(
            first_five=fifth.tolist(),
            median=float(np.median(fifth)),
            blocks_of_16=v.reshape(len(v), 5, 16).sum(axis=(0, 2)).tolist(),
        )
        series[name] = np.cumsum(v[:, :64], axis=1).mean(axis=0)
        target = np.array([[q["results"][name]["legal"] for q in r["rows"]] for r in optical])
        summary["visual"][name] = dict(
            training=int(target[:, :96].sum()),
            held=int(target[:, 96:].sum()),
            held_episodes=int(target[:, 96:].size),
        )
    for name in ("bounce", "cold"):
        v = [r["results"][name] for r in live]
        summary["live"][name] = dict(
            first=sum(x["near"] > 0 for x in v),
            replies=sum(x["far"] for x in v),
            second=sum(x["near"] > 1 for x in v),
            points=len(v),
        )
    for name in ("cold", "adjustment", "tool_and_adjustment"):
        summary["solo"][name] = scores(
            [r["results"][name]["metrics"]["longest_consecutive_rebounds"] for r in solo]
        )
    exploration = read("exploration-transfer")
    changed = [c for c in exploration["cases"] if c["angle"]]
    curves = {
        k: np.mean([r[k] for c in changed for r in c["curves"]], axis=0)
        for k in ("learned", "uniform", "shuffled")
    }
    summary["exploration"] = dict(
        budgets=exploration["budgets"], held_per_group=24, curves={k: v.tolist() for k, v in curves.items()}
    )
    _write(OUT / "transfer-results.json", summary)
    colors = {"bounce": "#047057", "cold": "#5b6573", "shuffled": "#b06a10"}
    labels = {"bounce": "Bounce practice", "cold": "No bounce practice", "shuffled": "Shuffled rewards"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for name in ("bounce", "cold", "shuffled"):
        axes[0].plot(range(1, 65), series[name], color=colors[name], label=labels[name], lw=2)
    axes[0].set(
        title="Learning with the same grip",
        xlabel="Table-tennis practice shots",
        ylabel="Mean cumulative legal returns",
    )
    axes[0].legend(frameon=False)
    for key, color, label in (
        ("learned", colors["bounce"], "Bounce-informed joint order"),
        ("uniform", colors["cold"], "All 120 orders, averaged"),
        ("shuffled", colors["shuffled"], "Shuffled bounce rewards"),
    ):
        axes[1].plot(
            exploration["budgets"], curves[key] / 24 * 100, marker="o", ms=4, color=color, label=label
        )
    axes[1].set(
        title="Adapting to four unfamiliar grips",
        xlabel="Table-tennis calibration shots",
        ylabel="Legal returns on fresh feeds (%)",
        ylim=(0, 65),
    )
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "learning-speed.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    summarize()
