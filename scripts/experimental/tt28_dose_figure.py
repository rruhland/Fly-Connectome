"""Plot pooled physical outcomes at each point in the solo-practice sequence."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tt4_transfer_trial import _read
from .tt28_dose import DOSES, MOUNTS, OUTPUT


def run():
    source = [_read(OUTPUT / f"dose-solo-{m}.json.gz") for m in MOUNTS]
    target = [_read(OUTPUT / f"dose-target-{m}.json.gz") for m in MOUNTS]
    fig, ax = plt.subplots(1, 2, figsize=(10.7, 4.5))
    colors = ["#226495", "#ad511d", "#347353"]
    x = np.arange(len(DOSES))
    for data, title, color in zip(source, ("Twisted grip", "Canted grip"), colors[:2], strict=True):
        matrix = np.array([[r["results"][str(d)]["streak"] for r in data["held"]] for d in DOSES])
        ax[0].plot(x, np.median(matrix, axis=1), marker="o", linewidth=2.2, label=title, color=color)
    groups = [
        ("perturbed", "Variations of practiced shots"),
        ("nearby", "Nearby recorded shots"),
        ("unrestricted", "Unrestricted recorded shots"),
    ]
    for (g, title), color in zip(groups, colors, strict=True):
        ratios = []
        for dose in DOSES:
            rows = [r for d in target for r in d["rows"] if r["group"] == g]
            ratios.append(100 * np.mean([r["results"][str(dose)]["legal"] for r in rows]))
        ax[1].plot(x, ratios, marker="o", linewidth=2.2, color=color, label=f"{title} (n={len(rows)})")
    for a, title, ylabel in zip(
        ax,
        ("Solo bouncing", "Regulation receiving"),
        ("Median consecutive bounces", "Legal first returns (%)"),
        strict=True,
    ):
        a.set_title(title, loc="left", fontsize=15, weight="bold")
        a.set_xticks(x, [str(d) for d in DOSES])
        a.set_xlabel("Solo practice drops")
        a.set_ylabel(ylabel)
        a.grid(axis="y", alpha=0.22)
        a.spines[["top", "right"]].set_visible(False)
        a.set_ylim(bottom=-0.6)
        a.legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(
        "What carries over from longer bouncing practice?", x=0.06, ha="left", fontsize=18, weight="bold"
    )
    fig.text(
        0.06,
        0.008,
        "Same held starts at each checkpoint. Receiving rates combine both grip changes. Each launch was tested once per grip.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.90])
    fig.savefig(OUTPUT / "bounce-practice.png", dpi=165, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
