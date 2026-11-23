"""Plot successive motor changes beside independently measured trajectory coverage."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tt4_transfer_trial import _read
from .tt27_flight_feedback import OUT


def run():
    motor = _read(OUT / "motor-recurrence.json.gz")
    if any(len(s["held"]) != 6 for s in motor["streams"]) or len(motor["streams"]) != 2:
        raise ValueError("Incomplete sequential experiment")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.3, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.19, wspace=0.40)
    colors = ("#16736a", "#704498", "#4f5d68")
    x = np.arange(1, 7)

    def duration(row, branch):
        return (
            motor["warmup_ticks"]
            + len(row[branch]["probes"]) * (motor["pulse_move_ticks"] + motor["packet_settle_ticks"])
        ) * motor["tick_s"]

    for stream, label, color, marker in zip(
        motor["streams"], ("Carried, course 1", "Carried, course 2"), colors[:2], ("o", "s"), strict=True
    ):
        times = [duration(r, "active_probe") for r in stream["held"]]
        left.plot(
            x,
            times,
            label=label,
            c=color,
            linestyle="-" if marker == "o" else ":",
            linewidth=2,
            marker=marker,
            markerfacecolor="white" if marker == "s" else color,
            markersize=6,
        )
    left.plot(
        x,
        [duration(r, "reset_probe") for r in motor["streams"][0]["held"]],
        c=colors[2],
        linestyle="--",
        lw=1.4,
        label="Reset, both courses",
        zorder=0,
    )
    left.set(
        xticks=x,
        yticks=[0, 1, 38 / 30, 58 / 30],
        yticklabels=["0", "1", "1.27", "1.93"],
        ylim=(0, 2.2),
        xlabel="Successive hidden change",
        ylabel="Seconds of preparation",
    )
    left.set_title("Identifying a new motor interface", loc="left", fontweight="bold")
    left.legend(loc="lower center", frameon=False, fontsize=9)
    trials = _read(OUT / "lateral-confirmation.json.gz")["blocks"]
    scores = [
        [
            sum(r["results"][name]["legal"] for b in trials if b["changed"] == flag for r in b["rows"])
            for flag in (False, True)
        ]
        for name in ("original", "blend")
    ]
    y = np.array([1, 0])
    for i, (label, c) in enumerate((("Original", colors[2]), ("After lateral practice", colors[0]))):
        bars = right.barh(y + (i - 0.5) * 0.26, scores[i], height=0.21, color=c, label=label)
        right.bar_label(bars, padding=4)
    right.set(
        yticks=y,
        yticklabels=["Standard court", "Changed surface\nand gravity"],
        xlim=(0, max(max(r) for r in scores) + 6),
        xlabel="Legal openings per 370 recorded shots",
        xticks=[0, 5, 10, 15, 20],
        ylim=(-0.8, 1.65),
    )
    right.set_title("Returning unfamiliar trajectories", loc="left", fontweight="bold")
    right.legend(loc="upper right", frameon=False, fontsize=9)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "learning-across-changes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
