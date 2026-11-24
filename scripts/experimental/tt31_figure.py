"""Compare the actual blade-center paths during the same alternating game."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tt4_transfer_trial import _read
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt31_replay import make_arena


def path(source, seed):
    data = _read(source)
    i, r = next(
        (i, r) for i, r in enumerate(data["rows"]) if r["opponent_seed"] == seed
    )
    with centered(data["ready"]):
        arena = make_arena(data, i)
        try:
            arena.launch(r["shot"])
            arena.partner.reset(seed=seed)
            track = []
            for u in r["actions"]:
                arena.exchange(u, np.zeros(5))
                track.append(
                    (
                        arena.data.time,
                        100 * arena.data.geom_xpos[arena._racket_ids[1], 0],
                    )
                )
            if digest_json(r["events"]) != digest_json(arena.events):
                raise ValueError("Position replay changed events")
        finally:
            arena.close()
    hits = np.array(
        [
            (l["near_contact"]["time"], 100 * l["near_contact"]["location"][0])
            for l in r["placements"]["delivered"]
            if l["near_landed"]
        ]
    )
    return np.asarray(track), hits


def plot(folder):
    plain, _h1 = path(folder / "ready-periodic-staged.json.gz", 315509)
    ready, h2 = path(folder / "recovery-periodic.json.gz", 315509)
    fig, ax = plt.subplots(figsize=(8.4, 4.1), layout="constrained")
    ax.axhline(0, color="#a6b0b7", lw=0.8)
    ax.plot(
        *ready.T, color="#087968", lw=2.5, label="Return toward readiness after impact"
    )
    ax.plot(*plain.T, color="#6f507b", ls="--", lw=1.9, label="Original follow-through")
    ax.scatter(
        *h2.T,
        s=60,
        color="#087968",
        edgecolor="white",
        zorder=4,
        label="Contact that landed legally",
    )
    ax.set_title(
        "The same incoming left ball became a legal return",
        loc="left",
        weight="bold",
        fontsize=15,
    )
    ax.set_ylabel("Blade center relative to shoulder (cm)")
    ax.set_xlabel("Time from opening release (s)")
    ax.set_ylim(-40, 45)
    ax.set_xlim(0, ready[-1, 0] + 0.06)
    ax.set_yticks(
        [-30, -15, 0, 15, 30], ["30 left", "15 left", "0", "15 right", "30 right"]
    )
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, ncols=1, frameon=False)
    ax.annotate(
        "Second arm return",
        (h2[1, 0], h2[1, 1]),
        xytext=(h2[1, 0] + 0.32, -32),
        arrowprops={"arrowstyle": "-", "color": "#35434b"},
        fontsize=9,
    )
    fig.savefig(folder / "recovery-path.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    plot(p.parse_args().folder)
