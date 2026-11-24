"""Plot two native ball trajectories under one timing intervention on the same incoming ball."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .tt4_transfer_trial import _read
from .tt29_conditions import cooperative_targets
from .tt29_partner_fast import FastPartnerArena
from .tt30_retry import retry


def plot(source, output):
    archive = _read(source)
    row = next(r for r in archive["rows"] if r["shot"]["id"] == 7345)
    beginning = max(e["step"] for e in row["events"] if e["kind"] == "contact" and e["side"] == -1)
    expected = [e for e in row["events"] if e["step"] > beginning]
    station = FastPartnerArena(strike_ready=archive["pose"], fast_search=True, clearance_guard=True,
                               motor_envelope=True, velocity_guard=True, rate_scale=.8, guard_table=True,
                               max_steps=9000, targets=cooperative_targets(archive["pose"], archive["spread"]),
                               net_rescue=archive["net_rescue"])
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    try:
        states = []
        for shift, legend, color in ((0, "Original movement · net", "#b34a35"),
                                     (-1, "Movement 16.7 ms earlier · legal landing", "#216b62")):
            trace = []
            score = retry(station, row, beginning, shift, expected, trace)
            states.append(score["state"])
            assert score["landed"] == (shift == -1)
            hit = next(e for e in score["events"] if e["kind"] == "contact" and e["side"] == 1)
            positions = np.asarray(trace)
            after = positions[positions[:, 0] >= hit["time"]]
            ax.plot(after[:, 2], after[:, 3], label=legend, color=color, linewidth=2.4)
            if shift == 0:
                last = positions[positions[:, 0] < hit["time"]]
                ax.plot(last[:, 2], last[:, 3], color="#93979d", lw=1.4, ls="--", label="Incoming ball")
            ax.scatter(hit["ball"][1], hit["ball"][2], color=color, s=18)
        assert len(set(states)) == 1
        ax.plot([-1.37, 1.37], [.76, .76], lw=3.2, color="#253831")
        ax.plot([0, 0], [.76, .9125], lw=2.5, color="#253831")
        ax.set(xlabel="Distance along table (m) · arm on left", ylabel="Ball height (m)",
               title="Same incoming shot, same recorded joint commands, different timing")
        ax.set_ylim(bottom=.70)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        fig.savefig(output, dpi=160)
    finally:
        plt.close(fig)
        station.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    plot(args.source, args.output)
