"""Plot held flight accuracy and real versus simulated outgoing rally speeds."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .tt25_recorded_flight import HELD, OUTPUT, compact_court, propagate, quantiles, records


def cdf(axis, values, color, label):
    ordered = np.sort(values)
    axis.step(
        ordered,
        np.arange(1, len(ordered) + 1) / len(ordered),
        where="post",
        color=color,
        label=label,
        linewidth=2.3,
    )
    median = np.median(ordered)
    axis.scatter([median], [0.5], color=color, s=25, zorder=4)


def additional_reference(path):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != "300f4bf19556dd8eb69663f46de7f642008048b746321cc98754150a4e0e1237":
        raise ValueError("Expected the pinned DeepMind rallies.json")
    rows = json.loads(raw)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate shot IDs")
    distributions = {}
    for name, prefix in [("speed", "vel"), ("spin", "w_vel")]:
        vectors = [[r[f"{prefix}_{axis}"] for axis in "xyz"] for r in rows]
        distributions[name] = np.linalg.norm(vectors, axis=1)
    receipt = dict(
        source="google-deepmind/competitive_robot_table_tennis",
        commit="54cf18e41a8f134d98ad38710001ed50aabb8ebc",
        sha256=digest,
        license="CC BY 4.0",
        **{k: quantiles(v) for k, v in distributions.items()},
    )
    (OUTPUT / "recorded-secondary-reference.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return distributions


def plot(source, deepmind):
    rows, real = records(source)
    held = [r for r in rows if r.player in HELD]
    sim, _ = compact_court()
    additional = additional_reference(deepmind)
    fit = json.loads((OUTPUT / "recorded-flight.json").read_text())["held_scores"]
    fig, (landing, speed) = plt.subplots(1, 2, figsize=(10.2, 4.3), layout="constrained")
    orange, blue = "#af4a15", "#155895"
    for name, color, label in (
        ("current", orange, "Previous physics"),
        ("match_fit", blue, "Match-calibrated physics"),
    ):
        predicted, _ = propagate(held, fit[name]["coefficients"], predict_landing=True)
        error = 100 * np.linalg.norm(predicted - np.stack([row.end[:2] for row in held]), axis=1)
        cdf(landing, error, color, label)
    landing.set(
        xlabel="Distance from recorded bounce (cm)",
        xlim=(0, 115),
        ylabel="Share of shots at or below value",
    )
    landing.set_title("Predicting 417 real bounces", loc="left", fontweight="bold")
    cdf(speed, sim["speed"], orange, "Simulated practice (37 shots)")
    cdf(speed, real["speed"], blue, "Sony matches (541 shots)")
    cdf(speed, additional["speed"], "#47733c", "DeepMind rallies (13,088 shots)")
    speed.set(xlabel="Outgoing ball speed (m/s)", xlim=(0, max(real["speed"]) * 1.01))
    speed.set_title("Pace after the opening serve", loc="left", fontweight="bold")
    for a in (landing, speed):
        a.set_ylim(0, 1.02)
        a.yaxis.set_major_formatter(PercentFormatter(1))
        a.grid(axis="y", alpha=0.2)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(loc="lower right", frameon=True, fontsize=9)
    fig.savefig(OUTPUT / "recorded-comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("deepmind", type=Path)
    args = parser.parse_args()
    plot(args.csv, args.deepmind)
