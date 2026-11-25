"""Summarize matched motor-learning runs and how performance changes over successive course blocks."""

from pathlib import Path

import numpy as np

from . import tt52_practice_dose as dose
from . import tt53_responsive_report as report
from .tt4_transfer_trial import _read, _write, _hash


def compare(folder, name, count):
    path = folder / name
    card, v, cost = report.scores(path)
    parent = dose.ROOT / "live-from-48.json.gz"
    c, base, ordinary_cost = report.scores(parent)
    if len(v) != count or c[:count] != card:
        raise ValueError(f"Changed physical roster for {path}")
    src = _read(parent)
    cpu = sum(r["runtime_cpu_s"] for r in src["rows"][:count])
    opening = np.array([a["id"] for a in card])
    distinct = np.unique(opening)
    diff = v - base[:count]
    means = np.array([diff[opening == i].mean() for i in distinct])
    rng = np.random.default_rng(567429)
    ci = np.quantile(
        rng.choice(means, (10000, len(means))).mean(axis=1), [0.025, 0.975]
    )
    return dict(
        new=dict(**report.summarize(v), sum_cpu_s=cost),
        stock=dict(**report.summarize(base[:count]), sum_cpu_s=cpu),
        games_better=int(sum(diff > 0)),
        games_worse=int(sum(diff < 0)),
        ties=int(sum(diff == 0)),
        release_cluster_interval=ci.tolist(),
        files=[_hash(path), _hash(parent)],
    )


def practice():
    a = dose.ROOT / "live-from-12.json.gz"
    b = dose.ROOT / "live-from-48.json.gz"
    ca, x, _ = report.scores(a)
    cb, y, _ = report.scores(b)
    if ca != cb or len(x) != 96 or len(y) != 96:
        raise ValueError("Incomplete acquisition comparison")
    blocks = []
    for k in (0, 32, 64):
        blocks.append(
            dict(
                start_game=k + 1,
                after12=report.summarize(x[k : k + 32]),
                after48=report.summarize(y[k : k + 32]),
            )
        )
    _write(
        dose.ROOT / "dose-results.json",
        dict(
            blocks=blocks,
            whole=compare(dose.ROOT, a.name, 96),
            note="The group called new is the 12-practice checkpoint. Stock is the 48-practice checkpoint. Both continue learning for all 96 games.",
        ),
    )
    for folder, file, count in [
        ("tt54_earned_low", "context-games.json.gz", 32),
        ("tt55_anticipatory_joints", "live.json.gz", 24),
        ("tt56_long_motor_memory", "games.json.gz", 16),
        ("tt56_scored_motor_atlas", "games.json.gz", 16),
        ("tt54_stratified_memory", "continuous.json.gz", 64),
    ]:
        folder = Path("artifacts") / folder
        rows = _read(folder / file)["rows"]
        if len(rows) == count:
            _write(folder / "broad-comparison.json", compare(folder, file, count))


if __name__ == "__main__":
    practice()
