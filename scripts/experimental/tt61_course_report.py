"""Repeat all cross-course checks on finalized physical checkpoints."""

from pathlib import Path

from . import tt53_responsive_report as reports
from . import tt56_course_comparisons as earlier
from .tt4_transfer_trial import _write

ROOT = Path("artifacts")


def comparison(folder, subject, others, n):
    paths = {
        "responsive": ROOT / folder / subject[1],
        **{k: ROOT / v for k, v in others.items()},
    }
    d = reports.compare(paths, n)
    d["groups"][subject[0]] = d["groups"].pop("responsive")
    d["treatment"] = subject[0]
    _write(ROOT / folder / "assessment.json", d)


def run():
    comparison(
        "tt57_practice_retention",
        ("after_96", "after.json.gz"),
        {"before": "tt57_practice_retention/before.json.gz"},
        16,
    )
    comparison(
        "tt56_staircase_practice",
        ("staircase", "staircase-frozen.json.gz"),
        {
            "ordinary": "tt53_responsive_school/ordinary-frozen.json.gz",
            "original": "tt46_season/before.json.gz",
            "earlier_adaptive": "tt53_responsive_school/responsive-frozen.json.gz",
        },
        24,
    )
    comparison(
        "tt57_atlas_transfer",
        ("acquired", "acquired.json.gz"),
        {"ablated": "tt57_atlas_transfer/ablated.json.gz"},
        32,
    )
    for folder, file, n in [
        ("tt57_rally_strategy", "games.json.gz", 24),
        ("tt61_motor_lead", "live.json.gz", 32),
        ("tt62_relative_servo", "live.json.gz", 32),
    ]:
        path = ROOT / folder / file
        if path.exists() and len(reports.scores(path)[0]) == n:
            _write(
                path.parent / "broad-comparison.json",
                earlier.compare(path.parent, file, n),
            )


if __name__ == "__main__":
    run()
