"""Build a paired report for the fresh course and the pilot of scoreless motor replay."""

from .tt4_transfer_trial import _hash, _read, _write
from .tt34_latency import OUT
from .tt34_results import paired

DIRECTORY = OUT.parent / "tt35"


def summarize():
    plan = _read(DIRECTORY / "confirmation-plan.json.gz")
    cases = {
        "pilot_reference": (
            OUT.parent / "tt33/joint-fresh-raw.json.gz",
            [
                DIRECTORY / f"hindsight-{mode}-reference.json.gz"
                for mode in ("aligned", "as-observed", "reverse", "static", "familiar")
            ],
        ),
        "pilot_novel": (
            OUT / "body-novel-stock.json.gz",
            [
                DIRECTORY / f"hindsight-{mode}-novel.json.gz"
                for mode in ("aligned", "as-observed", "reverse", "static", "familiar")
            ],
        ),
        "confirmation": (
            DIRECTORY / "confirm-nominal.json.gz",
            [
                DIRECTORY / f"confirm-{mode}.json.gz"
                for mode in ("aligned", "as-observed", "static", "familiar")
            ],
        ),
        "varied_replies": (
            DIRECTORY / "replies-nominal.json.gz",
            [DIRECTORY / f"replies-{mode}.json.gz" for mode in ("aligned", "static")],
        ),
        "confirmation_with_delay": (
            DIRECTORY / "confirm-lag-stale.json.gz",
            [DIRECTORY / "confirm-lag-learned.json.gz"],
        ),
    }
    exclusions = set(plan["exclude_ids"])
    ids = {card["id"] for card in plan["cards"]}
    if ids & exclusions or len(ids) != 24:
        raise ValueError(
            "Confirmation course contains a duplicate or previously used release"
        )
    report = {}
    for label, (reference_path, paths) in cases.items():
        ref = _read(reference_path)
        if label == "varied_replies" and any(
            _read(path)["reply_seeds"] != ref["reply_seeds"] for path in paths
        ):
            raise ValueError("Different opponent reply seeds")
        if label.startswith("confirmation") and ref["cards"] != plan["cards"]:
            raise ValueError("Confirmation cards differ from the saved roster")
        report[label] = {
            "baseline_sha256": _hash(reference_path),
            "comparisons": {
                path.name: {"sha256": _hash(path), **paired(ref, _read(path))}
                for path in paths
            },
        }
    _write(DIRECTORY / "confirmation-results.json", report)


if __name__ == "__main__":
    summarize()
