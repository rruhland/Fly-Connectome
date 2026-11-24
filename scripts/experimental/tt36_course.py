"""Prepare fresh regulation first deliveries and identical openings with changed opponent replies."""

from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt34_body import ARTIFACTS, plan

OUT = ARTIFACTS / "tt36"


def prepare(feeds):
    confirm = _read(ARTIFACTS / "tt35/confirmation-plan.json.gz")
    excluded = set(confirm["exclude_ids"]) | {c["id"] for c in confirm["cards"]}
    plan(feeds, OUT / "course-new.json.gz", count=32, seed=361204, exclude_ids=excluded)
    # Select before running new learners. These familiar openings let a frozen
    # direct pupil return the ball, so the opponent can actually offer later shots.
    previous = [
        _read(ARTIFACTS / p)
        for p in (
            "tt35/confirmation-plan.json.gz",
            "tt34/body-novel-plan.json.gz",
            "tt32/fresh-plan.json.gz",
        )
    ]
    selected = (5530, 14832, 3025, 8239)
    by_id = {c["id"]: c for doc in previous for c in doc["cards"]}
    cards = [by_id[i] for i in selected for _ in range(4)]
    seeds = list(range(361737, 361737 + len(cards)))
    _write(
        OUT / "course-replies.json.gz",
        {
            "cards": cards,
            "reply_seeds": seeds,
            "selection": "Four previously studied openings, four new opponent continuations each",
            "seed_controls": "Opponent placement RNG changes after the first far stroke",
        },
    )


def retention():
    source = ARTIFACTS / "tt35/replies-nominal.json.gz"
    old = _read(source)
    _write(OUT / "course-retained.json.gz", {
        "cards": old["cards"],
        "reply_seeds": old["reply_seeds"],
        "source_sha256": _hash(source),
        "selection": "All eight earlier varied replies on release 5530 in the original order",
    })


def confirmation(feeds):
    earlier = _read(OUT / "course-new.json.gz")
    excluded = set(earlier["exclude_ids"]) | {c["id"] for c in earlier["cards"]}
    for p in ("course-practice", "course-replies"):
        excluded.update(c["id"] for c in _read(OUT / f"{p}.json.gz")["cards"])
    plan(
        feeds,
        OUT / "course-confirm.json.gz",
        count=32,
        seed=369220,
        exclude_ids=excluded,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--retained", action="store_true")
    args = parser.parse_args()
    if args.retained:
        retention()
    else:
        (confirmation if args.confirm else prepare)(args.feeds)
