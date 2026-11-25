"""Restore an exact mid-season checkpoint with a proven ancestry and test the carried player on new releases."""

import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt42_learning_course as course
from . import tt46_context_search as trained
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write


def ancestral_restore(model, path, doc, parent):
    if model.serial != 0 or not doc["rows"]:
        raise ValueError(
            "Expected an unplayed model and a completed practice checkpoint"
        )
    previous = _read(parent)
    if (
        doc["carry_sha256"] != _hash(parent)
        or previous["checkpoint_sha256"]
        != _hash(parent.with_suffix("").with_suffix(".npz"))
        or previous["rows"][-1]["memory"]["after"] != doc["rows"][0]["memory"]["before"]
        or doc["plan_sha256"] != _hash(season.PLAN)
        or doc["cards"][: len(doc["rows"])]
        != _read(season.PLAN)["practice"][: len(doc["rows"])]
    ):
        raise ValueError("Season checkpoint provenance or parent memory differs")
    for a, b in zip(doc["rows"], doc["rows"][1:]):
        if a["memory"]["after"] != b["memory"]["before"]:
            raise ValueError("Discontinuous season practice")
    # The baseline checkpoint contains precisely these completed parent games.
    # The ordinary restorer also verifies serial, the actual numeric digest
    # of the reconstructed memory, and the NPZ checksum.
    season_original(model, path, {**doc, "rows": previous["rows"] + doc["rows"]})


season_original = course.resume


def run(a):
    with patch.object(
        course,
        "resume",
        lambda model, path, doc: ancestral_restore(model, path, doc, a.parent),
    ):
        if a.player == "standard":
            season.run(a)
        else:
            a.policy = a.carry
            trained.run(a)
    receipt = _read(a.out)
    receipt.update(ancestor_document=_hash(a.parent), restored_practice=_hash(a.carry))
    _write(a.out, receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("player", choices=("standard", "explored"))
    p.add_argument("carry", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--parent", type=Path, default=season.CARRY)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(mode="frozen", deck="assessment", cap=240, resume=False)
    run(p.parse_args())
