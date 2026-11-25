"""Measure forward transfer from a single uninterrupted, physically certified learning point."""

import argparse
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from . import tt46_continuous_season as season
from . import tt48_practice_transfer as stream
from . import tt49_cumulative_memory as history
from .tt4_transfer_trial import _hash, _read, _write

course = season.course
ROOT = Path("artifacts/tt51_point_transfer")
SOURCE = stream.ROOT / "live-plain-12.json.gz"
PLAN = ROOT / "paired.json"


def snapshot():
    ROOT.mkdir(exist_ok=True, parents=True)
    doc = _read(SOURCE)
    base = history.load_rows(stream.SOURCE)
    end = history.load_rows(SOURCE)
    n = [r["planner"].get("credited_fragments", 0) for r in doc["rows"]]
    overlap = len(end) - sum(n)
    if overlap <= 0 or not all(
        history.identical(a, b)
        for a, b in zip(base[-overlap:], end[:overlap], strict=True)
    ):
        raise ValueError("Lost practice overlap")
    allrows = base + end[overlap:]
    parent = _read(stream.SOURCE)
    with np.load(stream.SOURCE.with_suffix("").with_suffix(".npz")) as z:
        serial = int(z["serial"])
    for games in (9, 10):
        i = len(base) + sum(n[:games])
        bank = course.mpc.Fragments.__new__(course.mpc.Fragments)
        bank.online = deque(allrows[i - len(end) : i], maxlen=len(end))
        if bank.digest() != doc["rows"][games - 1]["memory"]["after"]:
            raise ValueError("Could not reconstruct the original boundary")
        out = ROOT / f"after-{games:02}.json.gz"
        model = SimpleNamespace(priors=bank, serial=serial + games)
        fingerprint = course.save(model, out.with_suffix("").with_suffix(".npz"))
        _write(
            out,
            dict(
                doc,
                cards=doc["cards"][:games],
                rows=doc["rows"][:games],
                exact_source=_hash(SOURCE),
                parent_serial=serial,
                checkpoint_sha256=fingerprint,
            ),
        )
    pool = {c["id"]: c for c in _read(stream.PLAN)["practice"]}
    excluded = {c["id"] for c in doc["cards"][:10]}
    excluded.update(c["id"] for c in parent["cards"][: len(parent["rows"])])
    held = sorted(set(pool) - excluded)
    rng = np.random.default_rng(512047891)
    games = []
    for index in rng.permutation(held):
        games.append(
            dict(
                pool[index],
                seed=int(rng.integers(2**30)),
                reply_seed=int(rng.integers(2**30)),
            )
        )
    _write(
        PLAN,
        dict(
            assessment=games,
            practice=games,
            seed=512047891,
            source=_hash(SOURCE),
            rule="Exclude all 10 observed releases in both conditions. Same initial model, one additional 63-return uninterrupted experience. Independent new placement and opening seeds on all remaining releases. Evaluate the whole course.",
        ),
    )


def evaluate(a):
    source = ROOT / f"after-{a.games:02}.json.gz"
    d = _read(source)
    a.carry = source
    a.deck = "assessment" if a.mode == "frozen" else "practice"
    a.n = len(_read(PLAN)[a.deck])
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))

    def restore(model, path, doc):
        if doc["exact_source"] != _hash(SOURCE):
            raise ValueError("Changed point-boundary receipt")
        from .tt41_early_optics_carry import resume

        resume(model, path, {**doc, "rows": [None] * d["parent_serial"] + doc["rows"]})

    with (
        patch.object(course, "PLAN", PLAN),
        patch.object(course, "resume", restore),
        patch.object(
            course.mpc, "MPC", lambda *x, **kw: season.Player(*x, coverage=rank, **kw)
        ),
    ):
        course.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("extract", "live", "frozen"))
    p.add_argument("games", nargs="?", type=int, choices=(9, 10))
    p.add_argument("out", nargs="?", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(resume=False, cap=240)
    args = p.parse_args()
    snapshot() if args.mode == "extract" else evaluate(args)
