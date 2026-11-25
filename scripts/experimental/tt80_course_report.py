"""Measure actual continuing gains and compare frozen versus live common random streams."""

import argparse
from pathlib import Path

from . import tt80_prospective_course as game
from .tt4_transfer_trial import _read, _write, _hash
from .tt66_memory_report import compare
from .tt68_clamp_transfer import describe
from .tt37_opponent import draw


def matched_subset(left, right):
    counts = {}
    changes = {}
    for u, v in zip(left, right, strict=True):
        for a, b in zip(u["placements"]["delivered"], v["placements"]["delivered"]):
            if (
                not a.get("near_bounce")
                or a["near_bounce"] != b.get("near_bounce")
                or a["far_contact"] != b.get("far_contact")
            ):
                continue
            key = f"{int(bool(a['near_landed']))}_to_{int(bool(b['near_landed']))}"
            counts[key] = counts.get(key, 0) + 1
            t = a["far_contact"]["step"] - 1
            end = min(
                z["near_contact"]["step"] if z["near_contact"] else len(doc["actions"])
                for z, doc in ((a, u), (b, v))
            )
            if u["actions"][t:end] != v["actions"][t:end]:
                changes[key] = changes.get(key, 0) + 1
    return dict(outcomes=counts, changed_actions=changes)


def report(new, stock, n):
    summary = compare(dict(control=stock, treatment=new), n)
    rows = _read(new)["rows"][:n]
    original = _read(stock)["rows"][:n]
    for c, r in zip(_read(new)["cards"], rows):
        for j, h in enumerate(r["placements"]["requested"]):
            if h["draw"] != "random" or h["target"] != list(
                draw(c, j, "random", None)[0]
            ):
                raise ValueError("Different opponent law")
    summary.update(
        blocks=[
            dict(
                first=i + 1,
                last=min(i + 16, n),
                treatment=describe(rows[i : i + 16]),
                control=describe(original[i : i + 16]),
            )
            for i in range(0, n, 16)
        ],
        matched=matched_subset(original, rows),
    )
    return summary


def season():
    source = game.ROOT / "live.json.gz"
    d = _read(source)
    if len(d["rows"]) != 128 or d["cards"] != _read(game.PLAN)["practice"]:
        raise ValueError("Unfinished or altered 128-game stream")
    dct = dict(
        course=_hash(game.PLAN),
        source=_hash(source),
        total=describe(d["rows"]),
        blocks=[],
    )
    for i in range(0, 128, 32):
        dct["blocks"].append(dict(start=i + 1, **describe(d["rows"][i : i + 32])))
    _write(game.ROOT / "live-results.json", dct)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op", choices=("season", "compare"))
    p.add_argument("--new", type=Path, default=game.ROOT / "live.json.gz")
    p.add_argument("--control", type=Path, default=game.ROOT / "frozen.json.gz")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--out", type=Path, default=game.ROOT / "within-and-across.json")
    a = p.parse_args()
    if a.op == "season":
        season()
    else:
        _write(a.out, report(a.new, a.control, a.n))
