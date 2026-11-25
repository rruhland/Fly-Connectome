"""Validate the coach's preview accuracy and separately assess actual practice-acquired memory."""

import argparse
import numpy as np
from . import tt70_body_retention as retain
from . import tt73_partner_rehearsal as teacher
from .tt4_transfer_trial import _read, _write, _hash


def summarize(folder):
    src = folder / "rehearsal.json.gz"
    d = _read(src)
    if not d.get("complete"):
        raise ValueError("Incomplete practice pilot")
    rows = [x for game in d["practice_diagnostics"] for x in game]
    pairs = [(x["survey"][x["chosen"]], x["actual"]) for x in rows]
    both = np.asarray([(a, b) for a, b in pairs if a is not None and b is not None])
    diff = np.abs(both[:, 0] - both[:, 1])
    out = dict(
        exposures=d["received"],
        matches=len(both),
        max_abs=diff.max(0).tolist(),
        mean_abs=diff.mean(0).tolist(),
        preview_only=sum(a is not None and b is None for a, b in pairs),
        actual_only=sum(a is None and b is not None for a, b in pairs),
        previews=sum(len(x["survey"]) for x in rows),
        source=_hash(src),
        sampled_draws=len(rows),
        practice_cpu=sum(r["runtime_cpu_s"] for r in d["rows"]),
    )
    _write(folder / "accuracy.json", out)
    print(out, flush=True)
    return out


def report():
    from .tt53_responsive_report import compare

    folder = teacher.ROOT / "course"
    treatment = ("uniform", "rehearsal")
    practice = {}
    for name in treatment:
        d = _read(folder / f"{name}.json.gz")
        if not d["complete"] or d["received"] != sum(
            r["score"]["opponent"] for r in d["rows"]
        ):
            raise ValueError("Incomplete or inconsistent exposure count")
        practice[name] = dict(
            received=d["received"],
            games=len(d["rows"]),
            cells=d["cells"],
            arm_returns=sum(r["score"]["learner"] for r in d["rows"]),
            process_cpu_s=sum(r["runtime_cpu_s"] for r in d["rows"]),
        )
    assessment = {}
    for fixture in ("original", "lateral"):
        paths = {
            "responsive": folder / f"rehearsal-{fixture}.json.gz",
            "uniform": folder / f"uniform-{fixture}.json.gz",
            "before": retain.ROOT / f"{fixture}-before.json.gz",
        }
        result = compare(paths, 16)
        result["groups"]["rehearsal"] = result["groups"].pop("responsive")
        result["uniform_minus_before"] = compare(
            {"responsive": paths["uniform"], "before": paths["before"]}, 16
        )["comparisons"]["before"]
        assessment[fixture] = result
    _write(
        folder / "results.json",
        dict(
            practice=practice,
            assessments=assessment,
            fidelity=_read(folder / "accuracy.json"),
        ),
    )


def assess(a):
    folder = teacher.ROOT / "course"
    source = folder / (a.phase + ".json.gz")
    d = _read(source)
    if (
        not d["complete"]
        or d["received"] < 256
        or d["protocol"] != _hash(folder / "course.json")
    ):
        raise ValueError("Unfinished checkpoint")
    retain.run(
        a.body,
        "after",
        a.feeds,
        acquired=source,
        destination=folder / (a.phase + "-" + a.body + ".json.gz"),
    )


if __name__ == "__main__":
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "phase", choices=("pilot", "verify", "uniform", "rehearsal", "report")
    )
    p.add_argument("body", choices=("original", "lateral"), nargs="?")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    if a.phase in ("pilot", "verify"):
        summarize(teacher.ROOT / ("pilot" if a.phase == "pilot" else "course"))
    elif a.phase == "report":
        report()
    elif a.body:
        assess(a)
    else:
        p.error("Choose an assessment body")
