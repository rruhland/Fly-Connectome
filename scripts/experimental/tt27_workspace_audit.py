"""Score the learned regulation receiver specifically on incoming balls near sampled arm poses."""

import argparse
from pathlib import Path
import json

from .tt4_transfer_trial import _read, _write
from .tt25_recorded_reach import workspace, trajectories
from .tt26_grip_course import PRIOR, load_match
from .tt27_flight_feedback import OUT


def run(path):
    shots, _ = load_match(path)
    normal = [b for b in _read(OUT / "lateral-confirmation.json.gz")["blocks"] if not b["changed"]]
    records = [r for b in normal for r in b["rows"]]
    tree, radius, num = workspace()
    coeff = json.loads((PRIOR / "recorded-flight.json").read_text())["held_scores"]["match_fit"][
        "coefficients"
    ]
    geometric = trajectories([shots[r["id"]] for r in records], coeff, tree, radius)
    for r, g in zip(records, geometric, strict=True):
        assert r["id"] == g["id"]
        g["legal"] = {k: v["legal"] for k, v in r["results"].items()}
        g["contact"] = {k: v["contact"] for k, v in r["results"].items()}
    _write(
        OUT / "workspace-audit.json.gz",
        dict(
            source="lateral-confirmation.json.gz",
            poses=num,
            clear=tree.n,
            blade_radius=radius,
            rows=geometric,
        ),
    )
    for flag in (False, True):
        sub = [r for r in geometric if r["near_blade"] == flag]
        print(
            flag, len(sub), {k: sum(r["legal"][k] for r in sub) for k in ("original", "blend")}, flush=True
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
