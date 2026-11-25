"""Describe native offer and body differences between the factual terminal cuts."""

from math import dist
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json


ROOT = Path("artifacts/tt52_terminal_bank_crossplay")
TAPE = ROOT / "actions.json.gz"
OUTPUT = ROOT / "hidden-cut.json"


def report():
    tape = _read(TAPE)
    if not tape["complete"]:
        raise ValueError("Expected the completed closed-loop trial archive")
    rows = []
    for after, before in zip(tape["rows"][::2], tape["rows"][1::2], strict=True):
        cases = [r["case"] for r in (before, after)]
        if [c["origin"] for c in cases] != ["before", "after"] or cases[0]["card"] != cases[1]["card"]:
            raise ValueError("Cannot compare hidden cuts across unmatched assignments")
        a, b = [r["counterfactual"]["witness"] for r in (before, after)]
        row = dict(id=cases[0]["card"]["id"], factual_own=[r["factual_own"] for r in (before, after)])
        if not (a and b):
            rows.append(dict(row, same_incoming=None, paired_native_cuts=False))
            continue
        one, two = [c["original_offer"] for c in cases]
        x, y = a["body"], b["body"]
        same = all(digest_json(one[k]) == digest_json(two[k]) for k in ("far_contact", "near_bounce"))
        lr = [one["receive_line"], two["receive_line"]]
        rows.append(dict(row, same_incoming=same, paired_native_cuts=True,
                         legs=[o["leg"] for o in (one, two)], tick=[a["tick"], b["tick"]],
                         blade_m=dist(x["blade_xyz"], y["blade_xyz"]),
                         joints_rad=dist(x["q"][:5], y["q"][:5]),
                         reference_rad=dist(x["reference"], y["reference"]),
                         native_receive_line=lr,
                         receive_height_delta_m=abs(lr[0]["xyz"][2] - lr[1]["xyz"][2]) if all(lr) else None,
                         receive_velocity_delta_mps=dist(lr[0]["velocity"], lr[1]["velocity"]) if all(lr) else None))
    eligible = [r for r in rows if r["paired_native_cuts"]]
    maxima = {k: max(r[k] for r in eligible if r[k] is not None) for k in
              ("blade_m", "joints_rad", "reference_rad", "receive_height_delta_m", "receive_velocity_delta_mps")}
    _write(OUTPUT, dict(source_sha256=_hash(TAPE), rows=rows, paired_native_cuts=len(eligible), max_deltas=maxima,
                        exact_original_incoming=sum(r["same_incoming"] for r in eligible),
                        boundary="Evaluator-only comparison of the two factual points at their own terminal cuts. "
                                 "Do not treat differences in the subsequent received ball as a bank intervention."))


if __name__ == "__main__":
    report()
