"""Compare composed joint commands on exactly the same physical incoming offer."""

import argparse
from pathlib import Path

import numpy as np

from . import tt43_temporal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json


def report(reference, source, out):
    left, right = _read(reference), _read(source)
    controls = left["rows"][: len(right["rows"])]
    if left["cards"][: len(controls)] != right["cards"] or len(controls) != len(
        right["rows"]
    ):
        raise ValueError("Incomplete or unpaired comparison")
    results = []
    for c, a, b in zip(right["cards"], controls, right["rows"], strict=True):
        before, after = a["score"]["learner"], b["score"]["learner"]
        if a["memory"]["before"] != b["memory"]["before"]:
            raise ValueError("Motor priors differed")
        first = next(
            (k for k, (p, q) in enumerate(zip(a["actions"], b["actions"])) if p != q),
            None,
        )
        r = dict(id=c["id"], reference=before, composed=after, first=first)
        if first is None:
            if a["actions"] != b["actions"] or digest_json(a["events"]) != digest_json(
                b["events"]
            ):
                raise ValueError("Identical commands produced different play")
        else:
            old = [e for e in a["events"] if e["step"] <= first]
            new = [e for e in b["events"] if e["step"] <= first]
            if digest_json(old) != digest_json(new):
                raise ValueError("Mismatch before intervention")
            for n, (x, y) in enumerate(
                zip(a["placements"]["delivered"], b["placements"]["delivered"])
            ):
                if (
                    not x["far_contact"]
                    or not y["far_contact"]
                    or x["far_contact"]["step"] > first
                ):
                    continue
                end = x.get("near_contact")
                if end and end["step"] <= first:
                    continue
                if digest_json(x["far_contact"]) != digest_json(y["far_contact"]):
                    continue
                r.update(
                    changed_offer=n + 1,
                    stock_landed=x["near_landed"],
                    composed_landed=y["near_landed"],
                    same_bounce=digest_json(x["near_bounce"])
                    == digest_json(y["near_bounce"]),
                )
        results.append(r)
    _write(
        out,
        dict(
            reference=_hash(reference),
            source=_hash(source),
            points=results,
            whole_game_cpu=[
                sum(r["runtime_cpu_s"] for r in d) for d in (controls, right["rows"])
            ],
            simulated_seconds=[
                sum(r["events"][-1]["time"] for r in d)
                for d in (controls, right["rows"])
            ],
            median=[
                float(np.median([r["score"]["learner"] for r in d]))
                for d in (controls, right["rows"])
            ],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--reference", type=Path, default=servo.ROOT / "physical.json.gz")
    a = p.parse_args()
    report(a.reference, a.source, a.out)
