"""Evaluate frozen shot-error memories with fresh physics, vision and motor memory per point."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_action_memory import run
from .tt30_openings import nearest_openings


def evaluate(feeds, movements, output, mode, donor, openings, offset):
    uses_donor = mode in ("carried", "reversed", "combined", "combined-reversed")
    if uses_donor != (donor is not None):
        raise ValueError("Provide a donor only for a carried, reversed or combined comparison")
    results = []
    with TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        bank = donor
        if mode in ("reversed", "combined-reversed"):
            restore = _read(donor)["optical_outcome_checkpoint"]
            for example in restore["samples"]:
                example["error_m"] = (-np.asarray(example["error_m"])).tolist()
            bank = scratch / "sign-reversed.json.gz"
            _write(bank, {"optical_outcome_checkpoint": restore})
        for i, shot in enumerate(openings):
            temporary = scratch / "point.json.gz"
            if mode in ("motor", "combined", "combined-reversed", "repair"):
                from .tt29_motor_correction import run as edit
                edit(
                    feeds, temporary, [shot], movements,
                    mode="repair" if mode == "repair" else "edit", seed_offset=offset + i,
                    return_target=24, optical_checkpoint=bank,
                )
            else:
                run(
                    feeds, temporary, [shot], "left+right", "lateral", checkpoint=movements,
                    rated=True, motor_envelope=True, robust_contact=True, rally_objective=True,
                    recent=True, blend=True, fast_partner=True, cached_stereo=True, net_rescue=True,
                    camera_period=2, spread="varied", seed_offset=offset + i, return_target=24,
                    optical_outcomes="record" if mode == "control" else "learn",
                    freeze_optical_outcomes=mode != "online", optical_checkpoint=bank,
                )
            report = _read(temporary)
            row = report["rows"][0]
            results.append(row)
            _write(output, dict(mode=mode, cap=24, seed_offset=offset, feed_sha256=_hash(feeds),
                                movement_sha256=_hash(movements), donor_sha256=_hash(donor) if donor else None,
                                ids=openings, rows=results))
            print(mode, shot, row["score"]["returns"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("movements", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("mode", choices=("control", "online", "carried", "reversed", "motor", "combined", "combined-reversed", "repair"))
    p.add_argument("--donor", type=Path)
    p.add_argument("--offset", type=int, default=91537)
    args = p.parse_args()
    evaluate(args.feeds, args.movements, args.output, args.mode, args.donor, nearest_openings(args.feeds), args.offset)
