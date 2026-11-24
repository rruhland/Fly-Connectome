"""Audit and summarize the paired fresh-opening comparison and its signed-error control."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_optical_outcomes import ShotResiduals


def summarize(path, donor):
    data = _read(path)
    rows = data["rows"]
    if [r["shot"]["id"] for r in rows] != data["ids"]:
        raise ValueError(f"Incomplete or misaligned opening sequence: {path}")
    errors = []
    for row in rows:
        checked = audit(row["events"])
        if any(checked[k] != row["score"][k] for k in ("returns", "learner", "opponent")):
            raise ValueError(f"Saved events disagree with the scored point in {path}")
        for attempt in (row.get("optical_outcomes") or {}).get("attempts", []):
            if attempt.get("measured") is None or "features" not in attempt:
                continue
            bias, _ = donor.predict(np.asarray(attempt["features"])[None, :])
            observed_error = np.asarray(attempt["measured"]) - attempt["predicted"]
            errors.append([observed_error, observed_error - bias[0], observed_error + bias[0]])
    returns = [r["score"]["returns"] for r in rows]
    cpu = sum(r["process_cpu_s"] for r in rows)
    simulated = sum(len(r["actions"]) for r in rows) / 60  # SO-101 control ticks
    return dict(source_sha256=_hash(path), feed_sha256=data["feed_sha256"],
                movement_sha256=data["movement_sha256"], openings=data["ids"], seeds=[r["opponent_seed"] for r in rows],
                starting_examples=[r["initial_examples"] for r in rows], returns=returns,
                arm=sum(r["score"]["learner"] for r in rows), total=sum(returns),
                process_cpu_s=cpu, simulated_s=simulated, cpu_per_return_s=cpu / max(1, sum(returns)),
                placement_samples=len(errors),
                mean_absolute_error_m=None if not errors else np.abs(np.asarray(errors)).mean(axis=0).tolist())


def main(directory):
    donor_path = directory / "wide-shot-errors.json.gz"
    donor = ShotResiduals(_read(donor_path)["optical_outcome_checkpoint"])
    out = {}
    for name in ("fresh-control", "fresh-online", "fresh-carried", "fresh-reversed",
                 "fresh-motor", "fresh-combined", "fresh-combined-reversed", "repair", "confirm-control", "confirm-repair", "confirm-combined"):
        path = directory / (name + ".json.gz")
        if name in {"fresh-carried", "fresh-reversed", "fresh-combined", "fresh-combined-reversed", "confirm-combined"}:
            if _read(path)["donor_sha256"] != _hash(donor_path):
                raise ValueError(f"Optical donor mismatch: {name}")
        result = summarize(path, donor)
        if out:
            comparison = out["confirm-control" if name.startswith("confirm-") and name != "confirm-control" else "fresh-control"]
            if any(result[key] != comparison[key] for key in ("openings", "starting_examples", "feed_sha256", "movement_sha256")):
                raise ValueError(f"Opening or starting movement library mismatch: {name}")
            if name != "confirm-control" and result["seeds"] != comparison["seeds"]:
                raise ValueError(f"Opponent seed mismatch: {name}")
            if name != "confirm-control":
                delta = np.subtract(result["returns"], comparison["returns"])
                result["better_tied_worse"] = [int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())]
        out[name] = result
    _write(directory / "transfer-summary.json", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
