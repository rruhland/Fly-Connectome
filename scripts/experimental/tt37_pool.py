"""Combine distinct action-verified camera training sets without replaying the same teacher twice."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write


def combine(paths, out):
    combined = {}
    keys = set()
    sources = {}
    practice = []
    arrays = {}
    for path in paths:
        d = _read(path.with_suffix(".json"))
        if _hash(path) != d["bank_sha256"]:
            raise ValueError(f"Changed bank {path}")
        if combined and (
            d["ready"] != combined["ready"] or d["depth"] != combined["depth"]
        ):
            raise ValueError("Mismatch in physical body posture")
        for k, v in d["sources"].items():
            if k in sources and sources[k] != v:
                raise ValueError(f"Conflict in source {k}")
            sources[k] = v
        for case in d["practice"]:
            c = case["card"]
            key = (c["id"], c["seed"], c["reply_seed"])
            if key in keys:
                raise ValueError("Repeated training point")
            keys.add(key)
            practice.append(case)
        combined = d
        with np.load(path) as z:
            for k in ("states", "targets_now", "targets_lead", "marks"):
                arrays.setdefault(k, []).append(z[k])
    np.savez_compressed(
        out, **{k: np.concatenate(values) for k, values in arrays.items()}
    )
    _write(
        out.with_suffix(".json"),
        {
            "ready": combined["ready"],
            "depth": combined["depth"],
            "sources": sources,
            "practice": practice,
            "bank_sha256": _hash(out),
            "frames": sum(c["frames"] for c in practice),
            "legal": sum(c["mentor_returns"] for c in practice),
            "input_banks": {p.name: _hash(p) for p in paths},
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("banks", nargs="+", type=Path)
    a = p.parse_args()
    combine(a.banks, a.out)
