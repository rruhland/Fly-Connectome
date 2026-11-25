"""Retain rare, earned joint movements when the continual motor bank reaches capacity."""

import argparse
import hashlib
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write


def descriptor(row):
    state, goals, _ = row
    action = goals[[4, 12], :5].reshape(-1)
    return np.r_[state, 0.32 * action / np.tile([0.30, 0.32, 0.32, 0.34, 0.45], 2)]


class Repertoire(deque):
    def append(self, row):
        if len(self) < self.maxlen:
            super().append(row)
            return
        current = list(self)
        mutable = np.array(
            [i for i, r in enumerate(current) if r[2][0] != "carried"], dtype=int
        )
        if not len(mutable):
            return
        embedding = np.stack([descriptor(r) for r in [*current, row]])
        # A deterministic subset keeps the comparison cheap at a fixed bank size.
        seed = int.from_bytes(
            hashlib.sha256(np.asarray(row[0], dtype="<f8").tobytes()).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(seed)
        sample = rng.choice(mutable, min(20, len(mutable)), replace=False)
        distances = np.sum(
            (embedding[sample, None, :] - embedding[None, :, :]) ** 2, axis=2
        )
        distances[np.arange(len(sample)), sample] = np.inf
        most_redundant = sample[np.argmin(distances.min(axis=1))]
        del self[int(most_redundant)]
        super().append(row)


class Diverse(course.Carried):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.online = Repertoire(self.online, maxlen=self.online.maxlen)


def run(a):
    # Resume exactly where the common initial motor experience was still below
    # capacity. All subsequent games and opponent draws stay in their original order.
    if not a.resume:
        initial = Path("artifacts/tt42/learning-12.json.gz")
        d = _read(initial)
        if len(d["rows"]) != 12 or d["rows"][-1]["memory"]["after"]["rows"] >= 768:
            raise ValueError("Common prefix was already evicting movements")
        a.out.write_bytes(initial.read_bytes())
        target = a.out.with_suffix("").with_suffix(".npz")
        target.write_bytes(initial.with_suffix("").with_suffix(".npz").read_bytes())
        if _hash(target) != d["checkpoint_sha256"]:
            raise ValueError("Practice snapshot mismatch")
        a.resume = True
    with patch.object(course, "Carried", Diverse):
        course.run(a)
    d = _read(a.out)
    d["retention"] = (
        "Retain starting scored practice; remove redundant local sensory/motor alternatives when full."
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(mode="live", deck="practice", n=48, cap=240, carry=None)
    run(p.parse_args())
