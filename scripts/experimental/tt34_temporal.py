"""Learn the visual similarity metric of a direct joint policy from unrewarded consecutive observations."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.linalg import eigh

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt34_latency import OUT, PUPIL

BASE = joint.features


def visual_metric(bank, mode):
    src = np.load(bank)
    past = BASE(src["flight"], "raw")
    later = BASE(src["next"][:, 2, :], "raw")
    if mode == "shuffled":
        later = later[np.random.default_rng(340572).permutation(len(later))]
    cov = np.cov(past.T) + 0.05 * np.eye(past.shape[1])
    vals, dirs = eigh(np.cov((later - past).T) + 0.005 * np.eye(past.shape[1]), cov)
    # Three slow directions express repeatable visual state over two camera intervals.
    # Keep the full observed state as well, so transient velocity can still affect control.
    v = dirs[:, :3]
    metric = 0.60 * np.eye(6) + 0.65 * (v @ v.T)
    return np.linalg.cholesky(metric), vals


@contextmanager
def use(matrix):
    def observed(x, mode, q=None, home=None):
        vals = BASE(x, mode, q, home)
        if mode == "raw-proprioception":
            return np.concatenate((vals[..., :6] @ matrix, vals[..., 6:]), axis=-1)
        return vals

    with patch.object(joint, "features", observed):
        yield


def fit(raw, mode, output):
    bank = OUT / "lag-bank.npz"
    mat, vals = visual_metric(bank, mode)
    with use(mat):
        joint.fit(raw, output, _read(PUPIL)["options"].copy())
    result = _read(output)
    result["metric"] = {
        "source_sha256": _hash(bank),
        "mode": mode,
        "matrix": mat.tolist(),
        "eigenvalues": vals.tolist(),
    }
    _write(output, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("ordered", "shuffled", "play"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--raw", type=Path)
    p.add_argument("--feeds", type=Path)
    p.add_argument("--course", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.mode == "play":
        with use(np.asarray(_read(a.checkpoint)["metric"]["matrix"])):
            joint.evaluate(
                a.feeds,
                a.course,
                a.checkpoint,
                a.output,
                cap=16,
                limit=None,
                trace=False,
            )
    else:
        fit(a.raw, a.mode, a.checkpoint)
