"""Transfer demonstrated raw joint feedback across small spatial displacements using missed-shot body observations."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_vision import Rig
from .tt31_curriculum_opponent import replay_arena
from .tt33_joint import fit
from .tt34_latency import OUT, PUPIL


def triangulate(images, rig):
    rays = []
    origins = []
    for i, name in enumerate(("left", "right")):
        cam = rig.fixed[name]
        uv = (images[:, 2 * i : 2 * i + 2] - cam.center) / cam.focal
        ray = np.column_stack((uv, np.ones(len(uv)))) @ cam.rotation
        rays.append(ray / np.linalg.norm(ray, axis=1)[:, None])
        origins.append(cam.eye)
    a, b = rays
    delta = origins[1] - origins[0]
    cross = (a * b).sum(axis=1)
    ta = (a @ delta - cross * (b @ delta)) / (1.0 - cross**2)
    tb = cross * ta - b @ delta
    p = origins[0] + a * ta[:, None]
    r = origins[1] + b * tb[:, None]
    return 0.5 * (p + r), np.linalg.norm(p - r, axis=1)


class LearnedBody:
    def __init__(self, source):
        arena = replay_arena([])
        try:
            p, gap = triangulate(source["xy"], Rig.from_arena(arena))
        finally:
            arena.close()
        valid = gap < 0.035
        self.q = source["q"][valid]
        self.p = p[valid]
        self.scale = np.maximum(0.20, self.q.std(axis=0))
        self.tree = cKDTree(self.q / self.scale)

    def jacobian(self, q):
        dist, ix = self.tree.query(q / self.scale, 48)
        weight = np.exp(-0.5 * (dist / max(0.12, dist[16])) ** 2)
        weight /= max(weight.sum(), 1e-12)
        x = self.q[ix] - weight @ self.q[ix]
        y = self.p[ix] - weight @ self.p[ix]
        jac = np.linalg.solve(
            x.T @ (weight[:, None] * x) + 0.0007 * np.eye(5),
            x.T @ (weight[:, None] * y),
        ).T
        return jac


def augment(raw, bank, out, mode):
    with np.load(raw) as archive:
        d = dict(archive)
    with np.load(bank) as archive:
        body = LearnedBody(archive)
    blocks = []
    # The same two lateral displacements are attempted on every demonstration.
    for displacement in (-0.075, 0.075):
        block = {k: v.copy() for k, v in d.items()}
        for i, q in enumerate(d["joints"]):
            j = body.jacobian(q[:5])
            delta = j.T @ np.linalg.solve(
                j @ j.T + 0.0015 * np.eye(3), [displacement, 0, 0]
            )
            delta *= min(1.0, 0.34 / max(np.max(abs(delta)), 1e-9))
            dp = j @ delta
            block["optical"][i, :3] += dp
            if mode == "aligned":
                block["joints"][i, :5] += delta
                block["targets"][i, :5] += delta
            elif mode == "reverse":
                block["joints"][i, :5] -= delta
                block["targets"][i, :5] -= delta
        blocks.append(block)
    np.savez_compressed(out, **{k: np.concatenate([b[k] for b in blocks]) for k in d})
    receipt = _read(raw.with_suffix(".json"))
    receipt["sources"][bank.name] = _hash(bank)
    receipt["receipts"] = []
    _write(out.with_suffix(".json"), receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("mode", choices=("aligned", "visual-only", "reverse"))
    p.add_argument("output", type=Path)
    a = p.parse_args()
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as folder:
        added = Path(folder) / "translated.npz"
        augment(a.raw, OUT / "lag-bank.npz", added, a.mode)
        options = _read(PUPIL)["options"] | {"additional": [added]}
        fit(a.raw, a.output, options)
