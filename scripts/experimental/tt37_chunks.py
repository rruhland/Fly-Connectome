"""Fit overlapping short action predictions without backpropagation or a stroke library."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read
from .tt37_joint_deep import SCALES, run
from .tt37_opponent import arena

HORIZON = 8


def build(archive, dest):
    meta = _read(archive.with_suffix(".json"))
    with np.load(archive) as data:
        marks = data["marks"]
        labels = data["targets_now"]
    a = arena()
    axis = a._arm_ids[1][2]
    step = a._joint_increments[axis] * (a.control_dt / a.model.opt.timestep)
    low, high = a._joint_low[axis], a._joint_high[axis]
    a.close()
    offset = 0
    source_cache = {}
    chunks = []
    prev = []
    for item in meta["practice"]:
        path = archive.parent / item["source"]
        if item["source"] not in source_cache:
            if _hash(path) != meta["sources"][item["source"]]:
                raise ValueError(f"Changed teacher source {path}")
            source_cache[item["source"]] = _read(path)
        tape = source_cache[item["source"]]["rows"][item["source_row"]]["actions"]
        ctrl = np.array(meta["ready"])
        refs = [ctrl.copy()]
        for command in tape:
            ctrl = np.clip(ctrl + step * np.array(command), low, high)
            refs.append(ctrl)
        refs = np.stack(refs)
        indices = marks[offset : offset + item["frames"], 0].astype(int)
        # An episode may terminate partway through the final control update.
        if not np.allclose(
            refs[indices + 1], labels[offset : offset + len(indices)], atol=1e-5
        ):
            raise ValueError(
                f"Issued and reconstructed reference disagree in {item['source']}"
            )
        chunks.append(
            refs[np.minimum(indices[:, None] + 1 + np.arange(HORIZON), len(tape))]
        )
        prev.append(refs[indices])
        offset += len(indices)
    if offset != len(labels):
        raise ValueError("Source frames missing")
    np.savez_compressed(
        dest,
        chunks=np.vstack(chunks).astype("f"),
        previous=np.vstack(prev).astype("f"),
        archive_sha256=_hash(archive),
    )


class Ensemble:
    def __init__(self, archive, chunkfile, *, style):
        with np.load(archive) as reader, np.load(chunkfile) as target:
            if str(target["archive_sha256"]) != _hash(archive):
                raise ValueError("Chunk labels and optical archive differ")
            x = reader["states"]
            self.labels = target["chunks"].astype(float)
            ref = target["previous"]
        self.scale = SCALES["local"]
        self.x = np.c_[x / self.scale, (ref[:, :5] - x[:, 6:11]) / 0.32]
        self.groups = {}
        for sign in (-1, 1):
            i = np.flatnonzero(np.sign(x[:, 4]) == sign)
            self.groups[sign] = (i, cKDTree(self.x[i]))
        self.style = style
        self.begin_point()

    def begin_point(self):
        self.recent = deque(maxlen=HORIZON)
        self.last_sign = None
        self.tick = 0

    def clock_tick(self, tick):
        self.tick = tick

    def get(self, optical, q, *, reference):
        sign = -1 if optical[4] < 0 else 1
        if self.last_sign != sign:
            self.recent.clear()
        self.last_sign = sign
        center = np.r_[
            np.r_[optical, q[:5]] / self.scale, (reference[:5] - q[:5]) / 0.32
        ]
        idx, tree = self.groups[sign]
        dist, local = tree.query(center, 24)
        ix = idx[local]
        weight = np.exp(-0.5 * (dist / max(0.15, dist[8])) ** 2)
        conf = float(np.clip((2.8 - dist[3]) / 1.0, 0.0, 1.0))
        if conf > 0.0:
            weight /= weight.sum()
            dev = self.x[ix] - center
            mean = weight @ dev
            delta = dev - mean
            labels = self.labels[ix].reshape(len(ix), -1)
            anchor = weight @ labels
            jac = np.linalg.solve(
                delta.T @ (delta * weight[:, None]) + 0.060 * np.eye(len(center)),
                delta.T @ (weight[:, None] * (labels - anchor)),
            )
            schedule = (anchor - np.clip(mean @ jac, -0.19, 0.19)).reshape(HORIZON, -1)
            self.recent.append((self.tick, schedule, conf))
        series = []
        cred = []
        for start, seq, c in self.recent:
            age = self.tick - start
            if age >= HORIZON or (self.style == "single" and age > 0):
                continue
            w = c * np.exp(-age / 4.0)
            series.append(seq[age])
            cred.append(w)
        if not cred:
            return None, 0.0
        return np.average(series, axis=0, weights=cred), max(cred)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "ensemble", "single"])
    p.add_argument("archive", type=Path)
    p.add_argument("chunks", type=Path)
    p.add_argument(
        "--plan", type=Path, default=Path("artifacts/tt37/opponent-plan.json")
    )
    p.add_argument("--deck", default="held_openings")
    p.add_argument("--out", type=Path)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.mode == "prepare":
        build(a.archive, a.chunks)
    else:
        run(
            Path("/tmp/tt25-rallies.json"),
            a.archive,
            a.plan,
            a.deck,
            "local",
            a.out,
            cap=200,
            limit=a.limit,
            field_hook=lambda _: Ensemble(a.archive, a.chunks, style=a.mode),
            details={
                "chunk_style": a.mode,
                "horizon_ticks": HORIZON,
                "chunks_sha256": _hash(a.chunks),
            },
        )
