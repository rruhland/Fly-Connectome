"""Adapt an individual scored movement to currently observed flight progress and body geometry."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried


def align(source, commands, x, q, reference, body, *, spatial):
    stored = source * mpc.OPEN_NORM
    previous = stored[:3]
    velocity = stored[3:6]
    speed2 = velocity @ velocity
    if speed2 < 0.36:
        return commands
    dt = np.clip((x[:3] - previous) @ velocity / speed2, -0.064, 0.064)
    # Nearest phase along the sampled flight and relative speed of progression
    # are generic properties of a tracked moving object.
    ratio = np.clip(x[3:6] @ velocity / speed2, 0.7, 1.32)
    times = np.arange(mpc.HORIZON)
    query = dt / mpc.DT + times * ratio
    out = np.stack([np.interp(query, times, r) for r in commands.T], axis=-1)
    if spatial:
        err = x[:3] - (previous + dt * velocity)
        _, _, linear, _ = body.pose(stored[6:11], jacobian=True)
        offset = linear.T @ np.linalg.solve(
            linear @ linear.T + 0.035**2 * np.eye(3), err
        )
        out[:, :5] += np.clip(offset, -0.25, 0.25)
    # Keep the initial reference change small where the new approach would
    # otherwise introduce a discontinuity unrelated to observed ball progress.
    mismatch = reference[:5] - (stored[6:11] + stored[11:])
    out[:, :5] += 0.50 * np.exp(-times[:, None] / 5.0) * np.clip(mismatch, -0.28, 0.28)
    return out


class Adaptive(Carried):
    def __init__(self, bank, engine, *, practice, held, spatial):
        super().__init__(bank, engine, practice=practice, held=held)
        self.body = mpc.BladeIK(engine.twin)
        self.spatial = spatial
        self.proposals = 0

    def priors(self, x, q, ref, limit):
        key = mpc.query(x, q, ref)
        d, ix = self.tree.query(key, min(150, len(self.rows)))
        choices = [(ds, *self.rows[i], False) for ds, i in zip(d, ix, strict=True)]
        choices.extend(
            (max(0.0, np.linalg.norm(s - key) - 0.10), s, cmd, tag, True)
            for s, cmd, tag in self.online
        )
        picked = []
        used = set()
        for dist, state, cmd, tag, actual in sorted(choices, key=lambda r: r[0]):
            if tag in used:
                continue
            if len(picked) >= 2 and dist > 3.1:
                break
            used.add(tag)
            picked.append((state, cmd, actual))
            if len(picked) >= limit:
                break
        found = [(c.copy(), real) for _, c, real in picked]
        for state, cmd, real in picked[:3]:
            found.append(
                (align(state, cmd, x, q, ref, self.body, spatial=self.spatial), real)
            )
            self.proposals += 1
        return found


def run(a):
    held = {c["id"] for c in _read(a.plan)[a.deck][: a.limit]}
    instances = []

    def bank(file, engine):
        m = Adaptive(
            file,
            engine,
            practice=[mpc.ROOT / "tt37/camera-time-school.npz"],
            held=held,
            spatial=a.spatial,
        )
        instances.append(m)
        return m

    with patch.object(mpc, "Fragments", bank):
        mpc.run(
            a.feeds,
            mpc.BANK,
            a.plan,
            a.deck,
            "live",
            a.out,
            optic=mpc.ROOT / "tt37/rebound.npz",
            limit=a.limit,
            variant="elastic",
            cap=240,
        )
    obj = instances[0]
    ck = a.out.with_suffix("").with_suffix(".npz")
    np.savez_compressed(
        ck,
        states=[r[0] for r in obj.online],
        goals=[r[1] for r in obj.online],
        tags=[repr(r[2]) for r in obj.online],
    )
    d = _read(a.out)
    d.update(
        temporal=True,
        spatial=a.spatial,
        proposals=obj.proposals,
        checkpoint_sha256=_hash(ck),
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--spatial", action="store_true")
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
