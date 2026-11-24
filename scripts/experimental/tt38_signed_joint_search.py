"""Use signed virtual departures to adjust local time-varying joint commands on difficult shots."""

import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write


class RewardedDepartures:
    def __init__(self, path):
        with np.load(path) as z:
            s = z["states"]
            lag = z["marks"][:, 1]
            sel = (lag > 0.061) & (lag < 0.15) & (s[:, 4] > 1)
            dt = lag[sel, None]
            v = s[sel, 3:6]
            a = (
                np.array([0.0, 0.0, -9.81])
                - 0.13 * np.linalg.norm(v, axis=1, keepdims=True) * v
            )
            self.vel = v - a * dt
            self.pos = s[sel, :3] - dt * v + 0.5 * dt**2 * a
        self.tree = cKDTree(self.pos / [0.22, 0.13, 0.11])

    def targets(self, position):
        _, ix = self.tree.query(position / [0.22, 0.13, 0.11], k=min(24, len(self.vel)))
        return np.quantile(self.vel[np.atleast_1d(ix)], [0.30, 0.5, 0.78], axis=0)


def pattern(length, params):
    t = np.arange(1, length + 1) / length
    ramp = np.minimum(t / 0.42, 1.0)
    change = np.clip((t - 0.45) / 0.49, -1.0, 1.0)
    return ramp[None, :, None] * (
        params[:, None, :5] + change[None, :, None] * params[:, None, 5:]
    )


class SignedMPC(mpc.MPC):
    def __init__(self, *args, memory, **kwargs):
        super().__init__(*args, **kwargs)
        self.departures = memory
        self.capture = False

    def trial(self, goals, *, long=False, traces=False):
        result = super().trial(goals, long=long, traces=traces)
        if self.capture and not long:
            for g, f in zip(goals, result[-1], strict=True):
                if f[1] > 0.5:
                    self.hits.append((g, f.copy()))
        return result

    def search(self, tick, x, q, reference, base):
        self.hits = []
        self.capture = True
        try:
            super().search(tick, x, q, reference, base)
        finally:
            self.capture = False
        if (
            not self.notes
            or self.notes[-1]["tick"] != tick
            or self.notes[-1]["chosen"]
            or not self.hits
        ):
            return
        start = process_time()
        # A departing (including badly backward) ball makes the candidate informative.
        # At most two distinct action sequences get a local inverse fit.
        viable = []
        for g, f in self.hits:
            wanted = self.departures.targets(f[5:8])
            err = np.linalg.norm((wanted[1] - f[2:5]) / [1.9, 2.4, 1.6]) + 4 * f[8]
            viable.append((err, g, wanted))
        selected = []
        for _, g, wanted in sorted(viable, key=lambda v: v[0]):
            if any(np.max(np.abs(g - old)) < 0.05 for old in selected):
                continue
            selected.append(g)
            answer = self.refine_direct(g, wanted, tick)
            if answer is not None:
                choice, consistency, score, stats = answer
                self.best = choice
                self.tick = tick
                self.lease = mpc.LEASE
                self.notes[-1].update(
                    chosen="signed_outcome",
                    robust=consistency,
                    score=score,
                    local_response=stats,
                )
                self.diag["accepted"] += 1
                self.diag["signed_acceptances"] += 1
                self.diag["robust_accepted"] += bool(consistency)
                break
            if len(selected) >= 2:
                break
        self.diag["search_cpu_s"] += process_time() - start

    def refine_direct(self, g, wanted, tick):
        n = len(g)
        rng = np.random.default_rng([388105, tick])
        scales = np.r_[
            np.array([0.09, 0.13, 0.15, 0.16, 0.20]),
            np.array([0.09, 0.11, 0.12, 0.14, 0.17]),
        ]
        dirs = np.vstack(
            [np.zeros(10), np.eye(10), -np.eye(10), rng.normal(0, 0.64, (16, 10))]
        )

        def materialize(params):
            actions = np.broadcast_to(g, (len(params), *g.shape)).copy()
            actions[:, :, :5] += pattern(n, params * scales)
            return actions

        designs = materialize(dirs)
        _, _, _, _, f = self.trial(designs)
        keep = f[:, 1] > 0.5
        if keep.sum() < 4:
            return None
        z = dirs[keep]
        v = f[keep, 2:5]
        mean = z.mean(0)
        vy = v.mean(0)
        centered = z - mean
        jac = np.linalg.solve(
            centered.T @ centered + 0.22 * np.eye(10), centered.T @ (v - vy)
        )
        vel = vy - mean @ jac
        proposals = []
        for target in wanted:
            delta = jac @ np.linalg.solve(jac.T @ jac + 0.18 * np.eye(3), target - vel)
            for gain in (0.5, 0.85, 1.2, 1.65):
                proposals.append(np.clip(gain * delta, -2.6, 2.6))
        distance = np.min(
            np.linalg.norm(
                (v[:, None, :] - wanted[None, :, :]) / [1.2, 1.6, 1.0], axis=2
            ),
            axis=1,
        )
        candidates = np.vstack((z[np.argsort(distance)[:7]], proposals))
        actions = materialize(candidates)
        score, legal, _, _, _ = self.trial(actions, long=True)
        valid = np.flatnonzero(legal)
        if not len(valid):
            return None
        best = valid[np.argsort(score[valid])[:5]]
        which, consistent, value = self.robust(actions[best], best, score[best])
        if not consistent:
            return None
        return (
            actions[which],
            consistent,
            value,
            {
                "contact_examples": int(keep.sum()),
                "positive": len(valid),
                "jacobian_norm": float(np.linalg.norm(jac)),
            },
        )


def run(deck, out, departures, bank, limit):
    memories = RewardedDepartures(departures)
    optic = mpc.ROOT / "tt37/rebound.npz"
    with patch.object(
        mpc, "MPC", lambda *args, **kw: SignedMPC(*args, memory=memories, **kw)
    ):
        mpc.run(
            Path("/tmp/tt25-rallies.json"),
            bank,
            mpc.PLAN,
            deck,
            "live",
            out,
            optic=optic,
            cap=200,
            limit=limit,
            variant="elastic",
        )
    res = _read(out)
    res.update(
        departure_prior_sha256=_hash(departures),
        signed_solver_sha256=_hash(Path(__file__)),
    )
    _write(out, res)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deck", choices=("development", "held_openings", "held_replies"))
    p.add_argument("out", type=Path)
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--bank", type=Path, default=mpc.BANK)
    p.add_argument(
        "--departures", type=Path, default=mpc.ROOT / "tt37/self-physical.npz"
    )
    a = p.parse_args()
    run(a.deck, a.out, a.departures, a.bank, a.limit)
