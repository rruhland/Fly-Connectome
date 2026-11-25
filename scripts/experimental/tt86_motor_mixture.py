"""A continuously updated mixture of conditional direct-joint trajectory models."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import softmax

from . import tt56_scored_motor_atlas as archive
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt86_motor_mixture")
MODEL = ROOT / "policy.npz"
dose, season, core = archive.dose, archive.season, archive.mpc
course = season.course
K = 64
LATENT = 12
D = 18


def cluster(z):
    rng = np.random.default_rng(863312)
    centers = z[rng.choice(len(z), K, replace=False)].copy()
    sq = np.sum(z * z, axis=1)
    for _ in range(14):
        distance = sq[:, None] + np.sum(centers**2, axis=1) - 2 * z @ centers.T
        labels = np.argmin(distance, axis=1)
        for j in range(K):
            group = z[labels == j]
            centers[j] = group.mean(0) if len(group) > 8 else z[rng.integers(len(z))]
    return labels


class Mixture:
    def __init__(self, fields):
        for name, value in fields.items():
            setattr(self, name, value.copy())

    @classmethod
    def fit(cls, state, moves, won):
        u = moves[:, :, :5].reshape(len(moves), -1).astype(float)
        xmu = state.mean(0)
        xscale = np.maximum(state.std(0), 0.18)
        x = np.clip((state - xmu) / xscale, -6, 6)
        ymu = u[won].mean(0)
        yscale = np.maximum(u[won].std(0), 0.05)
        a = (u - ymu) / yscale
        eig, basis = np.linalg.eigh(np.cov(a[won], rowvar=False))
        proj = basis[:, -LATENT:][:, ::-1].T
        zscale = np.sqrt(np.maximum(eig[-LATENT:][::-1], 0.01))
        z = (a @ proj.T) / zscale
        labels = cluster(np.c_[x, 0.95 * z])
        means = []
        precision = []
        offset = []
        reg = []
        likelihood = []
        counts = []
        for j in np.unique(labels):
            ix = labels == j
            wins = ix & won
            if sum(wins) < 12:
                continue
            mx = x[ix].mean(0)
            dx = x[wins] - mx
            features = np.c_[np.ones(len(dx)), dx]
            ridge = np.diag(np.r_[0.01, np.full(D, 12.0)])
            b = np.linalg.solve(features.T @ features + ridge, features.T @ z[wins])
            var = np.cov(x[ix], rowvar=False)
            cov = 0.70 * var + 0.30 * np.diag(np.diag(var)) + 0.21 * np.eye(D)
            means.append(mx)
            precision.append(np.linalg.inv(cov))
            offset.append(b[0])
            reg.append(b[1:])
            likelihood.append(
                np.log(sum(ix) / len(ix)) - 0.5 * np.linalg.slogdet(cov)[1]
            )
            counts.append([sum(ix & ~won), sum(wins)])
        return cls(
            dict(
                xmu=xmu,
                xscale=xscale,
                ymu=ymu,
                yscale=yscale,
                proj=proj,
                zscale=zscale,
                means=np.array(means),
                precision=np.array(precision),
                offset=np.array(offset),
                reg=np.array(reg),
                likelihood=np.array(likelihood),
                counts=np.array(counts, dtype=float),
            )
        )

    def infer(self, state):
        dx = np.clip((state - self.xmu) / self.xscale, -6, 6) - self.means
        distances = np.einsum("ki,kij,kj->k", dx, self.precision, dx)
        prob = (self.counts[:, 1] + 2) / (self.counts.sum(1) + 4)
        rank = self.likelihood - 0.5 * distances + 2 * np.log(prob)
        top = np.argsort(rank)[-4:][::-1]
        latent = self.offset[top] + np.einsum(
            "ki,kij->kj", np.clip(dx[top], -2.5, 2.5), self.reg[top]
        )
        moves = self.ymu + (latent * self.zscale) @ self.proj * self.yscale
        return (
            top,
            dx[top],
            latent,
            softmax(rank[top]),
            moves.reshape(-1, core.HORIZON, 5),
            distances[top],
        )

    def learn(self, state, move, won):
        top, dx, expected, p, _, _ = self.infer(state)
        z = (
            ((move[:, :5].reshape(-1) - self.ymu) / self.yscale) @ self.proj.T
        ) / self.zscale
        responsibility = softmax(
            np.log(p + 1.0e-12) - 0.8 * np.sum((expected - z) ** 2, axis=1)
        )
        self.counts[top, int(won)] += responsibility
        if won:
            error = np.clip(z - expected, -2, 2)
            phi = np.clip(dx, -2.5, 2.5)
            # Normalized local delta update on the observed successful motion.
            rate = 0.14 * responsibility / (1 + np.sum(phi**2, axis=1))
            self.offset[top] += rate[:, None] * error
            self.reg[top] += rate[:, None, None] * phi[:, :, None] * error[:, None, :]


def fit():
    ROOT.mkdir(exist_ok=True)
    if _hash(archive.MODEL) != _read(archive.ROOT / "fit.json")["model"]:
        raise ValueError("Changed original public movement collection")
    with np.load(archive.MODEL) as f:
        x, u, y, ids = [f[k] for k in ("keys", "moves", "reward", "release")]
    groups = np.unique(ids)
    hold = np.random.default_rng(8610341).permutation(groups)[:24]
    test = np.isin(ids, hold)
    model = Mixture.fit(x[~test], u[~test], y[~test])
    errors = []
    for i in np.flatnonzero(test & y):
        _, _, _, _, pred, _ = model.infer(x[i])
        diff = np.sqrt(np.mean((pred[:, :5] - u[i, :5, :5]) ** 2, axis=(1, 2)))
        errors.append([diff[0], np.min(diff), np.sqrt(np.mean(u[i, :5, :5] ** 2))])
    model = Mixture.fit(x, u, y)
    np.savez_compressed(MODEL, **vars(model))
    _write(
        ROOT / "fit.json",
        dict(
            model=_hash(MODEL),
            source=_hash(archive.MODEL),
            train_windows=len(x),
            scored=int(sum(y)),
            releases=groups.tolist(),
            hold=hold.tolist(),
            validation_n=len(errors),
            joint_rmse_first_mode_best_of_four_and_hold_rad=np.mean(
                errors, axis=0
            ).tolist(),
            modes=len(model.means),
            latent_rank=LATENT,
        ),
    )


class Player(season.Player):
    def __init__(self, *args, policy, direct, **kw):
        super().__init__(*args, **kw)
        self.policy = policy
        self.direct = direct
        self.pending = None
        original = self.priors.priors

        def options(x, q, ref, limit):
            ordinary = original(x, q, ref, limit)
            if self.pending is None:
                return ordinary
            for seq in self.pending:
                ordinary.append((seq, True))
            return ordinary

        self.priors.priors = options
        keep = self.priors.keep

        def paid(tape, tag):
            self.update(tape, True)
            return keep(tape, tag)

        self.priors.keep = paid

    def begin(self):
        super().begin()
        self.previous = None
        self.observations = deque(maxlen=16)
        self.free = None

    def update(self, tape, won):
        lookup = {t: i for i, (t, *_) in enumerate(tape)}
        for t, s in self.observations:
            if t not in lookup:
                continue
            i = lookup[t]
            targets = np.asarray([r[-1] for r in tape[i : i + core.HORIZON]])
            if len(targets) >= 3:
                self.policy.learn(
                    s, core.tail(targets, core.HORIZON) - tape[i][-2], won
                )
                self.diag["mixture_updates"] += 1
        self.observations.clear()

    def missed(self, tape):
        self.update(tape, False)

    def search(self, tick, x, q, reference, base):
        state = archive.key(tick, x[:3], self.engine.bounced, reference, self.previous)
        self.previous = (tick, x[:3].copy(), reference.copy())
        modes, _, _, weights, moves, dist = self.policy.infer(state)
        self.observations.append((tick, state))
        options = []
        for seq in moves:
            path = np.tile(reference, (core.HORIZON, 1))
            path[:, :5] += np.clip(seq, -0.90, 0.90)
            options.append(path)
        self.pending = options
        try:
            super().search(tick, x, q, reference, base)
        finally:
            self.pending = None
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        n = self.notes[-1]
        n["mixture"] = [int(modes[0]), float(weights[0]), float(dist[0])]
        self.free = None
        if (
            self.direct
            and not n["legal_trials"]
            and weights[0] > 0.52
            and dist[0] < 22
            and not n.get("prepare")
        ):
            self.free = tick, options[0]

    def step(self, tick, eye, x, q, reference, base):
        ordinary = super().step(tick, eye, x, q, reference, base)
        if self.free is not None and x is not None and x[4] < -0.9:
            start, path = self.free
            if 0 <= tick - start < 3 and (
                self.best is None or not 0 <= tick - self.tick < self.lease
            ):
                self.diag["direct_mixture_ticks"] += 1
                return ordinary + 0.65 * np.clip(
                    path[tick - start] - ordinary, -0.32, 0.32
                )
        return ordinary


def run(a):
    receipt = _read(ROOT / "fit.json")
    if receipt["model"] != _hash(MODEL) or set(receipt["releases"]) & {
        c["id"] for c in _read(dose.PLAN)["practice"]
    }:
        raise ValueError("Training overlap or changed source")
    with np.load(MODEL) as fields:
        policy = Mixture(fields)
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with (
        enabled(),
        patch.object(course, "PLAN", dose.PLAN),
        patch.object(
            core,
            "MPC",
            lambda *args, **kw: Player(
                *args, coverage=rank, policy=policy, direct=a.phase == "direct", **kw
            ),
        ),
    ):
        course.run(a)
    path = ROOT / (a.phase + "-policy.npz")
    np.savez_compressed(path, **vars(policy))
    d = _read(a.out)
    d["learned_mixture"] = _hash(path)
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "propose", "direct"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live", deck="practice", carry=season.CARRY, resume=False, cap=240
    )
    a = p.parse_args()
    a.out = ROOT / (a.phase + ".json.gz")
    fit() if a.phase == "fit" else run(a)
