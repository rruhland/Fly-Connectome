"""Predict the arm's following legal return from a candidate and its presently observed flight."""

import argparse
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.optimize import minimize

from . import tt41_reply_control as paired
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from . import tt47_paired_consequences as contrast
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt49_sustain")
CHECKPOINT = ROOT / "fit.json"


def sources():
    primary = [Path("artifacts", f + ".json.gz") for f in contrast.SOURCES]
    recent = [
        fresh.ROOT / f"{a}.json.gz" for a in ("stock", "risk", "history", "combined")
    ]
    return sorted(set(paired.files() + primary + recent))


def dataset(paths, held):
    observations, hashes = {}, {}
    for path in paths:
        if not path.is_file():
            continue
        d = _read(path)
        if (
            not isinstance(d, dict)
            or not d.get("cards")
            or len(d["cards"]) != len(d.get("rows", ()))
        ):
            continue
        accepted = False
        for c, r in zip(d["cards"], d["rows"], strict=True):
            if (
                c["id"] in held
                or c.get("profile") != "reach"
                or not r.get("searches")
                or not r.get("placements")
            ):
                continue
            offers = r["placements"]["delivered"]
            for j, o in enumerate(offers):
                if not o.get("near_bounce") or not o.get("far_contact"):
                    continue
                if j + 1 >= len(offers) and not any(
                    e["kind"] == "point" for e in r["events"]
                ):
                    continue
                start = o["far_contact"]["step"]
                end = (o.get("near_contact") or r["events"][-1])["step"]
                decisions = [
                    n
                    for n in r["searches"]
                    if start < n["tick"] <= end
                    and n.get("target_errors")
                    and n.get("contact_delay_s") is not None
                ]
                if not decisions:
                    continue
                n = decisions[-1]
                if (
                    o.get("near_contact")
                    and abs(n["tick"] + n["contact_delay_s"] * 60 - end) > 5
                ):
                    continue
                # If there was no actual contact, a candidate superseded long
                # before the failure must not inherit the later move's loss.
                latest = max(
                    (k["tick"] for k in r["searches"] if start < k["tick"] <= end),
                    default=n["tick"],
                )
                if latest > n["tick"] + 5:
                    continue
                x = np.r_[
                    n["target_errors"], n["p"], n.get("robust", 0), n.get("score", 0.0)
                ]
                if len(x) != 12 or not np.isfinite(x).all():
                    continue
                future = j + 1 < len(offers) and bool(offers[j + 1].get("near_landed"))
                y = (
                    float(bool(o.get("near_landed"))),
                    float(o.get("near_landed") and future),
                )
                # Identical optical context, candidate and actual incoming have
                # one vote, while retaining different subsequent actual outcomes.
                key = (
                    c["id"],
                    *np.round(o["far_contact"]["incoming"], 5),
                    *np.round(x, 6),
                )
                if key not in observations:
                    observations[key] = [x, []]
                observations[key][1].append(y)
                accepted = True
        if accepted:
            hashes[str(path)] = _hash(path)
    x = np.stack([a for a, _ in observations.values()])
    y = np.stack([np.mean(b, axis=0) for _, b in observations.values()])
    g = np.array([k[0] for k in observations])
    return x, y, g, hashes


@lru_cache(maxsize=2)
def projection(width):
    return np.random.default_rng(49061).normal(size=(width, 64))


def make_basis(x, mu, sd):
    u = np.clip((x - mu) / sd, -4.0, 4.0)
    # Smooth nonlinear interactions between predicted flight and observed
    # approach matter when different received speeds favor different strokes.
    h = np.tanh(u @ projection(u.shape[1]) / np.sqrt(u.shape[1]))
    return np.c_[np.ones(len(x)), u, np.abs(u), h]


def train_part(x, y):
    mu = x.mean(axis=0)
    sd = np.maximum(0.025, x.std(axis=0))
    a = make_basis(x, mu, sd)
    penalty = np.r_[0.05, np.full(a.shape[1] - 1, 5.0)]

    def fit(y):
        def loss(w):
            z = a @ w
            return np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * (
                penalty * w
            ) @ w, a.T @ (expit(z) - y) + penalty * w

        opt = minimize(
            loss,
            np.zeros(a.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        if not opt.success:
            raise RuntimeError(opt.message)
        return opt.x

    return mu, sd, np.stack([fit(t) for t in y.T], axis=1)


def predict(parameters, x):
    mu, sd, w = parameters
    return expit(make_basis(np.atleast_2d(x), mu, sd) @ w)


def fit():
    held = {
        c["id"] for name in ("assessment", "practice") for c in _read(season.PLAN)[name]
    }
    x, y, g, files = dataset(sources(), held)
    prediction = np.empty_like(y)
    context = np.empty_like(y)
    mean = np.empty_like(y)
    for block in np.array_split(
        np.random.default_rng(49307).permutation(np.unique(g)), 6
    ):
        test = np.isin(g, block)
        prediction[test] = predict(train_part(x[~test], y[~test]), x[test])
        context[test] = predict(train_part(x[~test, 7:], y[~test]), x[test, 7:])
        mean[test] = y[~test].mean(axis=0)

    def stats(a):
        return dict(
            brier=np.mean((a - y) ** 2, axis=0).tolist(),
            tail_success=[
                [
                    float(y[ix, k].mean())
                    for ix in np.array_split(np.argsort(a[:, k]), 5)
                ]
                for k in range(2)
            ],
        )

    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        CHECKPOINT,
        dict(
            fit=[v.tolist() for v in train_part(x, y)],
            input_files=files,
            source_ids=np.unique(g).tolist(),
            examples=len(x),
            actual=y.mean(axis=0).tolist(),
            validation=dict(
                mean=stats(mean), context=stats(context), candidate=stats(prediction)
            ),
            coverage=x[:, :7].tolist(),
        ),
    )


class Player(season.Player):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        d = _read(CHECKPOINT)
        self.value = tuple(map(np.asarray, d["fit"]))
        self.support = cKDTree(np.asarray(d["coverage"]) / fresh.shot.SCALE)

    def search(self, tick, x, q, ref, base):
        self.optical_position = x[:3].copy()
        super().search(tick, x, q, ref, base)

    def robust(self, goals, rank, nominal):
        self.in_rank = True
        try:
            votes, quality = self.uncertainty(goals, nominal)
        finally:
            self.in_rank = False
        predicted = np.array([self.metrics[np.asarray(g).tobytes()] for g in goals])
        risk, supported = self.coverage.risk(predicted)
        ordinary, _ = fresh.shot.choice(votes, quality, risk, supported)
        x = np.c_[
            predicted, np.tile(self.optical_position, (len(goals), 1)), votes, quality
        ]
        actual = predict(self.value, x)
        support = self.support.query(predicted / fresh.shot.SCALE, k=8)[0][:, -1] < 1.6
        reward = 0.25 * actual[:, 0] + 0.75 * actual[:, 1]
        improve = reward - reward[ordinary]
        good = (
            support
            & (votes >= votes[ordinary])
            & (votes >= 1)
            & (quality <= quality[ordinary] + 0.75)
            & (actual[:, 0] >= actual[ordinary, 0] - 0.012)
            & (improve > 0.022)
        )
        pick = ordinary
        if supported[ordinary] and support[ordinary] and good.any():
            pick = int(
                np.argmax(
                    np.where(
                        good,
                        improve - 0.021 * np.maximum(0.0, quality - quality[ordinary]),
                        -np.inf,
                    )
                )
            )
            self.diag["actual_following_return_choices"] += 1
        return int(rank[pick]), int(votes[pick]), float(quality[pick])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit", "live"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="practice", mode="live", carry=season.CARRY, cap=240, resume=False
    )
    a = p.parse_args()
    if a.action == "fit":
        fit()
    else:
        doc = _read(CHECKPOINT)
        if set(doc["source_ids"]) & {c["id"] for c in _read(season.PLAN)[a.deck]}:
            raise ValueError("Outcome learner saw a held release")
        with patch.object(season, "Player", Player):
            season.run(a)
        result = _read(a.out)
        result["continuation_model"] = _hash(CHECKPOINT)
        _write(a.out, result)
