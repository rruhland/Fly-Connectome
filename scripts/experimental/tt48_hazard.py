"""Identify likely lost deliveries while there is still time to change direct joint action."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt45_composed_feedback as composition
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt48_hazard")
FIT = ROOT / "fit.json"
FILES = [
    course.ROOT / "learning-48.json.gz",
    *[
        fresh.ROOT / f"{name}.json.gz"
        for name in ("combined", "history", "stock", "risk")
    ],
]


def features(notes):
    n = notes[-1]
    previous = (
        notes[-2] if len(notes) > 1 and 0 < n["tick"] - notes[-2]["tick"] <= 9 else None
    )
    p = np.asarray(n["p"])
    delta = (
        (p - previous["p"]) * 60.0 / (n["tick"] - previous["tick"])
        if previous
        else np.zeros(3)
    )
    return np.r_[
        p,
        abs(p[0]),
        delta,
        abs(delta[0]),
        n["bounced"],
        n["contact_trials"] / 28,
        n["best_separation_m"],
        min(n["best_separation_m"], 0.12),
        bool(n.get("prepare")),
        float(previous is not None),
        previous.get("robust", 0) if previous else 0,
        previous["legal_trials"] if previous else 0,
    ]


def eligible(n):
    return n.get("robust", 0) == 0 and -1.13 < n["p"][1] < -0.49


def train():
    reserve = {
        c["id"] for deck in ("practice", "assessment") for c in _read(season.PLAN)[deck]
    }
    observations = {}
    for file in FILES:
        d = _read(file)
        for c, r in zip(d["cards"], d["rows"], strict=True):
            if c["id"] in reserve:
                raise ValueError("Risk training overlaps the live evaluation")
            for delivery in r["placements"]["delivered"]:
                if not delivery.get("near_bounce"):
                    continue
                start = delivery["far_contact"]["step"]
                end = (delivery.get("near_contact") or r["events"][-1])["step"]
                prior = []
                for n in r["searches"]:
                    if n["tick"] < start or n["tick"] >= end:
                        continue
                    prior.append(n)
                    if eligible(n):
                        x = features(prior)
                        key = (c["id"], n["tick"], *np.round(x, 7))
                        # Exactly shared early decision contexts can have inconsistent
                        # later returns across actors. Average those continuations.
                        if key not in observations:
                            observations[key] = [x, []]
                        observations[key][1].append(int(not delivery["near_landed"]))
    x = np.array([r[0] for r in observations.values()])
    y = np.array([np.mean(r[1]) for r in observations.values()])
    groups = np.array([k[0] for k in observations])
    cross = np.empty_like(y)
    prevalence = np.empty_like(y)
    for fold in np.array_split(
        np.random.default_rng(48003).permutation(np.unique(groups)), 5
    ):
        held = np.isin(groups, fold)
        params = fit(x[~held], y[~held])
        cross[held] = predict(params, x[held])
        prevalence[held] = y[~held].mean()
    threshold = 0.80
    review = []
    for lo, hi in ((0.0, 0.30), (0.30, 0.62), (0.62, threshold), (threshold, 1.01)):
        at = (cross >= lo) & (cross < hi)
        review.append(
            dict(
                lower=lo,
                upper=hi,
                search_decisions=int(at.sum()),
                actual_loss=float(y[at].mean()) if at.any() else None,
            )
        )
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        FIT,
        dict(
            coefficients=[v.tolist() for v in fit(x, y)],
            threshold=threshold,
            reliability=review,
            decisions=len(x),
            release_ids=np.unique(groups).tolist(),
            brier=float(np.mean((cross - y) ** 2)),
            prevalence_brier=float(np.mean((prevalence - y) ** 2)),
            inputs={str(p): _hash(p) for p in FILES},
        ),
    )


def basis(x, mean, scale):
    v = np.clip((np.atleast_2d(x) - mean) / scale, -4.0, 4.0)
    return np.c_[
        np.ones(len(v)),
        v,
        np.maximum(v, 0),
        v[:, 1, None] * v[:, [2, 8, 9, 10, 14]],
        v[:, 2] * v[:, 9],
    ]


def fit(x, y):
    mean, scale = x.mean(0), np.maximum(0.025, x.std(0))
    a = basis(x, mean, scale)
    penalty = np.r_[0.01, np.full(a.shape[1] - 1, 6.0)]

    def loss(w):
        z = a @ w
        return np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * (penalty * w) @ w, a.T @ (
            expit(z) - y
        ) + penalty * w

    result = minimize(loss, np.zeros(a.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return mean, scale, result.x


def predict(params, x):
    mean, scale, weights = params
    return expit(basis(x, mean, scale) @ weights)


class Player(season.Player):
    test = composition.Composed.test
    confirm = composition.Composed.confirm
    recover = composition.Composed.recover

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        d = _read(FIT)
        self.hazard = tuple(map(np.array, d["coefficients"]))
        self.threshold = d["threshold"]

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        if (
            not self.notes
            or self.notes[-1]["tick"] != tick
            or not eligible(self.notes[-1])
        ):
            return
        note = self.notes[-1]
        chance = float(predict(self.hazard, features(self.notes[-2:]))[0])
        note["predicted_delivery_loss"] = chance
        if chance >= self.threshold:
            self.recover(tick, x, q, reference, base, note)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "live"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="practice", mode="live", carry=season.CARRY, cap=240, resume=False
    )
    args = p.parse_args()
    if args.action == "train":
        train()
    else:
        with patch.object(season, "Player", Player):
            season.run(args)
        receipt = _read(args.out)
        receipt["risk_fit"] = _hash(FIT)
        _write(args.out, receipt)
