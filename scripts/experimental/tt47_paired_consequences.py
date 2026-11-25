"""Learn changes in reply difficulty from two different scored actions on the same physical incoming offer."""

import argparse
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt41_reply_control as paired
from . import tt43_outgoing_coverage as ranking
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt47_paired")
MODEL = ROOT / "effect.json"
SOURCES = (
    "tt42/learning-48",
    "tt42/diverse-learning-48",
    "tt42/split-memory-48",
    "tt42/graded-practice",
    "tt42/uniform-practice",
    "tt42/no-more-practice",
    "tt42/after12",
    "tt42/after24",
    "tt42/after36",
    "tt42/after48",
    "tt42/diverse-evaluation",
    "tt43_servo/physical",
    "tt43_servo/instant-games",
    "tt45_composition/held",
    "tt47_symmetry/held",
    "tt47_interpolate/held",
    "tt46_reliability/held",
    "tt45_tracking/held",
    "tt43_preparation/outgoing-coverage-live",
)


def feature(z):
    x = np.atleast_2d(z)
    u = np.clip(
        (x - [0.0, 0.63, 0.35, 0.0, 1.29, 1.185, 2.30]) / ranking.SCALE, -3.0, 3.0
    )
    return np.column_stack(
        (
            ranking.design(x),
            u[:, 1] * u[:, 6],
            u[:, 4] * u[:, 6],
            u[:, 5] * u[:, 6],
            u[:, 1] * u[:, 5],
            u[:, 3] * u[:, 0],
        )
    )


def fit(out):
    files = [
        Path(p)
        for p in _read(Path("artifacts/tt41/reply-control.json"))["source_sha256"]
    ]
    files += [Path("artifacts") / (n + ".json.gz") for n in SOURCES]
    groups, sources = paired.take(list(dict.fromkeys(files)))
    held = {
        c["id"] for deck in ("practice", "assessment") for c in _read(season.PLAN)[deck]
    }
    samples = []
    for ix, (key, group) in enumerate(groups.items()):
        if key[0] in held:
            continue
        vals = [
            (np.asarray(x), float(h)) for act, h, x in group.values() if x is not None
        ]
        if len(vals) < 2:
            continue
        for (x, h), (y, j) in combinations(vals, 2):
            if np.linalg.norm((x - y) / ranking.SCALE) < 0.04:
                continue
            samples.append(
                dict(
                    release=key[0],
                    group=ix,
                    first=x.tolist(),
                    second=y.tolist(),
                    heights=[h, j],
                )
            )
    if len(samples) < 20:
        raise ValueError("Insufficient matched physical actions")
    a, b = (
        np.asarray([z["first"] for z in samples]),
        np.asarray([z["second"] for z in samples]),
    )
    delta = feature(a) - feature(b)
    heights = np.array([z["heights"] for z in samples])
    utility = np.clip((heights - 0.94) / 0.09, 0.0, 1.0)
    target = utility[:, 0] - utility[:, 1]
    g = np.array([z["group"] for z in samples])
    releases = np.array([z["release"] for z in samples])
    _, counts = np.unique(g, return_counts=True)
    freq = dict(zip(np.unique(g), counts, strict=True))
    weight = np.array([1.0 / np.sqrt(freq[i]) for i in g])

    def solve(use):
        x = delta[use] * weight[use, None]
        y = target[use] * weight[use]
        return np.linalg.solve(x.T @ x + 3.0 * np.eye(x.shape[1]), x.T @ y)

    prediction = np.zeros(len(samples))
    for fold in np.array_split(
        np.random.default_rng(47277).permutation(np.unique(releases)), 6
    ):
        mask = np.isin(releases, fold)
        prediction[mask] = delta[mask] @ solve(~mask)
    strong = np.abs(target) > 0.10
    result = dict(
        pairs=len(g),
        groups=len(freq),
        releases=len(set(releases)),
        informative=int(strong.sum()),
        correct=float(
            np.average(
                np.sign(prediction[strong]) == np.sign(target[strong]),
                weights=weight[strong] ** 2,
            )
        )
        if strong.any()
        else None,
        base_rmse=float(np.sqrt(np.average(target**2, weights=weight**2))),
        held_rmse=float(
            np.sqrt(np.average((prediction - target) ** 2, weights=weight**2))
        ),
    )
    out.parent.mkdir(exist_ok=True, parents=True)
    _write(
        out,
        dict(
            rows=samples,
            fit=solve(np.ones(len(g), bool)).tolist(),
            metrics=result,
            held=sorted(held),
            sources={p: _hash(Path(p)) for p in sources},
        ),
    )
    print(result, flush=True)


class Contrast:
    def __init__(self, doc):
        self.w = np.array(doc["fit"])
        self.teacher = ranking.Coverage(_read(fresh.RISK))

    def risk(self, x):
        risk, supported = self.teacher.risk(x)
        effects = feature(x) @ self.w
        # Only relative risks are used in ranking. Keep the observed region
        # supported by the existing fit and reduce extreme extrapolation.
        score = np.clip(effects - np.median(effects), -0.4, 0.4)
        return risk - 0.6 * score, supported


def run(a):
    doc = _read(MODEL)
    if not {r["id"] for r in _read(season.PLAN)[a.deck][: a.n]} <= set(doc["held"]):
        raise ValueError("Release overlap")
    # Construct before patching to avoid changing the independent baseline.
    model = Contrast(doc)
    with patch.object(fresh.shot, "Coverage", lambda unused: model):
        season.run(a)
    d = _read(a.out)
    d.update(paired_effects=_hash(MODEL))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "live", "frozen"))
    p.add_argument("out", type=Path)
    p.add_argument("--n", default=16, type=int)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(deck="practice", carry=season.CARRY, cap=240, resume=False)
    a = p.parse_args()
    fit(a.out) if a.mode == "fit" else run(a)
