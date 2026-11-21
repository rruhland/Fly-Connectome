"""Rank SO-101 strokes using a regularized outcome model fitted to actual practice returns."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .tt4_transfer_trial import _read, _write
from .tt19_shoot import HORIZON
from .tt22_rally import ROBUST, RallyPlayer, cooperation_cost
from .tt22_rally_trial import Station


def predictors(shot, predicted, opening):
    xyz = np.asarray(shot["anchor"])
    landing = predicted["landing"]
    u = np.array(
        [
            xyz[0] / 0.15,
            (xyz[1] + 0.91) / 0.09,
            (xyz[2] - 0.22) / 0.15,
            (shot["pitch"] - 0.55) / 0.3,
            (shot["pace"] - 0.025) / 0.018,
            (shot["lead"] - 1.45) / 0.75,
            (shot["eta"] - 8) / 5.0,
            (predicted["net"] - 0.26) / 0.15 if predicted["net"] is not None else -2.0,
            (landing[1] - 0.5) / 0.4 if landing is not None else -2.0,
            float(opening),
        ]
    )
    u = np.clip(u, -3, 3)
    return np.r_[1.0, u, u[1:6] ** 2, u[2] * u[3:7], u[9] * u[1:6], u[3] * u[4], u[4] * u[5]]


def examples(row):
    shots = row["shots"]
    for i, shot in enumerate(shots):
        stop = shots[i + 1]["tick"] if i + 1 < len(shots) else len(row["actions"]) + 1
        good = any(
            shot["tick"] < e["step"] <= stop and e["kind"] == "legal_return" and e["side"] == 1
            for e in row["events"]
        )
        yield predictors(shot, shot["prediction"], i == 0), float(good)


def fit(records):
    x = np.stack([r[0] for r in records])
    y = np.array([r[1] for r in records])

    def obj(w):
        z = x @ w
        return (np.logaddexp(0, z).sum() - y @ z + 3 * (w @ w), x.T @ (expit(z) - y) + 6 * w)

    return minimize(obj, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B").x


class OutcomePlayer(RallyPlayer):
    def __init__(self, law, weights, gain):
        super().__init__(law)
        self.weights, self.gain = weights, gain

    def search(self, estimate, observation, target):
        tapes, params = self.goal_tapes(estimate, target)
        if not len(tapes):
            return None
        lengths = np.clip(np.ceil([p["eta"] for p in params]).astype(int) + 3, 3, HORIZON)
        self.sim.current = self.sim.twin.state()
        trials = self.sim.predict(tapes, lengths, observation, np.zeros((len(tapes), 9)))
        prob = expit(
            np.array([predictors(p, r, not self.shots) for p, r in zip(params, trials)]) @ self.weights
        )
        nominal = np.array([cooperation_cost(r) for r in trials])
        picks = np.argsort(nominal - self.gain * prob)[:8]
        if not trials[picks[0]]["hit"]:
            return None
        variants = self.sim.predict(
            np.repeat(tapes[picks], 2, axis=0),
            np.repeat(lengths[picks], 2),
            observation,
            np.tile(ROBUST, (len(picks), 1)),
        )
        robust = np.array([cooperation_cost(r) for r in variants]).reshape(-1, 2)
        final = (
            0.65 * nominal[picks]
            + 0.25 * robust.mean(axis=1)
            + 0.1 * robust.max(axis=1)
            - self.gain * prob[picks]
        )
        idx = int(picks[np.argmin(final)])
        if not trials[idx]["legal"] and estimate[0][1] > -0.64:
            return None
        return tapes[idx, : lengths[idx]], dict(
            params[idx], prediction=trials[idx], candidate_count=len(tapes), fitted_success=float(prob[idx])
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--practice", nargs="+", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8351422)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--gain", type=float, default=1.25)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    records = []
    seeds = set()
    for source in args.practice:
        data = _read(source)
        if data.get("privileged") is not False:
            raise ValueError("only use scored RGB practice")
        for row in data.get("rows", data.get("practice", [])):
            seeds.add(row["seed"])
            records.extend(examples(row))
    if args.shuffle:
        y = np.random.default_rng(24006).permutation([r[1] for r in records])
        records = [(r[0], yr) for r, yr in zip(records, y)]
    weights = fit(records)
    station = Station(0.30, 0.82, "anchor")
    law = station.player.law
    station.player.close()
    station.player = OutcomePlayer(law, weights, args.gain)
    result = dict(
        practice=sorted(seeds),
        weights=weights.tolist(),
        examples=len(records),
        gain=args.gain,
        spread=0.30,
        pace=0.82,
        rival="anchor",
        privileged=False,
        seed=args.seed,
        rows=[],
    )
    try:
        for v in np.random.default_rng(args.seed).integers(2**31, size=args.count):
            if v in seeds:
                raise ValueError("held practice overlap")
            row = station.point(int(v))
            result["rows"].append(row)
            _write(args.output, result)
            print(int(v), row["longest"], flush=True)
    finally:
        station.close()


if __name__ == "__main__":
    main()
