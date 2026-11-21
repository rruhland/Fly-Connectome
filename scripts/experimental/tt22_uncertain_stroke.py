"""Choose a stroke across contact models weighted by earlier optical returns."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_shoot import HORIZON
from .tt22_optical_identification import adjusted
from .tt22_rally import RallyPlayer, cooperation_cost
from .tt22_rally_trial import Station


def posterior(calibration, opening, k=5):
    options = np.array(calibration["options"])
    errors = np.array(
        [s["errors"] for r in calibration["rows"] for s in r["samples"] if s["opening"] == opening]
    )
    # Each practice stroke votes for hypotheses that reproduced its outgoing visible arc.
    likelihood = np.exp(-(errors - errors.min(axis=1, keepdims=True)) / 0.12)
    votes = np.mean(likelihood / likelihood.sum(axis=1, keepdims=True), axis=0)
    # Compress the empirical distribution to a few representatives in parameter space.
    scaled = options / [1.0, 65.0, 0.15, 0.18]
    D = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=2)
    picks = [int(np.argmin(votes @ D))]
    best = D[:, picks[0]]
    for _ in range(k - 1):
        selected = int(np.argmin(votes @ np.minimum(best[:, None], D)))
        picks.append(selected)
        best = np.minimum(best, D[:, selected])
    membership = np.argmin(D[:, picks], axis=1)
    return options[picks], np.bincount(membership, weights=votes, minlength=k)


class EnsemblePlayer(RallyPlayer):
    def __init__(self, law, calibration):
        super().__init__(law)
        self.ensemble = [posterior(calibration, opening) for opening in (False, True)]

    def search(self, estimate, obs, target):
        tapes, params = self.goal_tapes(estimate, target)
        if not len(tapes):
            return None
        lengths = np.clip(np.ceil([p["eta"] for p in params]).astype(int) + 3, 3, HORIZON)
        self.sim.current = self.sim.twin.state()
        native = self.sim.predict(tapes, lengths, obs, np.zeros((len(tapes), 9)))
        scores = np.array([cooperation_cost(r) for r in native])
        if not native[np.argmin(scores)]["hit"]:
            return None
        # Keep representatives from different depths as well as the overall nominal leaders.
        ordered = np.argsort(scores)
        by_depth = {}
        for i in ordered:
            key = params[i]["depth"]
            if len(by_depth.setdefault(key, [])) < 4:
                by_depth[key].append(int(i))
        picks = np.unique(np.r_[ordered[:12], [i for ids in by_depth.values() for i in ids]]).astype(int)
        configs, weights = self.ensemble[int(not self.shots)]
        offsets = []
        for conf in configs:
            alternate = adjusted(obs, conf)
            offsets.append(np.r_[np.zeros(3), alternate.velocity - obs.velocity, alternate.spin - obs.spin])
        n = len(weights)
        trials = self.sim.predict(
            np.repeat(tapes[picks], n, axis=0),
            np.repeat(lengths[picks], n),
            obs,
            np.tile(offsets, (len(picks), 1)),
        )
        risks = np.array([cooperation_cost(r) for r in trials]).reshape(len(picks), n)
        legal = np.array([r["legal"] for r in trials]).reshape(len(picks), n) @ weights
        loss = 0.22 * scores[picks] + 0.63 * (risks @ weights) + 0.15 * np.max(risks, axis=1)
        j = int(np.argmin(loss))
        idx = int(picks[j])
        if not native[idx]["legal"] and legal[j] < 0.50 and estimate[0][1] > -0.64:
            return None
        return tapes[idx, : lengths[idx]], dict(
            params[idx],
            prediction=native[idx],
            candidate_count=len(tapes),
            posterior_legal=float(legal[j]),
            posterior_cost=float(loss[j]),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=8351422)
    p.add_argument("--count", type=int, default=5)
    args = p.parse_args()
    station = Station(0.30, 0.82, "anchor")
    law = station.player.law
    station.player.close()
    station.player = EnsemblePlayer(law, _read(args.calibration))
    data = dict(
        ensemble=[(opts.tolist(), w.tolist()) for opts, w in station.player.ensemble],
        seed=args.seed,
        rows=[],
        spread=0.30,
        pace=0.82,
        rival="anchor",
        privileged=False,
    )
    try:
        for s in np.random.default_rng(args.seed).integers(2**31, size=args.count):
            r = station.point(int(s))
            data["rows"].append(r)
            _write(args.output, data)
            print(int(s), r["longest"], flush=True)
    finally:
        station.close()


if __name__ == "__main__":
    main()
