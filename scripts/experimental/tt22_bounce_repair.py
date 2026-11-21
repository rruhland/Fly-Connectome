"""Repair the already prepared stroke after a camera-observed bounce."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _read, _write
from .tt19_factory import RATE
from .tt19_model_rollout import BallObservation, observed_ball
from .tt19_shoot import HORIZON
from .tt22_bounce_filter import BouncePlayer, state
from .tt22_rally import ROBUST, cooperation_cost
from .tt22_rally_trial import Station

OFFSETS = qmc.Sobol(6, scramble=True, seed=82731).random_base2(6) * 2 - 1


def local_tapes(tape):
    count = len(tape)
    phase = np.arange(1, count + 1)
    poses = np.vstack((np.zeros(5), np.cumsum(tape, axis=0)))
    options = [tape.copy()]
    for row in OFFSETS:
        # Small bounded joint target and phase changes preserve the prepared stroke.
        shifted = np.clip(phase + row[5] * 1.2, 0, count)
        proposal = np.stack([np.interp(shifted, np.arange(count + 1), poses[:, j]) for j in range(5)], axis=1)
        strength = np.minimum(1.0, phase / 2.0)[:, None]
        proposal += strength * (row[:5] * np.array([0.07, 0.15, 0.19, 0.17, 0.12]) / RATE[:5])
        drive = np.zeros(5)
        rates = []
        for goal in proposal:
            rate = np.clip(goal - drive, -1, 1)
            rates.append(rate)
            drive += rate
        options.append(rates)
    candidates = np.zeros((len(options), HORIZON, 5))
    candidates[:, :count] = options
    return candidates


class LocalRepair(BouncePlayer):
    def __init__(self, law, settings):
        super().__init__(law, settings)
        self.late_refine = True

    def refine_shot(self, tick, estimate, current):
        shot = self.shots[-1]
        if "repair" in shot:
            return
        fitted = state(self.history, self.settings)
        if fitted is None or not fitted[2] or fitted[3] < -3:
            return
        p, v, _, tau = fitted
        start, tape = self.plan
        left = tape[tick - start :]
        if len(left) < 3 or shot["tick"] + shot["prediction"]["eta"] - tick < 1.5:
            return
        obs = observed_ball(self.history, (p, v), self.law)
        obs = BallObservation(p, v * 30.0, obs.spin, True)
        candidates = local_tapes(left)
        self.sim.current = self.sim.twin.state()
        lengths = np.full(len(candidates), len(left))
        results = self.sim.predict(candidates, lengths, obs, np.zeros((len(candidates), 9)))
        scores = np.array([cooperation_cost(r) for r in results])
        picks = np.unique(np.r_[0, np.argsort(scores)[:8]])
        perturbed = self.sim.predict(
            np.repeat(candidates[picks], 2, axis=0),
            np.repeat(lengths[picks], 2),
            obs,
            np.tile(ROBUST, (len(picks), 1)),
        )
        alt = np.array([cooperation_cost(r) for r in perturbed]).reshape(-1, 2)
        cost = 0.55 * scores[picks] + 0.30 * alt.mean(axis=1) + 0.15 * alt.max(axis=1)
        best = int(np.argmin(cost))
        take = picks[best] if cost[best] + 0.23 < cost[0] and results[picks[best]]["legal"] else 0
        shot["repair"] = dict(
            tick=tick,
            optical_bounce_offset=float(tau),
            option=int(take),
            old_cost=float(cost[0]),
            best_cost=float(cost[best]),
        )
        if take:
            self.plan = (tick, candidates[take, : len(left)])
            shot["repair"]["before"] = results[0]
            shot["repair"]["after"] = results[take]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8351422)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    settings = _read(args.calibration)["calibration"]["parameters"]
    station = Station(0.30, 0.82, "anchor")
    law = station.player.law
    station.player.close()
    station.player = LocalRepair(law, settings)
    data = dict(
        calibration=settings,
        spread=0.30,
        pace=0.82,
        seed=args.seed,
        rows=[],
        rival="anchor",
        privileged=False,
    )
    try:
        for s in np.random.default_rng(args.seed).integers(2**31, size=args.count):
            row = station.point(int(s))
            data["rows"].append(row)
            _write(args.output, data)
            print(
                int(s), row["longest"], [q.get("repair", {}).get("option") for q in row["shots"]], flush=True
            )
    finally:
        station.close()


if __name__ == "__main__":
    main()
