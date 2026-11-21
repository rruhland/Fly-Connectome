"""Test a practice-trained stroke ranker on wider serves and a learned physical partner."""

from pathlib import Path
from time import perf_counter
import argparse

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt22_opponent_learn import _replay, _station
from .tt22_opponent_trial import FarTape
from .tt22_point_feedback import OutcomePlayer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ranker", type=Path, required=True)
    p.add_argument("--partner", type=Path, required=True)
    p.add_argument("--seed", type=int, default=56024322)
    p.add_argument("--count", type=int, default=12)
    args = p.parse_args()
    learned = _read(args.ranker)
    partner = _read(args.partner)
    profile = partner["fit"]["profile"]
    weights = np.array(learned["weights"])
    stations = {}
    start = perf_counter()
    report = dict(
        profile=profile,
        seed=args.seed,
        weights=weights.tolist(),
        training_seeds=learned["practice"],
        rows={"frozen": [], "trained": []},
    )
    try:
        for arm in report["rows"]:
            station = _station(profile)
            law = station.player.law
            station.player.close()
            station.player = OutcomePlayer(law, weights, learned["gain"] if arm == "trained" else 0.0)
            stations[arm] = (station, FarTape(station.arena))
        for index, seed in enumerate(np.random.default_rng(args.seed).integers(2**31, size=args.count)):
            if int(seed) in report["training_seeds"]:
                raise ValueError("practice/test overlap")
            for arm in (("frozen", "trained") if index % 2 == 0 else ("trained", "frozen")):
                station, tape = stations[arm]
                tape.steps.clear()
                row = station.point(int(seed))
                row["rival_actions"] = tape.tape(len(row["actions"]))
                report["rows"][arm].append(row)
                _write(args.output, report)
                print(arm, int(seed), row["longest"], flush=True)
    finally:
        for station, _ in stations.values():
            station.close()
    report["verified"] = sum(_replay(rows) for rows in report["rows"].values())
    report["seconds"] = perf_counter() - start
    _write(args.output, report)


if __name__ == "__main__":
    main()
