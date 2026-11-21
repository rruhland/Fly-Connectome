"""Carry the optically learned short-horizon ready position through complete games."""

import argparse

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt21_ready_position import (
    OUTPUT,
    PRACTICE,
    PositionStation,
    fit,
    ready_metrics,
    training_pairs,
)
from .tt21_recovery_trial import ARENA, verify

SEEDS = (1427016321, 541340360)


def new_observation(row):
    decision = row["decision"]
    if decision is None:
        return None
    reads = [(t, np.asarray(p)) for t, p in row["optical"] if t >= decision["tick"]]
    for i in range(2, len(reads)):
        tick, p = reads[i]
        first_tick, earlier = reads[i - 2]
        dt = tick - first_tick
        delta = p - earlier
        if dt <= 4 and -0.69 < p[1] < -0.42 and delta[1] / dt < -0.03:
            later_x = p[0] + (-0.88 - p[1]) * delta[0] / delta[1]
            return {
                "seed": row["seed"],
                "cue": decision["cue"],
                "next_x": float(later_x),
                "observed_tick": tick,
            }
    return None


def run(policy):
    pairs = training_pairs("incoming")
    training = {p["seed"] for p in pairs}
    station = PositionStation()
    previous = None
    try:
        for game_seed in SEEDS:
            score = ServiceScore(first_server=-1)
            start = len(pairs)
            rows = []
            for seed in np.random.default_rng(game_seed).integers(2**31, size=100).tolist():
                if seed in training:
                    raise ValueError("seed overlap with practice or previous adaptation")
                coeff = np.zeros(3) if policy == "center" else fit(pairs)
                row = station.point(seed, score.server, coeff, cue_mode="incoming")
                obs = new_observation(row)
                row["coefficients"] = coeff.tolist()
                row["new_observation"] = obs
                if obs is not None and policy == "online":
                    pairs.append(obs)
                    training.add(seed)
                score.record(row["outcome"])
                rows.append(row)
                if score.finished:
                    break
            if not score.finished:
                raise RuntimeError("completed score required")
            path = OUTPUT / f"ready-game-{policy}-{game_seed}.json.gz"
            result = {
                "policy": policy,
                "seed": game_seed,
                "arena": ARENA,
                "practice": PRACTICE.name,
                "practice_sha256": _hash(PRACTICE),
                "scores": list(score.score),
                "points": rows,
                "previous": previous.name if previous else None,
                "previous_sha256": _hash(previous) if previous else None,
                "observations_before": start,
                "observations_after": pairs.copy(),
                "summary": ready_metrics(rows),
                "replayed": verify(rows),
            }
            _write(path, result)
            print(policy, game_seed, result["scores"], result["summary"], start, len(pairs), flush=True)
            # Reload the entire online spatial state for the next game.
            if policy == "online":
                previous = path
                pairs = _read(previous)["observations_after"]
    finally:
        station.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", choices=("center", "frozen", "online"))
    run(p.parse_args().policy)
