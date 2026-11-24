"""Learn a two-posture receiving motion from point-by-point scalar feedback."""

import argparse
from time import perf_counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _write
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver
from .tt29_reward_practice import play, practice_feed


def feedback(row):
    if not row["contact"]:
        return 0.0
    if not row["incoming"]:
        return 0.0
    if row["legal"]:
        return 4.0 + min(0.2, 0.1 * max(0.0, (row["net_z"] or 0.92) - 0.92))
    landing = row["descending_xy"]
    if landing is None:
        return 0.5 + 0.25 * (row["net_z"] is not None)
    lateral, depth = landing
    error = np.hypot(max(0.0, abs(lateral) - 0.6), depth - 0.78)
    return 1.0 + np.exp(-1.15 * error) + row["legal"] * 2.0


def run(draws, seed, output):
    a = ReferenceReceiver()
    axes = a._arm_ids[1][2][:5]
    low, high = a._joint_low[axes], a._joint_high[axes]
    home = a.joints[:5].copy()
    rng, feeds = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    prior = np.r_[home, np.zeros(5), 0.40]
    std = np.array([0.45, 0.65, 0.70, 1.0, 0.7, 0.45, 0.52, 0.55, 0.75, 0.7, 0.09])
    low = np.r_[low, np.full(5, -1.65), 0.19]
    high = np.r_[high, np.full(5, 1.65), 0.64]
    elite, history = [], []
    started = perf_counter()

    def trial(genes, shot):
        return play(a, shot, genes[:5], (genes[5:10], genes[-1]))

    try:
        for i in range(draws):
            if elite and rng.random() < 0.85:
                j = rng.choice(
                    len(elite),
                    p=(1 / (1 + np.arange(len(elite))))
                    / sum(1 / (1 + np.arange(len(elite)))),
                )
                parent = elite[j][1]
                scale = rng.choice([0.055, 0.18, 0.52, 0.95])
            else:
                parent, scale = prior, 1.0
            genes = np.clip(
                parent + rng.normal(size=len(prior)) * std * scale, low, high
            )
            row = trial(genes, practice_feed(feeds, i))
            reward = feedback(row)
            if reward > 0.0:
                check = trial(genes, practice_feed(feeds, i + draws))
                reward = (reward + feedback(check)) / 2
                elite.append((reward, genes))
                elite.sort(key=lambda p: -p[0])
                elite = elite[:24]
            history.append([int(row["contact"]), int(row["legal"]), reward])
            if (i + 1) % 128 == 0:
                print(
                    i + 1,
                    np.sum(history, axis=0)[:2],
                    elite[0][0] if elite else 0,
                    flush=True,
                )
        # Choose on a development deck. Held target seeds are disjoint from search and model selection.
        validation_feeds = np.random.default_rng(2901168)
        validation = [practice_feed(validation_feeds, draws * 3 + i) for i in range(10)]
        ranked = [(sum(feedback(trial(g, s)) for s in validation), g) for _, g in elite]
        best = max(ranked, key=lambda p: p[0])[1] if ranked else prior
        hrng = np.random.default_rng(29076384)
        held = []
        for i in range(40):
            s = practice_feed(hrng, 900000 + i)
            held.append(
                dict(shot=s, stationary=play(a, s, None), learned=trial(best, s))
            )
        result = dict(
            seed=seed,
            draws=draws,
            initial_angles=home.tolist(),
            genes=best.tolist(),
            history=history,
            held=held,
            seconds=perf_counter() - started,
        )
        _write(output, result)
        print(
            output,
            {k: sum(r[k]["legal"] for r in held) for k in ("stationary", "learned")},
            flush=True,
        )
    finally:
        a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws", type=int, default=768)
    p.add_argument("--seed", type=int, default=290361)
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT.parent / "tt29" / "shaped-practice.json.gz",
    )
    args = p.parse_args()
    run(args.draws, args.seed, args.output)
