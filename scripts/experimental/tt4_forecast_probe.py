"""Transfer a local RGB motion predictor; score trajectories independently of the controller."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_arena import TT4Arena, TABLE_HEIGHT
from .tt4_camera import PerspectiveTracker, TT3PerspectiveArena
from .tt4_native_rgb import native_tracker

HZ = 30
HISTORY = 4
HORIZON = 4


def optical_point(samples, tick):
    seen = [samples.get(tick - offset) for offset in reversed(range(HISTORY))]
    if any(p is None for p in seen):
        return None
    points = np.asarray(seen)
    # Local line through the past 100 milliseconds, common to all forecasts.
    t = np.arange(1-HISTORY, 1, dtype=float) / HZ
    v = (t - t.mean()) @ points / ((t - t.mean()) @ (t - t.mean()))
    if not 0.6 < np.linalg.norm(v) < 18.0:
        return None
    current = points.mean(axis=0) - v * t.mean()
    return np.concatenate([current, v]), current + v * HORIZON/HZ


def collect(seed, mini, view, width, height):
    histories = defaultdict(dict)
    true_ball = []  # Read only when scoring target forecasts after the point ends.
    if mini:
        arena = TT3PerspectiveArena(seed, view_seed=seed, view=view, width=width, height=height)
        images = arena.reset()
        tracker = PerspectiveTracker(arena.cameras, coordinates="world")
    else:
        arena = TT4Arena(seed, view_seed=seed, view=view, width=width, height=height, render_mode="mujoco")
        images = arena.reset(seed=seed)
        tracker = native_tracker(arena._camera_rig(), coordinates="world")
    done = False
    try:
        for tick in range(90):
            for tr in tracker.update(images):
                if tr.observed:
                    histories[tr.id][tick] = np.array(tr.position)
            if not mini:
                true_ball.append(arena.ball.copy() - [0, 0, TABLE_HEIGHT])
            if done:
                break
            if mini:
                # TT3 runs at 60 Hz. Both regressors operate at 30 Hz.
                for _ in range(2):
                    images, _, done, _info = arena.step(np.zeros(3))
                    if done:
                        break
            else:
                images, _, term, trunc, _info = arena.step(np.zeros(4))
                done = term or trunc
    finally:
        if not mini:
            arena.close()
    return histories, np.array(true_ball)


def training_data(seeds, view, width, height):
    features, residuals = [], []
    for seed in seeds:
        histories, _ = collect(seed, True, view, width, height)
        for series in histories.values():
            for t in series:
                # Require an unbroken optical track across the training label.
                if any(i not in series for i in range(t+1, t+HORIZON+1)):
                    continue
                data = optical_point(series, t)
                if data is not None:
                    x, base = data
                    features.append(x)
                    residuals.append(series[t+HORIZON] - base)
    x = np.asarray(features)
    y = np.asarray(residuals)
    # One scale for position, one for speed, both estimated on the source only.
    scale = np.repeat([np.median(x[:, :3].std(axis=0)), np.median(x[:, 3:].std(axis=0))], 3)
    tree = cKDTree(x / scale)
    return tree, y, scale


def score(seeds, source, view, width, height):
    tree, corrections, scale = source
    totals = Counter()
    rows = []
    for seed in seeds:
        histories, truth = collect(seed, False, view, width, height)
        errors = defaultdict(list)
        eligible = len(truth) - HORIZON - HISTORY + 1
        totals.update(eligible=eligible)
        for t in range(HISTORY-1, len(truth)-HORIZON):
            # Evaluator identifies whether the anonymous tracked region is the ball.
            # The forecasts, given this track, use only its past RGB-derived positions.
            candidates = []
            for s in histories.values():
                if t in s and np.linalg.norm(s[t]-truth[t]) <= .15:
                    sample = optical_point(s, t)
                    if sample is not None:
                        candidates.append((np.linalg.norm(s[t]-truth[t]), sample))
            if not candidates:
                continue
            _, (x, base) = min(candidates, key=lambda c: c[0])
            dist, idx = tree.query(x / scale, k=min(24, len(corrections)))
            weights = 1. / (.05 + dist)
            local = np.average(corrections[idx], axis=0, weights=weights)
            outcome = truth[t+HORIZON]
            for name, pred in [('constant_velocity', base), ('mini_average', base + corrections.mean(axis=0)),
                               ('mini_local', base + local)]:
                errors[name].append(float(100 * np.linalg.norm(pred-outcome)))
        totals.update(scored=len(errors['constant_velocity']))
        rows.append(dict(seed=int(seed), eligible=eligible, errors_cm=errors))
    summaries = {}
    for mode in ['constant_velocity', 'mini_average', 'mini_local']:
        all_errors = [e for row in rows for e in row['errors_cm'][mode]]
        summaries[mode] = {"mean_cm": float(np.mean(all_errors)), "median_cm": float(np.median(all_errors))}
    improvements = [
        np.mean(row["errors_cm"]["constant_velocity"]) - np.mean(row["errors_cm"]["mini_local"])
        for row in rows if row["errors_cm"]["mini_local"]
    ]
    random = np.random.default_rng(41811)
    means = random.choice(improvements, (4000, len(improvements)), replace=True).mean(axis=1)
    paired = {"mean_cm_saved": float(np.mean(improvements)), "bootstrap95": np.quantile(means, [.025, .975]).tolist(),
              "points_improved": int(np.count_nonzero(np.asarray(improvements) > 0)), "points_scored": len(improvements)}
    return {"counts": dict(totals), "errors": summaries, "paired_by_point": paired, "points": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--source-points", type=int, default=24)
    p.add_argument("--test-seed", type=int, default=921754)
    p.add_argument("--test-points", type=int, default=16)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--height", type=int, default=144)
    p.add_argument("--views", nargs="+", default=["first_person","overhead_side"])
    args = p.parse_args()
    results = {}
    for view in args.views:
        source = training_data(range(183015, 183015 + args.source_points), view, args.width, args.height)
        result = score(range(args.test_seed, args.test_seed + args.test_points), source, view, args.width, args.height)
        results[view] = dict(source_windows=len(source[1]), learned_mean_correction_m=source[1].mean(axis=0).tolist(), **result)
        print(view, result['counts'], result['errors'], flush=True)
    args.output.write_text(json.dumps(dict(source_points=args.source_points, test_points=args.test_points, test_seed=args.test_seed,
                                          width=args.width, height=args.height, results=results), separators=(",",":")))
