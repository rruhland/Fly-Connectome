"""Fit a high-arc, disturbance-recovery prior in compliant 3D source worlds."""

import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .tt24_world_prior import CHECKPOINT, WEIGHTS, WorldPrior
from .tt24_world_prior_train import data, save


SOURCE = Path(__file__).with_name("tt24_world_margin_native.cpp")
OUTPUT = CHECKPOINT.with_name("world-margin.npz")
RECEIPT = CHECKPOINT.with_name("world-margin-source.json")
FIELDS = (
    "rebounds", "frames", "longest", "contacts", "rise_total_m", "margin_credit", "high_arcs",
    "impulses", "recoveries", "requested_position_speed_m_s", "requested_normal_speed_s",
    "requested_position_acceleration_m_s2", "requested_normal_acceleration_s2",
    "requested_rate_over_limit", "actuator_rate_clamped", "offset_effort",
)
DEVELOPMENT = (2468051661, 2468051663)
QUALIFICATION = (
    ("compliant", 0, 9840163851),
    ("disturbed", 1, 9840163853),
    ("stronger_disturbance", 2, 9840163857),
)
OBJECTIVE = dict(survival=7., returns=.13, high_arc_margin=1.30, recovered=.85,
                 speed=.28, acceleration=.12, effort=.22)


class MarginWorlds:
    """Batch rollouts in the compliant actuator and impulse simulator."""

    def __init__(self, workers=2, frames=300):
        self.workers = workers
        self.frames = frames
        self.workspace = TemporaryDirectory(prefix="tt24-world-margin-")
        library = Path(self.workspace.name) / "margin.so"
        try:
            subprocess.run(["c++", "-std=c++17", "-O3", "-ffast-math", "-fopenmp", "-shared",
                            "-fPIC", str(SOURCE), "-o", str(library)], check=True)
            self.native = ctypes.CDLL(str(library))
            self.evaluate = self.native.tt24_world_margin_rollouts
            pointer, integer = ctypes.c_void_p, ctypes.c_int
            self.evaluate.argtypes = [pointer, integer, pointer, integer, integer, integer, integer, pointer]
            self.evaluate.restype = None
        except BaseException:
            self.workspace.cleanup()
            raise

    def close(self):
        self.workspace.cleanup()

    def play(self, weights, seeds, profile):
        weights = np.ascontiguousarray(np.atleast_2d(weights), dtype=np.float64)
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        if weights.ndim != 2 or weights.shape[1] != WEIGHTS or seeds.ndim != 1:
            raise ValueError("invalid policy or seed dimensions")
        if not len(weights) or not len(seeds) or not np.isfinite(weights).all():
            raise ValueError("need nonempty finite policies and world batches")
        if profile not in (0, 1, 2):
            raise ValueError("invalid physics profile")
        telemetry = np.empty((len(weights), len(seeds), len(FIELDS)), dtype=np.float64)
        self.evaluate(weights.ctypes.data, len(weights), seeds.ctypes.data, len(seeds), profile,
                      self.frames, self.workers, telemetry.ctypes.data)
        return telemetry


def scores(rows, frames):
    """Value long runs, high apex clearance and recovery, with smooth commands."""
    returns, duration, _, _, _, margin, _, _, recoveries = np.moveaxis(rows[..., :9], -1, 0)
    pos_rate, norm_rate, pos_acc, norm_acc = np.moveaxis(rows[..., 9:13], -1, 0)
    speed = pos_rate / .35 + norm_rate / 1.2
    acceleration = pos_acc / 20. + norm_acc / 65.
    return (OBJECTIVE["survival"] * duration / frames * np.minimum(returns / 4., 1.)
            + OBJECTIVE["returns"] * returns + OBJECTIVE["high_arc_margin"] * margin
            + OBJECTIVE["recovered"] * recoveries - OBJECTIVE["speed"] * speed
            - OBJECTIVE["acceleration"] * acceleration - OBJECTIVE["effort"] * rows[..., 15])


def summarize(rows, frames):
    mean = dict(zip(FIELDS, rows.mean(axis=0).tolist(), strict=True))
    rebounds, impulses = rows[:, 0].sum(), rows[:, 7].sum()
    return mean | {
        "episodes": len(rows),
        "objective": float(scores(rows, frames).mean()),
        "survival_full": float(np.mean(rows[:, 1] == frames)),
        "rebound_weighted_rise_m": float(rows[:, 4].sum() / rebounds) if rebounds else None,
        "high_arc_fraction": float(rows[:, 6].sum() / rebounds) if rebounds else None,
        "recovery_per_impulse": float(rows[:, 8].sum() / impulses) if impulses else None,
    }


def fit(worlds, args, canon):
    rng = np.random.default_rng(args.seed)
    development = [data(seed, args.development) for seed in DEVELOPMENT]
    profiles = tuple(range(len(development)))
    coefficients = canon.copy()
    best = coefficients.copy()
    history, launches = [], []
    exposure = dict(training_episodes=0, training_frames=0, development_episodes=0, development_frames=0)

    def evaluate(weights, step):
        rows = np.concatenate([worlds.play(weights, development[k], k)[0] for k in profiles])
        exposure["development_episodes"] += len(rows)
        exposure["development_frames"] += int(rows[:, 1].sum())
        summary = summarize(rows, worlds.frames)
        history.append(dict(iteration=step, **summary))
        print(f"{step:3d}  objective {summary['objective']:.3f}  rebounds {summary['rebounds']:.2f}"
              f"  rise {(summary['rebound_weighted_rise_m'] or 0.):.3f}m  pos speed "
              f"{summary['requested_position_speed_m_s']:.3f}m/s", flush=True)
        return summary["objective"]

    priority = evaluate(coefficients, 0)
    selected = 0
    for iteration in range(1, args.iterations + 1):
        progress = (iteration - 1) / max(1, args.iterations - 1)
        perturbation = rng.normal(size=(args.directions, WEIGHTS))
        spread = .145 * (1. - .55 * progress)
        variants = coefficients + spread * np.concatenate((perturbation, -perturbation))
        steady, disturbed = args.worlds // 3, args.worlds - args.worlds // 3
        batch = [rng.integers(1, 2**63, count, dtype=np.uint64) for count in (steady, disturbed)]
        launches.extend([seed.tolist() for seed in batch])
        rows = np.concatenate([worlds.play(variants, batch[k], k) for k in profiles], axis=1)
        exposure["training_episodes"] += int(np.prod(rows.shape[:2]))
        exposure["training_frames"] += int(rows[..., 1].sum())
        positive, negative = scores(rows, worlds.frames).mean(axis=1).reshape(2, args.directions)
        elite = np.argsort(-np.maximum(positive, negative), kind="stable")[:args.elite]
        std = np.std(np.r_[positive[elite], negative[elite]])
        advantage = (positive[elite] - negative[elite]) / max(.15, std)
        update = np.mean(perturbation[elite] * advantage[:, None], axis=0)
        coefficients = np.clip(coefficients + (.086 * (1. - .55 * progress)) * update, -5.5, 5.5)
        if iteration % 8 == 0 or iteration == args.iterations:
            value = evaluate(coefficients, iteration)
            if value > priority:
                priority, best, selected = value, coefficients.copy(), iteration
    return best, (exposure | dict(selected_iteration=selected, development_curve=history,
                                  training_seeds=launches, development_seeds=[x.tolist() for x in development]))


def qualification(world, policies, name, profile, seed, count, fitted):
    launch = data(seed, count)
    if fitted.intersection(launch.tolist()):
        raise ValueError("fit and qualification seeds overlap")
    result = world.play(tuple(policies.values()), launch, profile)
    summaries = {k: summarize(rows, world.frames) for k, rows in zip(policies, result, strict=True)}
    comparison = result[1, :, 0] - result[0, :, 0]
    rng = np.random.default_rng(8163 + profile)
    paired_bootstrap = comparison[rng.integers(len(launch), size=(3000, len(launch)))].mean(axis=1)
    print(name, {k: {m: round(s[m], 3) for m in ("objective", "rebounds", "rebound_weighted_rise_m",
                                                "requested_position_speed_m_s") if s[m] is not None}
                 for k, s in summaries.items()}, flush=True)
    return {
        "profile": profile, "seeds": launch.tolist(), "statistics": summaries,
        "paired_rebound_difference_95ci": np.quantile(paired_bootstrap, (.025, .975)).tolist(),
        "episodes": int(np.prod(result.shape[:2])), "frames": int(result[..., 1].sum()),
        "per_episode": {k: rows.tolist() for k, rows in zip(policies, result, strict=True)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--seed", type=int, default=814305331)
    parser.add_argument("--iterations", type=int, default=176)
    parser.add_argument("--directions", type=int, default=38)
    parser.add_argument("--elite", type=int, default=16)
    parser.add_argument("--worlds", type=int, default=39)
    parser.add_argument("--development", type=int, default=144)
    parser.add_argument("--held", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    if args.output.resolve() == args.initial.resolve():
        parser.error("output path must differ from initial policy")
    if min(args.iterations, args.directions, args.elite, args.development, args.workers, args.held) < 1:
        parser.error("budgets must be positive")
    if args.elite > args.directions or args.worlds < 3 or args.frames < 3:
        parser.error("invalid direction, world or frame budget")

    start = perf_counter()
    canon = WorldPrior.load(args.initial).weights
    world = MarginWorlds(args.workers, args.frames)
    try:
        prior, receipt = fit(world, args, canon)
        save(args.output, prior)
        policies = dict(canonical=canon, margin=prior, idle=np.zeros(WEIGHTS))
        fitted = {seed for batch in receipt["training_seeds"] + receipt["development_seeds"] for seed in batch}
        receipt["qualification"] = {
            name: qualification(world, policies, name, profile, seed, args.held, fitted)
            for name, profile, seed in QUALIFICATION
        }
        receipt |= {
            "schema": "tt24-world-margin-1",
            "observed_channels": FIELDS,
            "frames_per_episode": args.frames,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "objective_coefficients": OBJECTIVE,
            "initial_policy_sha256": sha256(args.initial.read_bytes()).hexdigest(),
            "learned_policy_sha256": sha256(args.output.read_bytes()).hexdigest(),
            "physics_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
            "shared_physics_sha256": sha256(SOURCE.with_name("tt24_world_prior_native.cpp").read_bytes()).hexdigest(),
            "elapsed_s": perf_counter() - start,
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, separators=(",", ":")) + "\n")
    finally:
        world.close()


if __name__ == "__main__":
    main()
