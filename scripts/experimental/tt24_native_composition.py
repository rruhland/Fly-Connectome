"""Isolate native Pong lateral control on a previously practiced five-joint body."""
import argparse
import gzip
import json
import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _features, _INC
from .tt24_world_arm import DIR, SOURCE, Grounding, perception
from .tt24_world_prior import WorldPrior


def play(arena, seed, mode, public_decoder):
    pupil = WristPupil.load(SOURCE)
    obs = arena.reset(seed=seed, split="held")
    home = obs["joints"].copy()
    inverse = Grounding(public_decoder, home)
    prior = None
    if mode in ("pong", "shuffled", "matched_cold"):
        prior = WorldPrior.load(DIR / f"pong-world-strong-initial-{mode}.npz")

    actions = []
    while True:
        full = pupil.sensor.observe(obs)
        inputs = full.copy()
        if mode != "intact":
            # The common demonstrated body readout sees vertical flight and
            # its own full pose. Only the outside policy sees lateral flight.
            inputs[[0, 1, 3, 4]] = 0.
        q_goal = home + pupil.actor.target(_features(inputs, obs["joints"], home))
        if prior is not None or mode == "zero":
            xy = prior.target(full)[:2] if prior is not None else np.zeros(2)
            current = inverse.pose(q_goal)
            delta = np.r_[inverse.origin[:2] + xy - current[:2], 0., 0., 0.]
            jacobian = inverse.jacobian(q_goal) / inverse.scale[:, None]
            step = np.linalg.solve(jacobian.T @ jacobian + .24*np.eye(5),
                                   jacobian.T @ (delta/inverse.scale))
            q_goal += np.clip(step, -.18, .18)
        command = np.clip((q_goal-home-pupil.issued)/_INC, -.70, .70)
        pupil.issued += _INC*command
        actions.append(command.tolist())
        obs, _, done = arena.step(command)
        if done:
            break
    judge = arena.judge_metrics()
    arena.reset(seed=seed, split="held", render=False)
    for command in actions:
        arena.step(command, render=False)
    if arena.judge_metrics() != judge:
        raise ValueError("different physical replay")
    return dict(seed=seed, run=judge["longest_consecutive_rebounds"],
                score=judge["rebounds"], actions=actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=354519128)
    parser.add_argument("--modes", nargs="+", choices=("pong","shuffled","zero","matched_cold","intact","muted"),
                        default=["pong","shuffled","zero","matched_cold","intact","muted"])
    args = parser.parse_args()
    arena = BounceArena(view="wrist", width=192, height=144, max_steps=450)
    decoder = perception(False, sensor_file=DIR/"public-body-fused.npz").decoder
    seeds = np.random.default_rng(args.seed).integers(1, 2**31-1, args.count).tolist()
    try:
        for mode in args.modes:
            rows = [play(arena, seed, mode, decoder) for seed in seeds]
            print(mode, [row["run"] for row in rows], flush=True)
            with gzip.open(DIR/f"native-composition-{mode}.json.gz","wt") as out:
                json.dump(dict(mode=mode, verified=True, rows=rows, horizon=450, seed=args.seed,
                               target_motor_demos=4, new_target_training=0), out, separators=(",",":"))
    finally:
        arena.close()


if __name__ == "__main__":
    main()
