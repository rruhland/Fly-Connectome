"""Learn which combinations of motor changes steer a rebounding ball in solo practice."""

import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _write
from .tt24_bounce import _sample_drop
from .tt24_bounce_ident import _ellipse
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_metric_stereo import triangulate
from .tt24_public_body import stationary, stock
from .tt26_bounce_optics import impact_candidates
from .tt26_optical_transfer import decoder

OUT = Path("artifacts/tt28")
DOSES = (24, 96)


def toss(seed, split, regime, speeds=(0.36, 0.95)):
    drop = _sample_drop(seed, split, regime)
    rng = np.random.default_rng(seed ^ 172830)
    direction = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(*speeds)
    dv = speed * np.array([np.cos(direction), np.sin(direction)])
    velocity, offset = np.array(drop.velocity), np.array(drop.offset)
    # The launcher tosses toward the initial paddle vicinity, from different directions.
    falling_time = (velocity[2] + np.sqrt(velocity[2] ** 2 + 2 * 9.81 * offset[2])) / 9.81
    offset[:2] -= falling_time * dv
    velocity[:2] += dv
    return replace(drop, offset=tuple(offset), velocity=tuple(velocity))


def play(arena, rig, eyes, player, seed, adjustment):
    obs = arena.reset(seed=seed, split="practice")
    player.begin_release(obs)
    angles, pixels, tape = [], [], []
    applied = np.zeros(5)
    for _ in range(arena.max_steps):
        angles.append(obs["joints"].copy())
        coords = [_ellipse(image) for image in arena.fixed_images()]
        pixels.append([x[:2] if x is not None else [np.nan, np.nan] for x in coords])
        policy = player.act(obs)
        cmd = np.clip(policy + (adjustment - applied) / (5.2 / 60), -1, 1)
        applied += (cmd - policy) * (5.2 / 60)
        tape.append(cmd.tolist())
        obs, _, done = arena.step(cmd)
        if done:
            break
    points, _, _ = triangulate(eyes, np.asarray(pixels), tolerance=0.018)
    detected = impact_candidates(
        dict(ball=points, joint_angles=np.array(angles)), np.array([0, 0, -9.81]), rig
    )
    event = None
    if detected:
        hit = detected[0]
        _, rotation = rig.links.read(np.array([angles[hit["tick"]]]))
        incoming, outgoing = [rotation[0].T @ hit[n] for n in ("incoming_m_s", "outgoing_m_s")]
        event = dict(
            incoming=incoming.tolist(),
            outgoing=outgoing.tolist(),
            horizontal_in=float(np.linalg.norm(hit["incoming_m_s"][:2])),
        )
    return dict(
        seed=seed,
        offset=adjustment.tolist(),
        event=event,
        rebounds=arena.judge_metrics()["rebounds"],
        steps=len(tape),
        actions=tape,
    )


def motor_basis(rows):
    seen = [r for r in rows if r["event"] is not None]
    if len(seen) < 10:
        raise ValueError(f"Only {len(seen)} visible reversals")
    x = np.array([r["offset"] for r in seen])
    incoming = np.array([r["event"]["incoming"] for r in seen])
    outgoing = np.array([r["event"]["outgoing"] for r in seen])
    # Absolute outgoing velocity in the gripper frame already conditions on achieved wrist orientation.
    # The incoming covariate accounts for variation in the toss, including orientation at impact.
    design = np.column_stack([np.ones(len(x)), incoming, x])
    regularizer = np.diag([1e-8, 0.02, 0.02, 0.02] + [0.035] * 5)
    coef = np.linalg.solve(design.T @ design + regularizer, design.T @ outgoing)
    _, singular, directions = np.linalg.svd(coef[-5:].T, full_matrices=False)
    return directions, singular


def run(kind):
    OUT.mkdir(parents=True, exist_ok=True)
    arena = stock()
    arena.max_steps = 32
    if kind == "oblique":
        arena.sample_drop = toss
    arena.reset(seed=72182, render=False)
    eyes = stationary(arena)
    rig, player = decoder(), CarriedWristPolicy()
    rng = np.random.default_rng(163827)
    rows = []
    try:
        for i in range(max(DOSES)):
            if kind == "staged":
                speeds = ((0.36, 0.95), (1.1, 2.0), (2.0, 3.2), (3.2, 4.6))[i // 24]
                arena.sample_drop = partial(toss, speeds=speeds)
            seed = int(rng.integers(2**29))
            delta = rng.uniform(-0.117, 0.117, 5)
            rows.append(play(arena, rig, eyes, player, seed, delta))
            if (i + 1) % 12 == 0:
                _write(OUT / f"toss-{kind}.json.gz", dict(rows=rows))
                print(kind, i + 1, sum(r["event"] is not None for r in rows), flush=True)
            if i + 1 in DOSES:
                directions, singular = motor_basis(rows)
                np.savez_compressed(
                    OUT / f"toss-{kind}-{i + 1}.npz", directions=directions, singular=singular
                )
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("ordinary", "oblique", "staged"))
    run(p.parse_args().kind)
