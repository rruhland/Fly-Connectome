"""Infer a changed command lag during solo practice and export timing for the next task."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce_ident import _ellipse
from .tt24_public_body import stock
from .tt29_bounce import WarmWrist, load_checkpoint


def features(q, commands, lag):
    q = np.asarray(q)[:, :5]
    cmd = np.asarray(commands)[: len(q) - 1, :5]
    delayed = np.concatenate((np.zeros((lag, 5)), cmd), axis=0)[: len(cmd)]
    nominal_target = q[0] + np.cumsum(delayed * (5.2 / 60.0), axis=0)
    v = np.diff(q, axis=0)
    features = np.stack(
        (
            v[1:-1],
            v[:-2],
            delayed[2:] * 0.085,
            delayed[1:-1] * 0.085,
            nominal_target[2:] - q[2:-1],
            np.sin(q[2:-1]),
            np.cos(q[2:-1]),
            np.ones_like(v[2:]),
        ),
        axis=-1,
    )
    return features, v[2:]


def nominal_model(rows):
    selected = [
        features(r["optics"]["enc"], r["actions"], 0)
        for r in rows
        if r["regime"] == "standard"
    ]
    x = np.concatenate([r[0] for r in selected])
    y = np.concatenate([r[1] for r in selected])
    return np.stack(
        [
            np.linalg.solve(
                x[:, j].T @ x[:, j] + np.eye(x.shape[-1]) * 0.0002, x[:, j].T @ y[:, j]
            )
            for j in range(5)
        ]
    )


def errors(weights, q, commands):
    result = []
    for lag in range(5):
        x, y = features(q, commands, lag)
        prediction = np.einsum("tjf,jf->tj", x, weights)
        result.append(float(np.mean(np.minimum((prediction - y) ** 2, 0.0008))))
    return result


def run(output):
    root = Path("artifacts/tt29")
    history = root / "tt29_bounce_course.json.gz"
    weights = nominal_model(_read(history)["practice"])
    checkpoint = load_checkpoint(root / "tt29_bounce_retained.json")
    arena = stock(fixture="table-edge")
    rows = []
    try:
        for hidden in (2, 0, 3, 1):
            for seed in (2909917, 2909918):
                obs = arena.reset(seed=seed, split="held", regime="standard")
                policy = WarmWrist(
                    checkpoint, (arena._control_low[:5], arena._control_high[:5])
                )
                policy.begin_release(obs)
                rng = np.random.default_rng(seed + 90)
                queue = deque(np.zeros((hidden, 5)))
                encoders = [obs["joints"].tolist()]
                commands = []
                actuated = []
                for _ in range(arena.max_steps):
                    command = policy.act(obs, _ellipse(obs["rgb"]))
                    commands.append(command.tolist())
                    queue.append(command.copy())
                    actual = np.clip(queue.popleft() + rng.normal(0, 0.025, 5), -1, 1)
                    actuated.append(actual.tolist())
                    obs, _, done = arena.step(actual)
                    encoders.append(obs["joints"].tolist())
                    if done:
                        break
                length = min(30, len(encoders))
                losses = errors(weights, encoders[:length], commands)
                row = dict(
                    seed=seed,
                    injected_delay=hidden,
                    inferred_delay=int(np.argmin(losses)),
                    ticks=length - 1,
                    loss=losses,
                    rebounds=arena.judge_metrics()["rebounds"],
                    issued=commands,
                    applied=actuated,
                    encoders=encoders,
                )
                rows.append(row)
                print(hidden, seed, row["inferred_delay"], row["rebounds"], flush=True)
    finally:
        arena.close()
    # The first source condition was specified before the table comparison.
    # It exports only the lag minimizing predictive error from the two wrist-game sequences.
    source = rows[:2]
    _write(
        output,
        dict(
            training_sha256=_hash(history),
            rows=rows,
            carried_lag=int(np.argmin(np.mean([r["loss"] for r in source], axis=0))),
        ),
    )


def verify(output):
    """Replay the original physical commands independently of visual inference."""
    archive = _read(output)
    arena = stock(fixture="table-edge")
    try:
        for row in archive["rows"]:
            obs = arena.reset(
                seed=row["seed"], split="held", regime="standard", render=False
            )
            encoders = [obs["joints"].tolist()]
            for actual in row["applied"]:
                obs, _, _ = arena.step(actual, render=False)
                encoders.append(obs["joints"].tolist())
            if (
                encoders != row["encoders"]
                or arena.judge_metrics()["rebounds"] != row["rebounds"]
            ):
                raise ValueError("Solo encoder or rebound trajectory changed on replay")
    finally:
        arena.close()
    archive["verified_encoder_tapes"] = len(archive["rows"])
    _write(output, archive)
    print("verified", archive["verified_encoder_tapes"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        run(args.output)
    verify(args.output)
