"""Verify archived solo rewards, optical directions and prefix checkpoints."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_metric_stereo import triangulate
from .tt24_public_body import stationary, stock
from .tt26_bounce_optics import impact_candidates
from .tt26_grip_course import Grip
from .tt26_optical_transfer import decoder
from .tt28_dose import DOSES, MOUNTS, NOMINAL, OUTPUT, SOURCE_SEEDS, SoloLearner, mkplan, release


def verify(label, directory):
    start = perf_counter()
    path = directory / f"dose-solo-{label}.json.gz"
    report = _read(path)
    prior = _read(NOMINAL)
    acceleration = np.asarray(prior["free_flight_acceleration_m_s2"])
    learner = SoloLearner(np.asarray(prior["mounts"]["nominal"]["normal_link"]))
    ckpt_path = directory / f"dose-checkpoints-{label}.npz"
    rng = np.random.default_rng(SOURCE_SEEDS[label] ^ 1761209)
    plan = mkplan(SOURCE_SEEDS[label], DOSES[-1])
    doses = list(map(int, report["checkpoints"]))
    if report["checkpoint_sha256"] != _hash(ckpt_path):
        raise ValueError("Frozen checkpoint file differs")
    if report["inherited_nominal_optics_sha256"] != _hash(NOMINAL):
        raise ValueError("Inherited nominal reference differs")
    arena = stock()
    arena.max_steps = report["practice_cap"]
    grip = Grip(arena._model)
    grip.set(*MOUNTS[label])
    rig = decoder()
    criterion = ("rotation", "selected", "samples", "attempts", "rewards")
    observations = 0
    try:
        arena.reset(seed=plan[0][0], render=False)
        cameras = stationary(arena)
        with np.load(ckpt_path, allow_pickle=False) as checkpoint:
            if checkpoint["doses"].tolist() != doses:
                raise ValueError("Checkpoint count differs")
            for i, row in enumerate(report["rows"]):
                if (row["seed"], row["regime"]) != plan[i] or row["attempt"] != i + 1:
                    raise ValueError("Practice schedule changed")
                arm = learner.select(i, rng)
                actual = release(arena, None, row["seed"], row["regime"], "practice", tape=row["actions"])
                if arm != row["action"] or any(row[k] != v for k, v in actual.items()):
                    raise ValueError(f"Practice tape or action selection differs on {label}/{i + 1}")
                optical = row["optics"]
                rgb_positions = np.asarray(optical["pixels"], dtype=float)
                ball, _, _ = triangulate(cameras, rgb_positions, tolerance=0.018)
                if not np.array_equal(ball, np.asarray(optical["ball"], dtype=float), equal_nan=True):
                    raise ValueError(f"Stereo samples differ on {label}/{i + 1}")
                encoders = np.asarray(optical["joint_angles"], dtype=np.float32)
                impacts = impact_candidates(dict(ball=ball, joint_angles=encoders), acceleration, rig)
                found = impacts[0] if impacts else None
                if found != row["first_impact"]:
                    raise ValueError(f"Optical reversal differs on {label}/{i + 1}")
                observations += int(found is not None)
                learner.update(arm, actual["reward"], None if found is None else found["raw_normal_link"])
                if i + 1 in doses:
                    expected = report["checkpoints"][str(i + 1)]
                    snap = learner.snapshot()
                    if any(snap[k] != expected[k] for k in criterion):
                        raise ValueError(f"Prefix learner state differs on {label}/{i + 1}")
                    if checkpoint["rotation"][doses.index(i + 1)].tolist() != snap["rotation"]:
                        raise ValueError(f"Stored rotation differs on {label}/{i + 1}")
    finally:
        arena.close()
    return dict(source_sha256=_hash(path), checkpoint_sha256=_hash(ckpt_path),
                practice_replays=len(report["rows"]), optical_impacts=observations,
                reconstructed_checkpoints=doses, wall_seconds=perf_counter() - start)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    receipts = {label: verify(label, args.output) for label in MOUNTS}
    _write(args.output / "dose-verification.json", receipts)
    print(receipts, flush=True)
