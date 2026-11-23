"""Learn a preferred wrist setting through solo rebounds and carry it into receiving."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_so101 import BLADE
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt25_pose_demonstrations import episode
from .tt25_recorded_flight import OUTPUT as PRIOR
from .tt25_reference_receiver import ReferenceReceiver

OUT = Path("artifacts/tt26")
MOUNTS = (-0.22, 0.22)
SETTINGS = np.array([0, -0.12, 0.12, -0.24, 0.24, -0.36, 0.36])


class Grip:
    """Evaluator-side rotation of the physical collision blade about its handle."""

    def __init__(self, model):
        self.blade = model.geom(BLADE)
        self.original = self.blade.quat.copy()

    def set(self, angle, axis=(0.0, 0.0, 1.0)):
        unit = np.asarray(axis, dtype=float)
        unit = unit / np.linalg.norm(unit)
        rotation = np.r_[np.cos(angle / 2), np.sin(angle / 2) * unit]
        result = np.empty(4)
        mujoco.mju_mulQuat(result, rotation, self.original)
        self.blade.quat[:] = result


def bounce(arena, policy, seed, offset, split, regime):
    obs = arena.reset(seed=seed, split=split, regime=regime)
    policy.begin_release(obs)
    intended = np.zeros(5)
    if np.ndim(offset) == 0:
        intended[4] = offset
    else:
        intended[:] = offset
    applied = np.zeros(5)
    total = 0.0
    for _ in range(arena.max_steps):
        action = policy.act(obs)
        adjusted = np.clip(action + (intended - applied) / (5.2 / 60), -1, 1)
        applied += (adjusted - action) * (5.2 / 60)
        action = adjusted
        obs, reward, done = arena.step(action)
        total += reward
        if done:
            break
    metric = arena.judge_metrics()
    return dict(reward=total, streak=metric["longest_consecutive_rebounds"], steps=metric["steps"])


def source():
    arena = BounceArena(view="wrist", width=192, height=144, max_steps=360)
    grip = Grip(arena._model)
    policy = CarriedWristPolicy()
    practice_seeds = np.random.default_rng(233407).integers(1, 2**30, 5)
    test_seeds = np.random.default_rng(744112).integers(1, 2**30, 16)
    records = []
    try:
        for angle in MOUNTS:
            grip.set(angle)
            trials = []
            for i, seed in enumerate(practice_seeds):
                for bias in SETTINGS:
                    result = bounce(
                        arena, policy, int(seed), bias, "practice", ("standard", "changed")[i % 2]
                    )
                    trials.append(dict(seed=int(seed), bias=bias, **result))

            def select(rows):
                return max(SETTINGS, key=lambda b: sum(r["reward"] for r in rows if r["bias"] == b))

            selected = select(trials)
            rng = np.random.default_rng(735187)
            shuffled = []
            for seed in practice_seeds:
                group = [r.copy() for r in trials if r["seed"] == seed]
                for row, bias in zip(group, rng.permutation(SETTINGS), strict=True):
                    row["bias"] = bias
                shuffled.extend(group)
            control = select(shuffled)
            held = []
            for i, seed in enumerate(test_seeds):
                values = {
                    str(b): bounce(arena, policy, int(seed), b, "held", ("standard", "changed")[i % 2])
                    for b in sorted({0.0, selected, control})
                }
                held.append(dict(seed=int(seed), results=values))
            records.append(
                dict(angle=angle, selected=selected, shuffled=control, practice=trials, held=held)
            )
            OUT.mkdir(parents=True, exist_ok=True)
            _write(OUT / "grip-source.json.gz", dict(records=records, candidates=SETTINGS.tolist()))
            print(
                angle,
                selected,
                control,
                {
                    k: sum(r["results"][str(b)]["streak"] for r in held)
                    for k, b in (("learned", selected), ("cold", 0.0), ("shuffled", control))
                },
                flush=True,
            )
    finally:
        arena.close()


def load_match(path):
    raw = path.read_bytes()
    if (
        sha256(raw).hexdigest()
        != json.loads((PRIOR / "recorded-secondary-reference.json").read_text())["sha256"]
    ):
        raise ValueError("Unknown recorded feed archive")
    shots = {r["id"]: r for r in json.loads(raw)}
    with np.load(PRIOR / "recorded-pose-imitation.npz", allow_pickle=False) as archive:
        model = [archive[k] for k in ("image_keys", "joint_goals", "scale")]
    return shots, model


def match(path):
    shots, (keys, goals, scales) = load_match(path)
    donors = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    groups = _read(PRIOR / "recorded-block-confirmation.json.gz")["cohorts"]
    remainder = _read(PRIOR / "recorded-progressive.json.gz")["tests"]
    cohorts = {
        "training": donors,
        "nearby": [r["id"] for g in groups for r in g],
        "confirmation": [r["id"] for r in remainder if r["group"] == "nearby"],
        "unrestricted": [r["id"] for r in remainder if r["group"] == "unfiltered"],
    }
    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    records = []
    try:
        for source_row in _read(OUT / "grip-source.json.gz")["records"]:
            grip.set(source_row["angle"])
            settings = [source_row["selected"], 0.0, source_row["shuffled"]]
            rows = []
            for group, ids in cohorts.items():
                for sid in ids:
                    initial, key = episode(arena, shots[sid], collect=True)
                    if key is None:
                        rows.append(
                            dict(
                                group=group,
                                id=sid,
                                choice=None,
                                results={str(b): initial for b in settings},
                            )
                        )
                        continue
                    choice = np.argmin(np.linalg.norm((keys - key) / scales, axis=1))
                    stats = {}
                    for bias in sorted(set(settings)):
                        goal = goals[choice].copy()
                        goal[4] += bias
                        result, _ = episode(arena, shots[sid], goal)
                        if result["legal"] and result != episode(arena, shots[sid], goal)[0]:
                            raise ValueError("Successful landing changed during physical replay")
                        stats[str(bias)] = result
                    rows.append(dict(group=group, id=sid, choice=int(choice), results=stats))
                print(
                    source_row["angle"],
                    group,
                    [
                        sum(r["results"][str(b)]["legal"] for r in rows if r["group"] == group)
                        for b in settings
                    ],
                    flush=True,
                )
            records.append(dict(angle=source_row["angle"], settings=settings, rows=rows))
            _write(OUT / "grip-match.json.gz", dict(records=records))
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("source", "match"))
    p.add_argument("--feeds", type=Path)
    args = p.parse_args()
    if args.part == "source":
        source()
    else:
        match(args.feeds)
