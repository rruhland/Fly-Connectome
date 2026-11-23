"""Reapply the setting learned during receiving to the original wrist-only drops."""

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt26_grip_course import Grip, OUT, PRIOR, SETTINGS, bounce


def run():
    trained = _read(OUT / "receiving-speed.json.gz")
    initial = _read(OUT / "grip-source.json.gz")["records"]
    if len(trained["records"]) != 8:
        raise ValueError("Eight complete training streams required")
    angles = np.array([r["angle"] for r in trained["records"]])
    indices = np.array([r["final"]["bounce"] for r in trained["records"]])
    path = OUT / "post-match-grip.npz"
    np.savez_compressed(
        path,
        angles=angles,
        settings=SETTINGS[indices],
        seeds=np.array([r["seed"] for r in trained["records"]]),
        wrist=_hash(BUNDLE),
        match=_hash(PRIOR / "recorded-pose-imitation.npz"),
        training=_hash(OUT / "receiving-speed.json.gz"),
    )
    with np.load(path, allow_pickle=False) as checkpoint:
        post = checkpoint["settings"]
        if checkpoint["wrist"].item() != _hash(BUNDLE) or checkpoint["match"].item() != _hash(
            PRIOR / "recorded-pose-imitation.npz"
        ):
            raise ValueError("Policy head changed before retention replay")
    arena = BounceArena(view="wrist", width=192, height=144, max_steps=360)
    policy = CarriedWristPolicy()
    grip = Grip(arena._model)
    rows = []
    try:
        for start in initial:
            angle = start["angle"]
            grip.set(angle)
            offsets = set(post[angles == angle]) | {start["selected"]}
            cases = []
            for i, held in enumerate(start["held"]):
                results = {
                    str(bias): bounce(
                        arena, policy, held["seed"], bias, "held", ("standard", "changed")[i % 2]
                    )
                    for bias in offsets
                }
                if results[str(start["selected"])] != held["results"][str(start["selected"])]:
                    raise ValueError("Baseline retention replay failed")
                cases.append(dict(seed=held["seed"], results=results))
            rows.append(dict(angle=angle, held=cases))
            print(
                angle, {k: sum(r["results"][str(k)]["streak"] for r in cases) for k in offsets}, flush=True
            )
    finally:
        arena.close()
    _write(OUT / "retention.json.gz", dict(cases=rows, checkpoint=_hash(path)))


if __name__ == "__main__":
    run()
