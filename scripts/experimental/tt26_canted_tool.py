"""Can solo rewards choose a useful position/attitude tradeoff for a canted tool?"""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt24_bounce import BounceArena
from .tt24_integrated_stress import InstalledArm, episode as drop, schedule
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import OUT, PRIOR, Grip, ReferenceReceiver, load_match
from .tt26_optical_bouncing import OpticalWrist
from .tt26_optical_transfer import MOUNTS, decoder, transport
from .tt26_receiving_speed import draw


def run(path):
    scales = (0.008, 0.025, 0.08)
    source = _read(OUT / "optical-face.json.gz")["mounts"]["pitch_plus"]
    rot = np.asarray(source["rotation_from_nominal"])
    station = InstalledArm("stock")
    station.arena.max_steps = 360
    launcher = BounceArena(view="none")
    grip = Grip(station.arena._model)
    grip.set(*MOUNTS["pitch_plus"])
    bounds = station.arena._control_low[:5], station.arena._control_high[:5]
    actors = {s: OpticalWrist(rot, bounds, position_scale=s) for s in scales}
    practice = []
    held = []
    try:
        for split, count, seed in (("practice", 6, 723413), ("held", 12, 419930)):
            if split == "held":
                chosen = max(scales, key=lambda s: sum(r["rewards"][str(s)] for r in practice))
            for p in [r for r in schedule(seed, count) if r["body"] == "stock"]:
                launcher.reset(seed=p["launcher_seed"], split="held", render=False)
                launch = [
                    v.copy()
                    for v in (launcher._ball_position, launcher._ball_velocity, launcher._ball_spin)
                ]
                tested = scales if split == "practice" else sorted({scales[0], chosen})
                scores = {s: drop(station, actors[s], p, launch, "bundle")["metrics"] for s in tested}
                (practice if split == "practice" else held).append(
                    dict(
                        seed=p["learner_seed"],
                        rewards={str(k): v["reward"] for k, v in scores.items()},
                        streaks={str(k): v["longest_consecutive_rebounds"] for k, v in scores.items()},
                    )
                )
        print(
            "selected",
            chosen,
            "held",
            {k: sum(r["streaks"][k] for r in held) for k in held[0]["streaks"]},
            flush=True,
        )
    finally:
        station.arena.close()
        launcher.close()
    arena = ReferenceReceiver()
    Grip(arena.model).set(*MOUNTS["pitch_plus"])
    rig = decoder()
    axes = arena._arm_ids[1][2][:5]
    bounds = arena._joint_low[axes], arena._joint_high[axes]
    shots, (keys, goals, scale) = load_match(path)
    banks = {
        str(s): np.stack([transport(rig, g, rot, bounds, position_scale=s) for g in goals])
        for s in (chosen, scales[0])
    }
    banks["cold"] = goals
    ids = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    rng = np.random.default_rng(823119)
    targets = []
    try:
        for i in range(48):
            shot = draw(shots, ids, rng, i)
            _, key = episode(arena, shot, collect=True)
            j = np.argmin(np.linalg.norm((keys - key) / scale, axis=1))
            values = {k: episode(arena, shot, bank[j])[0] for k, bank in banks.items()}
            targets.append(dict(donor=shot["id"], results=values))
        _write(
            OUT / "canted-tool.json.gz", dict(practice=practice, chosen=chosen, held=held, table=targets)
        )
        print("match", {k: sum(r["results"][k]["legal"] for r in targets) for k in banks}, flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
