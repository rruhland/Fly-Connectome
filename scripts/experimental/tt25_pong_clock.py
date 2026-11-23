"""Does a delay compensation learned in native Pong help recorded-shot receiving?"""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_pong_temporal_wrist import SOURCE, advance
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import PublicBody
from .tt25_pose_demonstrations import OBSERVE, episode
from .tt25_reference_receiver import ReferenceReceiver
from .tt25_recorded_flight import OUTPUT


def run(path):
    shots = {r["id"]: r for r in json.loads(path.read_bytes())}
    conditions = _read(OUTPUT / "recorded-block-confirmation.json.gz")
    source = _read(SOURCE)
    with np.load(OUTPUT / "recorded-pose-imitation.npz", allow_pickle=False) as f:
        keys, goals, scale = [f[name] for name in ("image_keys", "joint_goals", "scale")]
    arena = ReferenceReceiver()
    rows = []
    try:
        for group, cohort in enumerate(conditions["cohorts"]):
            for entry in cohort:
                shot = shots[entry["id"]]
                arena.launch(shot)
                body, eye = PublicBody(arena), ViewHistory()
                captured = []
                for tick in range(OBSERVE + 1):
                    captured.append(
                        eye.observe(arena.render_cameras(), arena.joints, body.target)[:14].copy()
                    )
                    if tick < OBSERVE:
                        zero = np.zeros(arena.action_size)
                        arena.step(zero, render=False)
                        body.issued(zero)
                        eye.issued(zero)
                # One or two frames of latency are scheduled independently of any outcome.
                lag = 1 + int(entry["id"] % 2)
                decisions = {}
                for name, gain in source["gains"].items():
                    key = captured[OBSERVE - lag].copy()
                    for e in (0, 7):
                        key[e : e + 2] = advance(key[e : e + 2], key[e + 2 : e + 4] * 30, lag / 30.0, gain)
                    decisions[name] = [
                        int(np.argmin(np.linalg.norm((keys[:n] - key) / scale, axis=1))) for n in (1, 3, 6)
                    ]
                prompt = int(np.argmin(np.linalg.norm((keys - captured[-1]) / scale, axis=1)))
                if prompt != entry["choice"]:
                    raise ValueError("Prompt policy changed from the verified held receipt")
                # Confirm each action chosen under latency once per real shot in the physical referee.
                chosen = set(j for modes in decisions.values() for j in modes)
                for j in chosen:
                    outcome, _ = episode(arena, shot, goals[j])
                    if outcome["legal"] != entry["legal"][j] or outcome["contact"] != entry["contact"][j]:
                        raise ValueError("A delayed choice disagrees with its physical trial")
                rows.append(
                    dict(
                        id=shot["id"],
                        cohort=group,
                        lag=lag,
                        decisions=decisions,
                        legal={k: [entry["legal"][j] for j in v] for k, v in decisions.items()},
                        prompt_legal=entry["legal"][prompt],
                    )
                )
        totals = {k: np.sum([r["legal"][k] for r in rows], axis=0).tolist() for k in source["gains"]}
        _write(
            OUTPUT / "recorded-pong-clock.json.gz",
            dict(
                source=SOURCE.name,
                source_sha256=_hash(SOURCE),
                native_points=source["source_practice_points"],
                gains=source["gains"],
                demo_counts=[1, 3, 6],
                starts=rows,
                totals=totals,
            ),
        )
        print(totals, flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    run(p.parse_args().deepmind)
