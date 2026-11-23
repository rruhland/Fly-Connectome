"""A second nearby cohort and an exhaustive control for the six visual labels."""

import argparse
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_nearby_lessons import features
from .tt25_pose_demonstrations import episode
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def run(path):
    raw = path.read_bytes()
    if (
        sha256(raw).hexdigest()
        != json.loads((OUTPUT / "recorded-secondary-reference.json").read_text())["sha256"]
    ):
        raise ValueError("Unexpected recorded dataset")
    shots = {r["id"]: r for r in json.loads(raw)}
    first = _read(OUTPUT / "recorded-nearby-lessons.json.gz")
    used = {r["id"] for r in json.loads((OUTPUT / "recorded-reach.json").read_text())["cases"]}
    used.update(r["id"] for r in first["cases"])
    donors = first["donors"]
    options = [r for r in shots.values() if r["id"] not in used and r["pos_y"] > 0 and r["vel_y"] < 0]
    vectors = np.stack([features(r) for r in options])
    distant = []
    for donor in donors:
        target = features(shots[donor])
        distance = np.linalg.norm((vectors - target) / first["metric"], axis=1)
        count = 0
        for j in np.argsort(distance):
            row = options[j]
            if row["id"] in used or np.allclose(vectors[j], target, atol=1e-5, rtol=0):
                continue
            used.add(row["id"])
            distant.append(dict(id=row["id"], donor=donor))
            count += 1
            if count == 8:
                break
    with np.load(OUTPUT / "recorded-pose-imitation.npz", allow_pickle=False) as f:
        keys, goals, scale = [f[k] for k in ("image_keys", "joint_goals", "scale")]
    arena = ReferenceReceiver()
    cohorts = []
    summaries = []
    try:
        for group, starts in enumerate([first["cases"], distant]):
            rows = []
            for scene in starts:
                shot = shots[scene["id"]]
                _, key = episode(arena, shot, collect=True)
                choice = int(np.argmin(np.linalg.norm((keys - key) / scale, axis=1)))
                all_scores = [episode(arena, shot, goal)[0] for goal in goals]
                if group == 0 and all_scores[choice] != scene["results"]["camera"]:
                    raise ValueError("Exhaustive actions diverged from live camera test")
                if group == 1:
                    actual, _ = episode(arena, shot, memory=(keys, goals, scale))
                    if actual != all_scores[choice]:
                        raise ValueError("Fresh visual decision diverged from action cross check")
                rows.append(
                    dict(
                        id=scene["id"],
                        donor=scene["donor"],
                        choice=choice,
                        contact=[s["contact"] for s in all_scores],
                        legal=[s["legal"] for s in all_scores],
                    )
                )
            cohorts.append(rows)
            actions = np.array([r["legal"] for r in rows])
            choose = np.array([r["choice"] for r in rows])
            observed = int(actions[np.arange(len(rows)), choose].sum())
            null = [
                int(actions[np.arange(len(rows)), np.array(perm)[choose]].sum())
                for perm in permutations(range(len(goals)))
            ]
            constant = actions.sum(axis=0)
            selected = (
                int(np.argmax(constant)) if group == 0 else summaries[0]["prior_selected_constant_index"]
            )
            summaries.append(
                dict(
                    starts=len(rows),
                    visual_returns=observed,
                    constant_by_goal=constant.tolist(),
                    prior_selected_constant_index=selected,
                    selected_constant_returns=int(constant[selected]),
                    any_goal_legal=int(actions.any(axis=1).sum()),
                    permutation_count=len(null),
                    as_good_or_better=sum(s >= observed for s in null),
                    permutation_mean=float(np.mean(null)),
                    permutation_max=max(null),
                )
            )
        _write(OUTPUT / "recorded-block-confirmation.json.gz", dict(cohorts=cohorts, summaries=summaries))
        print(summaries)
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    run(p.parse_args().deepmind)
