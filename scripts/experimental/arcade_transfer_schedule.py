"""Generate the committed evaluator-only qualification schedules from their revealed keys."""
import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "arcade-transfer-qualification-v1"
CHECKPOINT_SHA256 = "d09f1050ad2e5f04fbe12572d506b55cc30bb765db040720607633076c8ecb0e"
MAX_LIFE_TICKS = 1200
FIRST_OPPORTUNITIES = 5
SUSTAINED_RETURNS = 3
BREAKOUT_CONDITIONS = {
    "ordinary": {"mapping": [0, 1, 2], "delay_frames": 0, "activation_seconds": None},
    "remapped": {"mapping": [1, 2, 0], "delay_frames": 0, "activation_seconds": None},
    "delayed": {"mapping": [0, 1, 2], "delay_frames": 8, "activation_seconds": 0.15},
    "both": {"mapping": [1, 2, 0], "delay_frames": 8, "activation_seconds": 0.15},
}


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _priority(key, label):
    return hashlib.sha256(key + label.encode()).digest()


def _seed(key, label):
    return 1 + int.from_bytes(_priority(key, label)[:8], "big") % 2147483646


def _breakout(key):
    courses = []
    groups = [("ordinary", 8, {"ordinary": 8}), (
        "stress", 4, {"remapped": 2, "delayed": 2, "both": 2},
    )]
    for cohort, course_count, starts_per_side in groups:
        for course_number in range(1, course_count + 1):
            course_id = f"{cohort}-{course_number:02}"
            assignments = [
                (condition, side, repeat)
                for condition, count in starts_per_side.items()
                for side in (-1, 1)
                for repeat in range(count)
            ]
            assignments.sort(key=lambda case: _priority(key, f"order:{course_id}:{case}"))
            # Balance both launch directions in the first (fully cold) life.
            opening_side = -1 if course_number % 2 else 1
            opening = next(index for index, (_, side, _) in enumerate(assignments) if side == opening_side)
            assignments.insert(0, assignments.pop(opening))
            lives = []
            for index, (condition, side, _) in enumerate(assignments, 1):
                life_id = f"{course_id}-{index:02}"
                lives.append({
                    "id": life_id,
                    "environment_seed": _seed(key, f"environment:{life_id}"),
                    "layout_seed": _seed(key, f"layout:{life_id}"),
                    "serve_side": side,
                    "condition": condition,
                })
            courses.append({"id": course_id, "cohort": cohort,
                            "policy_seed": _seed(key, f"policy:{course_id}"), "lives": lives})
    return {"courses": courses, "conditions": BREAKOUT_CONDITIONS,
            "layout_split": "heldout", "distractors": True, "max_life_ticks": MAX_LIFE_TICKS}


def _rotated_pong(key):
    starts = []
    counts = {(longitudinal, lateral): 0 for longitudinal in (-1, 1) for lateral in (-1, 1)}
    used = set()
    index = 0
    while min(counts.values()) < 6:
        index += 1
        seed = _seed(key, f"environment:{index}")
        initial_rng = (seed * 48271 + 1) % 2147483647
        direction = (1 if initial_rng % 2 == 0 else -1,
                     1 if initial_rng > 2147483647 // 2 else -1)
        if seed in used or counts[direction] == 6:
            continue
        counts[direction] += 1
        used.add(seed)
        starts.append({"environment_seed": seed, "distractor_seed": _seed(key, f"distractors:{index}"),
                       "policy_seed": _seed(key, f"policy:{index}")})
    return {"rotations": [0, 1, 2, 3], "starts": starts}


def build_schedule(task, key):
    if len(key) != 32:
        raise ValueError("qualification key must contain 32 bytes")
    generators = {"breakout": _breakout, "rotated-pong": _rotated_pong}
    schedule = generators[task](key)
    return {"schema": SCHEMA, "task": task, "checkpoint_sha256": CHECKPOINT_SHA256,
            "key": key.hex(), **schedule}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("breakout", "rotated-pong"))
    parser.add_argument("key_file", type=Path, help="file containing the 64 hexadecimal key digits")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_bytes(canonical_json(build_schedule(args.task, bytes.fromhex(args.key_file.read_text()))))


if __name__ == "__main__":
    main()
