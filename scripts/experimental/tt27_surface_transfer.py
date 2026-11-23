"""Test whether optical bounce experience selects useful motor adjustments on a new table."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT, SOURCE, Player, load_bank
from .tt27_opening_course import Physics
from .tt27_physics_player import FlightPlayer, point

# Signed vertical displacements, expressed in metres in the natural table frame.
HEIGHTS = np.array([0.0, -0.04, 0.04, -0.08, 0.08, -0.13, 0.13])
PRACTICE = [0.54, 0.9, 1.65, 3.3]
NEW = [0.69, 2.6, 1.12, 2.05]


def block(args):
    path, label, number, scale, seed, count = args
    lookup, _ = load_match(path)
    refs = [r["id"] for r in load_bank()["rows"]]
    arena = ReferenceReceiver()
    player = Player()
    observer = FlightPlayer()
    rng = np.random.default_rng(seed)
    rows = []
    try:
        Physics(arena).set(scale, 1.0)
        # One scheduled inspection ball before the scored feeds, common to all.
        # Up to three balls may be needed if the camera never observes the bounce.
        inspection = []
        for t in range(3):
            shot = draw(lookup, refs, rng, t)
            r = point(arena, shot, observer, observer.source, gain=0.0)
            inspection.append(dict(id=shot["id"], legal=r["legal"], estimate=r["measured"]))
            if r["measured"] is not None:
                break
        if label == "raw":
            roster = _read(SOURCE / "fresh-recordings.json.gz")["records"][0]["rows"]
            selected = rng.choice(len(roster), size=count, replace=False)
            feeds = [lookup[roster[i]["id"]] for i in selected]
        else:
            feeds = [draw(lookup, refs, rng, t) for t in range(count)]
        for shot in feeds:
            _, key = episode(arena, shot, collect=True)
            if key is None:
                rows.append(
                    dict(id=shot["id"], origin=None, scores=[0] * len(HEIGHTS), contacts=[0] * len(HEIGHTS))
                )
                continue
            i = player.choose(key)
            policy = player.templates[i]
            direction = policy.matrix[:, 2]
            vals = []
            for h in HEIGHTS:
                goal = policy.goal + np.clip(direction * h, -0.55, 0.55)
                r, _ = episode(arena, shot, goal=goal)
                vals.append([r["legal"], r["contact"]])
            rows.append(
                dict(id=shot["id"], origin=i, scores=[v[0] for v in vals], contacts=[v[1] for v in vals])
            )
        print(
            label,
            number,
            scale,
            [r["estimate"] for r in inspection],
            np.sum([r["scores"] for r in rows], axis=0),
            flush=True,
        )
        return dict(label=label, number=number, damping=scale, seed=seed, inspection=inspection, rows=rows)
    finally:
        arena.close()


def key(block):
    seen = [r["estimate"] for r in block["inspection"] if r["estimate"] is not None]
    return None if not seen else np.array(seen[-1])[:2]


def assess(blocks):
    practice = [r for r in blocks if r["label"] == "practice"]
    # Only practice rewards and the observations collected before play in this world are available.
    # All evaluated branches receive their own chosen scalar reward each time.
    features = [key(b) for b in practice]
    table = np.array([[x["scores"] for x in b["rows"]] for b in practice])
    records = []
    for b in blocks:
        if b["label"] == "practice":
            continue
        measured = key(b)
        ds = np.array(
            [
                np.inf if f is None or measured is None else np.linalg.norm((f - measured) / [0.12, 0.14])
                for f in features
            ]
        )
        w = np.exp(-0.5 * np.minimum(ds, 20) ** 2)
        w = (w + 0.01) / (w + 0.01).sum()
        for blind in ("base", "reset", "pooled", "similar", "reversed"):
            if blind == "reversed":
                weights = w[::-1]
            else:
                weights = w if blind == "similar" else np.ones(len(w)) / len(w)
            mu = np.einsum("b,ba->a", weights, table.mean(axis=1))
            wins = 2 + 8 * mu if blind not in ("reset", "base") else np.ones(len(HEIGHTS))
            counts = np.ones(len(HEIGHTS)) * (12 if blind not in ("reset", "base") else 3)
            seen = []
            for t, r in enumerate(b["rows"]):
                action = 0 if blind == "base" else int(np.argmax(wins / counts))
                reward = r["scores"][action]
                seen.append(dict(action=action, legal=reward, contact=r["contacts"][action]))
                wins[action] += reward
                counts[action] += 1
            records.append(
                dict(
                    label=b["label"],
                    number=b["number"],
                    policy=blind,
                    steps=seen,
                    source_weights=weights.tolist(),
                )
            )
    return records


def run(path, raw):
    jobs = [(path, "practice", i, s, 548011 + i, 32) for i, s in enumerate(PRACTICE)]
    jobs += [(path, "held", i, s, 461552 + i, 32) for i, s in enumerate(NEW)]
    if raw:
        jobs += [(path, "raw", i, s, 297543 + i, 32) for i, s in enumerate(NEW)]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(block, jobs))
    assessment = assess(results)
    _write(
        OUT / "surface-transfer.json.gz",
        dict(heights=HEIGHTS.tolist(), blocks=results, assessment=assessment),
    )
    for label in ("held", "raw"):
        for name in ("base", "reset", "pooled", "similar", "reversed"):
            print(
                label,
                name,
                [
                    sum(r["legal"] for r in b["steps"])
                    for b in assessment
                    if b["label"] == label and b["policy"] == name
                ],
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    run(args.feeds, args.raw)
