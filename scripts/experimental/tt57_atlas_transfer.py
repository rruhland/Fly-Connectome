"""Compare an acquired motor repertory and its ablation from the same live arm checkpoint."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt56_scored_motor_atlas as atlas
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt57_atlas_transfer")
PLAN = ROOT / "schedule.json"
SOURCE = atlas.ROOT / "games.json.gz"
MOVES = atlas.ROOT / "new-movements.npz"
course = atlas.season.course


def plan():
    d = _read(SOURCE)
    if len(d["rows"]) != 16 or d["acquired"] != _hash(MOVES):
        raise ValueError("Incomplete paired initial practice")
    used = {c["id"] for c in d["cards"]}
    used.update(c["id"] for c in _read(atlas.season.CARRY)["cards"][:48])
    with np.load(atlas.MODEL) as f:
        used.update(f["release"].tolist())
    pool = {c["id"]: c for c in _read(atlas.season.PLAN)["practice"]}
    if used & set(pool) or len(pool) != 16:
        raise ValueError("Motor archive and intended held releases overlap")
    rng = np.random.default_rng(57016315)
    cards = [
        dict(
            pool[i],
            seed=int(rng.integers(1, 2**30)),
            reply_seed=int(rng.integers(1, 2**30)),
        )
        for _ in range(4)
        for i in rng.permutation(sorted(pool))
    ]
    ROOT.mkdir(parents=True, exist_ok=True)
    _write(
        PLAN,
        dict(
            practice=cards,
            assessment=cards,
            source=_hash(SOURCE),
            supplement=_hash(MOVES),
            archive=_hash(atlas.MODEL),
            rule="Four shuffled 16-opening blocks with fresh independent continuous opponent draws. Launch the first 32 per condition. The lateral, depth, height, and speed law, physics and body remain the normal TT37 benchmark. Every loss counts. Both clones start with the exact same primary motor bank earned in 16 real games.",
        ),
    )


def run(a):
    schedule = _read(PLAN)
    if (
        _hash(SOURCE) != schedule["source"]
        or _hash(MOVES) != schedule["supplement"]
        or _hash(atlas.MODEL) != schedule["archive"]
    ):
        raise ValueError("Training state changed")
    rank = atlas.season.fresh.shot.Coverage(_read(atlas.season.fresh.RISK))
    who = []

    def make(*args, **kwargs):
        if a.actor == "ablated":
            return atlas.season.Player(*args, coverage=rank, **kwargs)
        p = atlas.Player(*args, coverage=rank, **kwargs)
        with np.load(MOVES) as f:
            for prefix, store in (("earned_", p.earned), ("missed_", p.observed_fails)):
                if a.actor == "acquired":
                    store.extend(
                        zip(f[prefix + "keys"], f[prefix + "moves"], strict=True)
                    )
        who.append(p)
        return p

    restore = course.resume

    def resume(actor, path, d):
        restore(actor, path, dict(d, rows=[None] * 48 + d["rows"]))

    a.out = ROOT / (a.actor + ".json.gz")
    with (
        atlas.enabled(),
        patch.object(course, "PLAN", PLAN),
        patch.object(course, "resume", resume),
        patch.object(atlas.mpc, "MPC", make),
    ):
        course.run(a)
    d = _read(a.out)
    if who:
        p = who[-1]
        out = ROOT / (a.actor + "-new.npz")
        np.savez_compressed(
            out,
            **{
                label + suffix: np.array([row[i] for row in bank])
                for label, bank in (
                    ("earned_", p.earned),
                    ("missed_", p.observed_fails),
                )
                for i, suffix in enumerate(("keys", "moves"))
            },
        )
        d["acquired"] = _hash(out)
    d["actor"] = a.actor
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("actor", choices=("plan", "acquired", "ablated", "archive_only"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(carry=SOURCE, mode="live", deck="practice", cap=240, resume=False)
    a = p.parse_args()
    plan() if a.actor == "plan" else run(a)
