"""Prospective broad-random games for one continuously learning direct joint controller."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt41_early_optics_carry import PRACTICE, resume, save

ROOT = Path("artifacts/tt42")
PLAN = ROOT / "direct-course.json"
SOURCE = Path("artifacts/tt39/endurance-plan.json")
OPTIC = Path("artifacts/tt37/rebound.npz")


def make_plan():
    original = _read(SOURCE)
    rng = np.random.default_rng(4206325)
    rows = original["practice"][:48] + original["assessment"]
    cards = [
        dict(
            c, seed=int(rng.integers(1, 2**30)), reply_seed=int(rng.integers(1, 2**30))
        )
        for c in rows
    ]
    if len({c["id"] for c in cards}) != len(cards) or any(
        c["opening"] != "random" or c["profile"] != "reach" for c in cards
    ):
        raise ValueError("Broad random cards must use distinct releases")
    prior = [mpc.BANK.with_suffix(".json"), PRACTICE.with_suffix(".json")]
    used = {r["card"]["id"] for p in prior for r in _read(p)["practice"]}
    if used.intersection(c["id"] for c in cards):
        raise ValueError("Direct training examples overlap this course")
    _write(
        PLAN,
        {
            "practice": cards[:48],
            "assessment": cards[48:],
            "source_sha256": _hash(SOURCE),
            "rule": "First 48 distinct practice recordings and original 24 disjoint assessment recordings. Fresh independent first and subsequent draws per card. One real physical launch per game. Live bank carries through all 48 games, with assessment separately cloned at checkpoints.",
        },
    )


def run(a):
    cards = _read(PLAN)[a.deck][: a.n]
    ready = _read(mpc.BANK.with_suffix(".json"))["ready"]
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    feeds = mpc.measured_launches(a.feeds)
    receipt = {
        "cards": cards,
        "mode": a.mode,
        "deck": a.deck,
        "ready": ready,
        "cap": getattr(a, "cap", 240),
        "variant": "elastic",
        "budget": 28,
        "source_sha256": _hash(mpc.BANK),
        "previous_practice": {PRACTICE.name: _hash(PRACTICE)},
        "plan_sha256": _hash(PLAN),
        "optical_sha256": _hash(OPTIC),
        "carry_sha256": _hash(a.carry) if a.carry else None,
        "rows": [],
    }
    path = a.out.with_suffix("").with_suffix(".npz")
    document = _read(a.out) if a.resume else receipt.copy()
    if any(document[k] != v for k, v in receipt.items() if k != "rows"):
        raise ValueError("Checkpoint conditions differ")
    with mpc.centered(ready):
        arena = mpc.arena()
        camera, rig, stride, low, high = mpc.resources(arena)
        with patch.object(
            mpc,
            "Fragments",
            lambda file, engine: Carried(
                file, engine, practice=[PRACTICE], held={c["id"] for c in cards}
            ),
        ):
            model = mpc.MPC(mpc.BANK, raw, variant="elastic", budget=28)
        if a.carry:
            base = _read(a.carry)
            if {c["id"] for c in base["cards"][: len(base["rows"])]} & {
                c["id"] for c in cards
            }:
                raise ValueError("Releases overlap carried practice")
            resume(model, a.carry.with_suffix("").with_suffix(".npz"), base)
        if a.resume:
            # The shared restorer verifies serial against the total number of
            # games played. Assessment may start at an earlier course boundary.
            resume(
                model,
                path,
                {**document, "rows": [None] * model.serial + document["rows"]},
            )
        try:
            for index in range(len(document["rows"]), len(cards)):
                c = cards[index]
                r = mpc.play(
                    arena,
                    camera,
                    rig,
                    stride,
                    low,
                    high,
                    ready,
                    c,
                    feeds[c["id"]],
                    raw,
                    model,
                    OPTIC,
                    a.mode,
                    receipt["cap"],
                    20,
                )
                for leg, entry in enumerate(r["placements"]["requested"]):
                    expected, _ = draw(c, leg, "random", None)
                    if list(expected) != entry["target"]:
                        raise RuntimeError(
                            "Opponent did not use the specified independent random draw"
                        )
                document["rows"].append(r)
                document["checkpoint_sha256"] = save(model, path)
                _write(a.out, document)
                if a.deck == "practice" and (index + 1) % 12 == 0:
                    tag = a.out.with_name(
                        f"{a.out.name.split('.')[0]}-{index + 1:02}.json.gz"
                    )
                    tag.with_suffix("").with_suffix(".npz").write_bytes(
                        path.read_bytes()
                    )
                    _write(tag, document)
                print(
                    index + 1,
                    c["id"],
                    r["score"]["learner"],
                    r["memory"]["after"],
                    flush=True,
                )
        finally:
            model.close()
            camera.close()
            arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "live", "frozen"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--deck", choices=("practice", "assessment"), default="practice")
    p.add_argument("--n", default=48, type=int)
    p.add_argument("--cap", default=240, type=int)
    p.add_argument("--carry", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    args = p.parse_args()
    if args.mode == "plan":
        make_plan()
    else:
        run(args)
