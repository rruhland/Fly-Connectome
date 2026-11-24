"""Play frozen and sensory-updated joint policies against paired physical opponents."""

import argparse
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt34_results import paired
from .tt35_replies import varied_opponent
from .tt36_course import OUT

BASE = joint.JointPolicy


def policy(mode, decisions):
    hit = miss = gate = None
    hashes = {}
    if mode in ("hit", "both", "gate"):
        from .tt36_hit import HitAdjustment

        path = OUT / "hit-checkpoint.json"
        hit = partial(HitAdjustment, _read(path))
        hashes["hit"] = _hash(path)
    if mode in ("miss", "both", "gate"):
        from .tt36_miss import MissCorrection

        miss_path = OUT / "miss-checkpoint.json"
        miss = partial(MissCorrection, _read(miss_path))
        hashes["miss"] = _hash(miss_path)

    if mode in ("effect", "score"):
        from .tt36_effect import Effect

        effect_path = OUT / "effect-checkpoint.json"
        hit = partial(Effect, _read(effect_path), score_only=mode == "score")
        hashes["effect"] = _hash(effect_path)
    if mode == "gate":
        from .tt36_gate import Gate

        gate_file = OUT / "gate-checkpoint.json"
        gate = partial(Gate, _read(gate_file))
        hashes["gate"] = _hash(gate_file)

    class JointWithFeedback(BASE):
        def __init__(self, *args):
            super().__init__(*args)
            self.decisions = []
            decisions.append(self.decisions)
            self.gate = gate() if gate else None
            self.hit = hit() if hit else None
            self.miss = miss() if miss else None
            if self.miss:
                self.miss.begin()

        def command(self, optical, q):
            selected = self.gate.choose(optical, q) if self.gate else mode
            if optical is not None and not self.decisions:
                self.decisions.append(selected)
            changes = []
            if self.hit and selected in ("hit", "both", "effect", "score"):
                changes.append(self.hit.bias(optical, q))
            if self.miss and selected in ("miss", "both"):
                changes.append(self.miss.offset(optical, q))
            changes = [np.asarray(change) for change in changes if change is not None]
            offset = sum(changes) if changes else None
            return super().command(optical, q, offset=offset)

    return JointWithFeedback if hit or miss else BASE, hashes


def run(feeds, mode, cohort):
    source = OUT / f"course-{cohort}.json.gz"
    plan = _read(source)
    decisions = []
    actor, hashes = policy(mode, decisions)
    model, launches = _read(PUPIL), measured_launches(feeds)
    result = {
        "ready": model["ready"],
        "opening_side": 1,
        "cards": plan["cards"],
        "mode": mode,
        "pupil_sha256": _hash(PUPIL),
        "corrections": hashes,
        "course_sha256": _hash(source),
        "rows": [],
        "feed_sha256": _hash(feeds),
    }
    reply_seeds = plan.get("reply_seeds", [None] * len(plan["cards"]))
    with centered(model["ready"]), patch.object(joint, "JointPolicy", actor):
        arena = replay_arena([])
        renderer, rig, stride, lo, hi = joint.resources(arena)
        try:
            for card, seed in zip(plan["cards"], reply_seeds, strict=True):
                context = (
                    varied_opponent(card["index"], seed)
                    if seed is not None
                    else nullcontext()
                )
                with context:
                    arena.cards = iter([card])
                    row = joint.point(
                        arena,
                        renderer,
                        rig,
                        model,
                        stride,
                        lo,
                        hi,
                        launches[card["id"]],
                        card,
                        cap=24,
                        trace=cohort == "practice",
                    )
                if cohort == "practice":
                    trace = row.pop("trace")
                    row["initial_observation"] = next(
                        (
                            item["optical"] + item["q"][:5]
                            for item in trace
                            if item["optical"] is not None
                        ),
                        None,
                    )
                if decisions:
                    row["selected_update"] = decisions[-1]
                result["rows"].append(row)
                _write(OUT / f"paired-{cohort}-{mode}.json.gz", result)
        finally:
            renderer.close()
            arena.close()


def outcomes(rows):
    legs = [p for r in rows for p in r["placements"]["delivered"]]
    next_legs = [p for r in rows for p in r["placements"]["delivered"][1:]]
    return {
        "points": len(rows),
        "opportunities": len(legs),
        "contacts": sum(p["near_contact"] is not None for p in legs),
        "own": sum(p["near_landed"] for p in legs),
        "following_opportunities": len(next_legs),
        "following_legal": sum(p["near_landed"] for p in next_legs),
        "best_own": max(r["score"]["learner"] for r in rows),
        "cpu_seconds": sum(r["runtime_cpu_s"] for r in rows),
    }


def compare():
    reports = {}
    for cohort in ("new", "replies", "confirm"):
        baseline = _read(OUT / f"paired-{cohort}-fixed.json.gz")
        results = {}
        modes = (
            ("fixed", "hit", "gate")
            if cohort == "confirm"
            else ("fixed", "hit", "miss", "both", "gate", "effect", "score")
        )
        for mode in modes:
            path = OUT / f"paired-{cohort}-{mode}.json.gz"
            stream = _read(path)
            if len(stream["rows"]) != len(baseline["cards"]):
                raise ValueError("Incomplete course")
            results[mode] = {
                "sha256": _hash(path),
                "totals": outcomes(stream["rows"]),
                "paired": paired(baseline, stream),
            }
        reports[cohort] = results
    _write(OUT / "paired-results.json", reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("fixed", "hit", "miss", "both", "gate", "effect", "score", "report"),
    )
    parser.add_argument("--cohort", choices=("new", "replies", "practice", "confirm"))
    parser.add_argument("--feeds", type=Path)
    args = parser.parse_args()
    if args.mode == "report":
        compare()
    else:
        run(args.feeds, args.mode, args.cohort)
