"""Carry the same motor bank and reward learner through a long preassigned random curriculum."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import card, opening_candidates
from .tt38_carry_value import checkpoint

ROOT = Path("artifacts/tt37")
DEST = Path("artifacts/tt39")
PLAN = DEST / "endurance-plan.json"


def plan(feeds):
    sources = [
        ROOT / n
        for n in (
            "opponent-plan.json",
            "memory-course.json",
            "school-plan.json",
            "clinic-learning-plan.json",
            "self-combined.json",
            "motor-value.json.gz",
        )
    ]
    used = set()
    for s in sources:
        d = _read(s)
        used.update(d.get("release", []))
        used.update(d.get("exclusions", []))
        for name in (
            "practice",
            "held_openings",
            "held_replies",
            "replication",
            "repeated",
        ):
            for r in d.get(name, []):
                if isinstance(r, dict):
                    c = r.get("card", r)
                    if "id" in c:
                        used.add(c["id"])
    rng = np.random.default_rng(3904491)
    eligible = [
        s["id"]
        for s in opening_candidates(measured_launches(feeds))
        if s["id"] not in used
    ]
    if len(eligible) < 120:
        raise ValueError(f"Too few independent opening recordings: {len(eligible)}")
    distinct = rng.choice(eligible, 120, replace=False)
    picks = np.r_[distinct[:96], rng.permutation(distinct[:96]), distinct[96:]]
    entries = [
        dict(
            card(int(i), "full", j, int(rng.integers(1, 2**30))),
            profile="reach",
            opening="random",
            reply_seed=int(rng.integers(1, 2**30)),
        )
        for j, i in enumerate(picks)
    ]
    _write(
        PLAN,
        {
            "practice": entries[:192],
            "assessment": entries[192:],
            "schedule": "Two passes over 96 releases with fresh independent first and subsequent targets. The last 24 releases are disjoint from both.",
            "feed_sha256": _hash(feeds),
            "excluded": {s.name: _hash(s) for s in sources},
            "criterion": "100 legal arm returns in one continuously played point. The cohort success fraction includes every assignment. Audits reproduce the action tape, CAD, random requests and physical scoring.",
        },
    )


def school():
    roster = _read(PLAN)["practice"]
    motor = ROOT / "independent-cold-value.json.gz"
    quality = ROOT / "carried-judgement.json.gz"
    for i in range(8):
        tape = DEST / f"endurance-{i + 1:02}.json.gz"
        state = DEST / f"judgement-{i + 1:02}.json.gz"
        done = tape.is_file() and len(_read(tape)["rows"]) == 24
        if not done:
            cmd = [
                sys.executable,
                "-m",
                "scripts.experimental.tt37_motor_clinic",
                "optical-learn",
                str(tape),
                "--plan",
                str(PLAN),
                "--group",
                "practice",
                "--start",
                str(i * 24),
                "--n",
                "24",
                "--samples",
                "35",
                "--rounds",
                "3",
                "--depth",
                "-1.56",
                "--cut",
                ".18",
                "--guided",
                "--refine",
                "--economy",
                "--body",
                "--cap",
                "200",
                "--visual",
                str(ROOT / "rebound.npz"),
                "--rollin",
                str(ROOT / "self-combined.npz"),
                "--learn-value",
                "--memory",
                str(motor),
                "--motor-value",
                str(quality),
            ]
            subprocess.run(cmd, check=True)
        d = _read(tape)
        if d["cards"] != roster[i * 24 : i * 24 + 24] or d["memory_start"] != len(
            _read(motor)["bank"]
        ):
            raise ValueError("Changed course or lost carried motor experience")
        checkpoint(tape, state)
        counts = [r["score"]["learner"] for r in d["rows"]]
        print("completed", i + 1, sum(counts), max(counts), flush=True)
        motor, quality = tape, state
        if max(counts) >= 100:
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["plan", "run"])
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    if a.mode == "plan":
        plan(a.feeds)
    else:
        school()
