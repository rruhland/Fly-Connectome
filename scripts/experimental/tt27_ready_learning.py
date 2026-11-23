"""Learn how far to prepare the arm from the preceding point's visible approach and outcome."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_so101 import SO101_HOME
from .tt19_view_full_game import digest_json
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import PublicBody
from .tt26_grip_course import load_match
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT, Player, command, load_bank
from .tt27_motor_active_continuous import ContinuousReceiver
from .tt27_opening_course import Physics

# A negative gain is an extra backswing. Both positive and negative preparatory moves
# have to earn their use from actual game reward on earlier trajectories.
GAINS = np.array([0.0, 0.20, -0.20, 0.48, -0.48, 0.85])
HOME = np.asarray(SO101_HOME)[:5]


def play(arena, shot, ready, player):
    arena.hold(shot["id"])
    body = PublicBody(arena)
    positions = []
    for _ in range(14):
        u = command(ready, body)
        arena.step(u, render=False)
        body.issued(u)
        positions.append(arena.joints[:5])
    arena.release(shot)
    eye = ViewHistory()
    choice = None
    for tick in range(110):
        if tick <= 3:
            k = eye.observe(arena.render_cameras(), arena.joints, body.target)[:14]
            if tick == 3:
                choice = player.choose(k)
        goal = player.templates[choice].goal if choice is not None else ready
        u = command(goal, body)
        _, _, done, cap, _ = arena.step(u, render=False)
        body.issued(u)
        positions.append(arena.joints[:5])
        if done or cap or arena.legal_returns:
            break
    return dict(
        legal=arena.legal_returns,
        contact=arena.contacts,
        choice=choice,
        events=digest_json(arena.events),
        path=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
    )


def group_run(args):
    path, cohort = args
    lookup, _ = load_match(path)
    donor_ids = [r["id"] for r in load_bank()["rows"]]
    rng = np.random.default_rng({"practice": 145512, "held": 28461, "original": 98122}[cohort])
    if cohort == "original":
        donor_ids = [
            r["id"] for r in _read(OUT / "growing-confirmation.json.gz")["rows"] if r["group"] == "nearby"
        ]
    selected = rng.choice(donor_ids, 10, replace=False)
    actor = Player()
    arena = ContinuousReceiver()
    env = Physics(arena)
    history = []
    prev = None
    try:
        for i, sid in enumerate(selected):
            env.set((1.0, 1.3, 0.78, 1.55)[i % 4], 1.0)
            rows = []
            for t in range(9):
                s = draw(lookup, [int(sid)], rng, t)
                results = []
                outcomes = {}
                for g in GAINS:
                    ready = (
                        HOME
                        if prev is None
                        else HOME + np.clip((actor.templates[prev].goal - HOME) * g, -0.80, 0.80)
                    )
                    tag = tuple(ready)
                    if tag not in outcomes:
                        outcomes[tag] = play(arena, s, ready, actor)
                    results.append(outcomes[tag])
                # Verify the first four RGB frames chose the same incoming receive across branches.
                # Different ready poses can change the silhouette. Only common-choice prefixes can be shared.
                if len({r["choice"] for r in results}) != 1:
                    raise ValueError("Prepared arm changed early observation; split the live policies")
                rows.append(dict(t=t, prev=prev, shot=s, results=results))
                prev = results[0]["choice"]
            history.append(dict(donor=int(sid), rows=rows))
        print(
            cohort,
            [sum(r["results"][j]["legal"] for b in history for r in b["rows"]) for j in range(len(GAINS))],
            flush=True,
        )
    finally:
        arena.close()
    return dict(cohort=cohort, blocks=history)


class ReadyMemory:
    def __init__(self, source, local=True):
        self.local = local
        self.data = []
        for b in source:
            for r in b["rows"]:
                for a, t in enumerate(r["results"]):
                    self.data.append((r["prev"], a, t["legal"]))

    def choose(self, previous):
        wins = np.array([1.2] + [1.0] * (len(GAINS) - 1))
        n = np.full(len(GAINS), 3.0)
        for key, a, r in self.data:
            w = 1.0 if not self.local or key == previous else 0.05
            wins[a] += w * r
            n[a] += w
        return int(np.argmax(wins / n))


def run(path):
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        source, held, raw = pool.map(group_run, [(path, g) for g in ("practice", "held", "original")])
    assessment = []
    for target in (held, raw):
        for mode in ("frozen", "previous", "source_only", "adaptive", "reset", "target_only", "pooled"):
            src = [] if mode == "target_only" else source["blocks"]
            model = ReadyMemory(src, mode != "pooled")
            stages = []
            for b in target["blocks"]:
                if mode == "reset":
                    model = ReadyMemory(src)
                steps = []
                for r in b["rows"]:
                    a = 0 if mode == "frozen" else (1 if mode == "previous" else model.choose(r["prev"]))
                    result = r["results"][a]
                    steps.append(dict(action=a, legal=result["legal"], contact=result["contact"]))
                    if mode in ("adaptive", "reset", "target_only", "pooled"):
                        model.data.append((r["prev"], a, result["legal"]))
                stages.append(steps)
            assessment.append(dict(cohort=target["cohort"], mode=mode, stages=stages))
            print(target["cohort"], mode, sum(x["legal"] for step in stages for x in step), flush=True)
    _write(
        OUT / "ready-learning.json.gz",
        dict(gains=GAINS.tolist(), cohorts=[source, held, raw], assessment=assessment),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
