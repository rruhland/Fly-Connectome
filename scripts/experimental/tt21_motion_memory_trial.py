"""Score two cheap visual delay estimators and play fresh strict games at unseen delays."""

import argparse
import json
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt20_camera_carry import SETTINGS
from .tt20_curriculum import OUTPUT as FACTORY
from .tt20_curriculum import configuration, model_arena
from .tt20_fixture import TT20Arena
from .tt20_latency import CommandPipe
from .tt20_natural import NaturalViewTransport
from .tt21_game_delay import GameDelayPlayer
from .tt21_game_delay_trial import (
    STEP_SCHEDULE,
    play_point,
    replay_game,
    summarize_game,
)
from .tt21_motion_memory import MotorTraceMemory, OnsetDelay
from .tt21_recovery_trial import OUTPUT

GAME_SEED = 473918123
LEVELS = [(0, 1), (4, 5), (9, 2)]


def historical():
    path = OUTPUT / "game-delay.json.gz"
    scores = {}
    for eye, games in _read(path)["results"].items():
        for pair in games:
            records = pair["games"]["adaptive"]["points"]
            game = scores.setdefault(eye, {}).setdefault(str(pair["seed"]), {})
            for name, estimator in [("memory", MotorTraceMemory()), ("onset", OnsetDelay())]:
                reports = []
                begin = process_time()
                for row in records:
                    reports.append(estimator.observe(np.asarray(row["actions"]), row["measured"]))
                game[name] = {
                    "reports": reports,
                    "cpu_s": process_time() - begin,
                    "correct": sum(r["evaluator_delay"] == o["delay"] for r, o in zip(records, reports)),
                    "points": len(reports),
                }
    _write(
        OUTPUT / "motion-memory-history.json.gz",
        {"source": path.name, "source_sha256": _hash(path), "scores": scores},
    )


def play(eye, original=None):
    game_seed = GAME_SEED if original is None else original["seed"]
    schedule = LEVELS if original is None else STEP_SCHEDULE
    programs, catalogue, law, response, candidate = configuration(FACTORY)
    player = GameDelayPlayer(programs, catalogue, law, response, model_arena())
    arena = TT20Arena(**SETTINGS, view_mode=eye)
    data = {"eye": eye, "seed": game_seed, "schedule": schedule, "arena": SETTINGS, "conditions": {}}
    path = OUTPUT / f"motion-onset-{eye}-game.json.gz"
    branches = [("onset", OnsetDelay())]
    if original is None:
        branches.insert(0, ("fixed", None))
    else:
        path = OUTPUT / f"motion-onset-{eye}-reproduction-{game_seed}.json.gz"
    try:
        for condition, estimator in branches:
            score = ServiceScore(first_server=-1)
            rows = []
            rng = np.random.default_rng(game_seed)
            player.lead = 0
            transport = NaturalViewTransport(eye, lens_fov={eye: arena.lens_intrinsics()[eye]}, law=law)
            while not score.finished and len(rows) < 80:
                delay = next(d for i, d in reversed(schedule) if len(rows) >= i)
                seed = int(rng.integers(2**31))
                row = play_point(
                    arena,
                    player,
                    transport,
                    eye,
                    seed,
                    score.server,
                    candidate,
                    CommandPipe(delay),
                    estimator,
                )
                score.record(row["outcome"])
                rows.append(dict(row, score=list(score.score), evaluator_delay=delay))
            game = {
                "branch": "adaptive" if estimator is not None else "frozen",
                "points": rows,
                "summary": summarize_game(rows),
                "score": list(score.score),
            }
            game["replayed"] = replay_game(arena, game, schedule)
            if original is not None:
                fields = (
                    "seed",
                    "role",
                    "actions",
                    "applied",
                    "lead",
                    "optical",
                    "rgb_sha256",
                    "events",
                    "outcome",
                )
                expected = original["games"]["adaptive"]["points"]
                for old, new in zip(expected, rows, strict=True):
                    if any(
                        json.dumps(old[k], sort_keys=True) != json.dumps(new[k], sort_keys=True)
                        for k in fields
                    ):
                        raise ValueError(f"Policy diverged from original game on {new['seed']}")
                data["verified_original_points"] = len(rows)
            data["conditions"][condition] = game
            _write(path, data)
            print(
                eye,
                condition,
                game["score"],
                [(r["evaluator_delay"], r["lead"], r["learning"] and r["learning"]["delay"]) for r in rows],
                flush=True,
            )
    finally:
        player.player.close()
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eye", choices=["left", "wrist"])
    parser.add_argument("--reproduce-wrist", action="store_true")
    args = parser.parse_args()
    if args.reproduce_wrist:
        for pair in _read(OUTPUT / "game-delay.json.gz")["results"]["wrist"]:
            play("wrist", original=pair)
    elif args.eye:
        play(args.eye)
    else:
        historical()
