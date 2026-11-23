"""Select a visual reply policy online using the pupil's own legal-return reward."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_trial import COACHED, STEP_SIZE, PublicBody
from .tt26_grip_course import Grip, OUT as PRIOR_OUT, ReferenceReceiver, load_match
from .tt26_match_continuation import archived_step
from .tt27_reply import DEV_DONORS, HELD_DONORS, OUT, Flight, PoseMap, examples, play
from .tt27_reply_recorded import SOURCE as FIRST_REPLICATION, transported

SOURCE = PRIOR_OUT / "match-broader.json.gz"
DESTINATION = OUT / "reply-online.json.gz"
POLICIES = ("track", "once", "feedback", "spatial")
OPPORTUNITIES = 8


def first_block(arena, tape, shot, goal):
    arena.launch(shot)
    tape.steps.clear()
    body = PublicBody(arena)
    rates = []
    for tick in range(arena.max_steps):
        near = np.zeros(arena.action_size)
        if tick >= 3:
            near[:5] = np.clip((goal - body.target[:5]) / STEP_SIZE[:5], -1., 1.)
        _, _, done, cap, _ = arena.step(near, render=False)
        body.issued(near)
        rates.append(near.tolist())
        if done or cap:
            break
    return dict(near=arena.legal_returns, far=max(0, arena.rival_returns - 1), actions=rates,
                rival_actions=tape.tape(len(rates)), events=arena.events.copy())


def course(path):
    shots, _ = load_match(path)
    old = _read(FIRST_REPLICATION)["conditional_roster"]
    forbidden = set(DEV_DONORS) | set(HELD_DONORS) | set(old)
    source = _read(SOURCE)
    practice = [(r["id"], r["example"]["goal"]) for r in source["training"] if r["example"] is not None]
    assessment = next(r for r in source["assessment"] if r["mount"] == "nominal")
    recorded = [(r["id"], assessment["goals"][r["results"]["expanded"]["choice"]]) for r in assessment["rows"]
                if r["results"]["expanded"]["legal"]]
    candidates = [(sid, goal) for sid, goal in recorded + practice if sid not in forbidden]
    memory = [e for e in examples(_read(OUT / "reply-traces.json.gz")["traces"]) if e["id"] in DEV_DONORS]
    arena, check = ReferenceReceiver(stationary_opponent=False), ReferenceReceiver()
    arena.max_steps = check.max_steps = 160
    profile = _read(COACHED)["metadata"]["opening"]["far_profile"]
    install_rally_opponent(arena, target_profile=profile)
    tape = FarTape(arena)
    model, flight = PoseMap(arena), Flight()
    bank = transported(memory, Grip(arena.model), model, 0.)
    counts = np.zeros(len(POLICIES), dtype=int)
    wins = np.zeros(len(POLICIES), dtype=int)
    opened, points = [], []
    try:
        for index, (sid, goal) in enumerate(candidates):
            shot = shots[sid]
            opening = first_block(arena, tape, shot, np.asarray(goal))
            opened.append(dict(id=sid, near=opening["near"], far=opening["far"]))
            if opening["far"]:
                case = dict(angle=0., idx=index, shot=shot, results=dict(bounce=opening))
                # A weak prior favors the development winner on the first return.
                # Later choices use only this learner's own previous legal result.
                prior = np.array([.04, .02, .01, 0.])
                choice = int(np.argmax((1 + wins) / (2 + counts) + prior))
                runs = {}
                for mode in dict.fromkeys(("track", POLICIES[choice])):
                    result = play(arena, tape, case, bank, model, flight, mode, reuse_reply=True)
                    check.launch(shot)
                    for tick, near in enumerate(result["actions"]):
                        archived_step(check, result["rival_actions"], tick, near)
                    if digest_json(result["events"]) != digest_json(check.events):
                        raise ValueError("Online second return did not reproduce")
                    runs[mode] = result
                outcome = int(runs[POLICIES[choice]]["near"] >= 2)
                counts[choice] += 1
                wins[choice] += outcome
                points.append(dict(id=sid, choice=POLICIES[choice], retry=wins.tolist(), trials=counts.tolist(),
                                   adaptive=runs[POLICIES[choice]], frozen=runs["track"]))
                print("opportunity", sid, POLICIES[choice], outcome, runs["track"]["near"] >= 2, flush=True)
            else:
                print("opening", sid, opening["near"], opening["far"], flush=True)
            _write(DESTINATION, dict(source_sha256=_hash(SOURCE), feed_sha256=_hash(path),
                                     target_opportunities=OPPORTUNITIES, opening=opened, points=points))
            if len(points) >= OPPORTUNITIES:
                break
    finally:
        arena.close()
        check.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    course(parser.parse_args().feeds)
