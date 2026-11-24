"""Compare tabletop curriculum order on matched physical SO-101 practice opportunities."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_action_memory import run as play
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import CourseArena, lane, replay_arena


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
OUTPUT = ARTIFACTS / "tt31"
MOVEMENTS = ARTIFACTS / "tt29" / "envelope-carry.json.gz"
ERRORS = ARTIFACTS / "tt30" / "wide-shot-errors.json.gz"
PRACTICE_OPENING = 12330
PRACTICE_COUNTS = {"fixed": 8, "periodic": 10, "continuous": 12}
TEST_POINTS = 10
TEST_CAP = 16


def opening_candidates(shots):
    columns = ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z")
    candidates = [s for s in shots.values() if s["id"] != PRACTICE_OPENING
                  and abs(s["pos_x"]) < .45 and -1.9 < s["pos_y"] < -.55
                  and .15 < s["pos_z"] < .68 and abs(s["vel_x"]) < 1.70
                  and 2.75 < s["vel_y"] < 6.50 and -.40 < s["vel_z"] < 3.]
    center = [0., -1.18, .29, 0., 4.1, .65]
    scale = [.12, .35, .25, .65, 1., 1.]
    return sorted(candidates, key=lambda s: np.linalg.norm((np.array([s[k] for k in columns]) - center) / scale))


def card(shot, mode, index, seed):
    return dict(id=shot, mode=mode, index=index, seed=seed, key=f"{mode}-{index}-{seed}")


def neutral_delivery(arena, shots, entry):
    arena.launch(shots[entry["id"]], card=entry)
    arena.partner.reset(seed=entry["seed"])
    for _ in range(110):
        _, _, done, capped, _ = arena.exchange(np.zeros(arena.action_size), np.zeros(5))
        if done or capped or arena.crossings:
            break
    report = arena.point_placements()
    reply = next(iter(report["delivered"]), None)
    if reply is None or reply["receive_line"] is None:
        return None
    return reply if reply["observed_lane"] == lane(reply["target"][0]) else None


def plan(feeds, output):
    shots = measured_launches(feeds)
    fixed = [card(PRACTICE_OPENING, "fixed", i, 310810 + i) for i in range(PRACTICE_COUNTS["fixed"])]
    periodic = [card(PRACTICE_OPENING, "periodic", i, 311010 + i) for i in range(PRACTICE_COUNTS["periodic"])]
    candidates = [card(PRACTICE_OPENING, "continuous", i, 311210 + i) for i in range(90)]
    held_candidates = [card(s["id"], "continuous", i, 318600 + 19 * i)
                       for i, s in enumerate(opening_candidates(shots))]
    # The neutral arm cannot learn or select a stroke. The screen bounds the
    # number of physical pupil opportunities and verifies both court sides.
    arena = replay_arena([])
    neutral_practice = []
    neutral_held = []
    try:
        for entry in fixed[:1] + periodic[:2]:
            reply = neutral_delivery(arena, shots, entry)
            if reply is None:
                raise ValueError(f"Stable delivery failed for {entry}")
            neutral_practice.append(dict(card=entry, first_reply=reply))
        varied = []
        dose = {"left": 0, "right": 0}
        for entry in candidates:
            reply = neutral_delivery(arena, shots, entry)
            if reply is not None and dose[reply["observed_lane"]] < PRACTICE_COUNTS["continuous"] // 2:
                side = reply["observed_lane"]
                dose[side] += 1
                varied.append(entry)
                neutral_practice.append(dict(card=entry, first_reply=reply))
                print("practice survey", entry["key"], side, flush=True)
            if len(varied) == PRACTICE_COUNTS["continuous"]:
                break
        if len(varied) != PRACTICE_COUNTS["continuous"]:
            raise ValueError(f"Insufficient practice courses, found {dose}")
        held = []
        dose = {"left": 0, "right": 0}
        for entry in held_candidates:
            reply = neutral_delivery(arena, shots, entry)
            if reply is not None and dose[reply["observed_lane"]] < TEST_POINTS // 2:
                side = reply["observed_lane"]
                dose[side] += 1
                held.append(entry)
                neutral_held.append(dict(card=entry, first_reply=reply))
                print("held survey", entry["id"], side, flush=True)
            if len(held) == TEST_POINTS:
                break
        if len(held) != TEST_POINTS:
            raise ValueError(f"Insufficient distinct unseen openings, found {dose}")
    finally:
        arena.close()
    staged = fixed + periodic + varied
    order = np.random.default_rng(3100204).permutation(len(staged))
    _write(output, dict(schema="tt31-course-plan-1", feed_sha256=_hash(feeds),
                        practice_opening=PRACTICE_OPENING, practice_counts=PRACTICE_COUNTS,
                        held_cap=TEST_CAP, staged=staged, mixed=[staged[i]["key"] for i in order],
                        held=held, neutral_practice=neutral_practice, neutral_held=neutral_held))


def checkpoint(report):
    return {key: report[key] for key in ("checkpoint", "optical_outcome_checkpoint")}


def play_card(feeds, entry, movement, errors, temporary, *, practice, cap):
    arenas = []

    def arena_factory(**settings):
        arena = CourseArena(cards=[entry], **settings)
        arenas.append(arena)
        return arena

    play(feeds, temporary, [entry["id"]], "left+right", "lateral",
         checkpoint=movement, optical_checkpoint=errors, arena_factory=arena_factory,
         opening_side=1, opponent_seeds=[entry["seed"]], return_target=cap,
         rated=True, motor_envelope=True, robust_contact=True, rally_objective=True,
         recent=True, blend=True, fast_partner=True, cached_stereo=True, net_rescue=True,
         camera_period=2, optical_outcomes="learn", censored_optical_outcomes=True,
         no_memory=not practice, freeze_optical_outcomes=not practice)
    report = _read(temporary)
    row = report["rows"][0]
    row["placements"] = arenas[0].point_placements()
    for key in ("initial_checkpoint_sha256", "initial_optical_checkpoint_sha256"):
        row[key] = report[key]
    return row, checkpoint(report)


def split_cards(deck, order, phase):
    if phase == "held":
        return deck["held"]
    staged = deck["staged"]
    if order == "staged":
        return staged
    cards = {entry["key"]: entry for entry in staged}
    return [cards[key] for key in deck["mixed"]]


def run(feeds, deck_path, output, order, phase):
    deck = _read(deck_path)
    if _hash(feeds) != deck["feed_sha256"]:
        raise ValueError("Course plan and measured feeds differ")
    cards = split_cards(deck, order, phase)
    practice = phase == "practice"
    cap = 2 if practice else deck["held_cap"]
    movement, errors = MOVEMENTS, ERRORS
    if not practice and order != "original":
        movement = errors = output / f"course-{order}-checkpoint.json.gz"
        training = _read(output / f"course-practice-{order}.json.gz")
        if len(training["rows"]) != len(training["cards"]) or training["checkpoint_sha256"] != _hash(movement):
            raise ValueError(f"Incomplete or mismatched {order} practice checkpoint")
    info = dict(schema="tt31-course-run-1", phase=phase, order=order, opening_side=1,
                feed_sha256=_hash(feeds), plan_sha256=_hash(deck_path), return_target=cap,
                initial_movement_sha256=_hash(movement), initial_error_sha256=_hash(errors),
                cards=cards, rows=[])
    result = output / f"course-{phase}-{order}.json.gz"
    with TemporaryDirectory(prefix="tt31-course-") as scratch:
        temporary = Path(scratch) / "point.json.gz"
        for index, entry in enumerate(cards):
            row, retained = play_card(feeds, entry, movement, errors, temporary, practice=practice, cap=cap)
            info["rows"].append(row)
            if practice:
                movement = errors = output / f"course-{order}-checkpoint.json.gz"
                _write(movement, retained)
                info["checkpoint_sha256"] = _hash(movement)
            row["final_optical_samples"] = len(retained["optical_outcome_checkpoint"]["samples"])
            row["final_censored_samples"] = len(retained["optical_outcome_checkpoint"].get("censored_samples", []))
            _write(result, info)
            print(phase, order, index + 1, len(cards), entry["id"], entry["mode"],
                  [r["observed_lane"] for r in row["placements"]["delivered"]],
                  row["score"]["learner"], row["score"]["returns"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("phase", choices=("plan", "practice", "held"))
    p.add_argument("--order", choices=("staged", "mixed", "original"), default="staged")
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    deck = args.output / "course-plan.json.gz"
    if args.phase == "plan":
        plan(args.feeds, deck)
    else:
        if args.phase == "practice" and args.order == "original":
            p.error("The original checkpoint is evaluated without practice")
        run(args.feeds, deck, args.output, args.order, args.phase)


if __name__ == "__main__":
    main()
