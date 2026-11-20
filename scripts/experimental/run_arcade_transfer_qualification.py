"""Run the sealed paired comparison on a frozen Breakout candidate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import random
from statistics import mean
import subprocess

from arcade_transfer_schedule import (
    CHECKPOINT_SHA256,
    FIRST_OPPORTUNITIES,
    SUSTAINED_RETURNS,
    build_schedule,
    canonical_json,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "artifacts/m1b/2026-11-19-pong-after-mixed-course.json"
SEALS = ROOT / "artifacts/m2/2026-11-20-qualification-seals.json"
ARMS = ("full", "motor_arrival", "frozen", "frozen_orientation", "fresh", "fresh_visual", "passive")
CARRIED_ARMS = frozenset(("full", "motor_arrival", "frozen", "frozen_orientation"))


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _git(*args):
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def verify_schedule(path, task):
    content = path.read_bytes()
    digest = _sha(content)
    expected = json.loads(SEALS.read_bytes())["schedules"][task]
    if digest != expected:
        raise ValueError(f"{task} schedule does not match its seal: {digest}")
    schedule = json.loads(content)
    regenerated = canonical_json(build_schedule(task, bytes.fromhex(schedule["key"])))
    if content != regenerated:
        raise ValueError("schedule differs from the precommitted generator")
    return schedule, digest


def _first_slots(contacts):
    slots = contacts[:FIRST_OPPORTUNITIES]
    return slots + [None] * (FIRST_OPPORTUNITIES - len(slots))


def _returns(contacts):
    return sum(contact is not None and contact["returned"] for contact in contacts)


def score_course(lives, max_life_ticks):
    """Keep every earlier life in elapsed time and reset rally length at each life."""
    elapsed_ticks = elapsed_frames = 0
    contacts = []
    first_return = sustained = None
    for life in lives:
        run_length = 0
        for index, contact in enumerate(life["contacts"], 1):
            event = dict(contact, life=life["id"], opportunity=index,
                         total_ticks=elapsed_ticks + contact["tick"],
                         total_frames=elapsed_frames + contact["frame"])
            contacts.append(event)
            run_length = run_length + 1 if event["returned"] else 0
            if run_length and first_return is None:
                first_return = event
            if run_length == SUSTAINED_RETURNS and sustained is None:
                sustained = event
        elapsed_ticks += life["ticks"]
        elapsed_frames += life["frames"]
    return {
        "first_opportunity": contacts[0] if contacts else None,
        "first_five": _first_slots(contacts),
        "first_return": first_return,
        "sustained": sustained,
        "observed_ticks": elapsed_ticks,
        "observed_frames": elapsed_frames,
        "capped_sustained_ticks": sustained["total_ticks"] if sustained else len(lives) * max_life_ticks,
    }


def _slot_summary(rows, key):
    slots = [row[key] for row in rows]
    return [{"returns": _returns(column), "reached": sum(event is not None for event in column),
             "scheduled": len(rows)} for column in zip(*slots)]


def _ratio_summary(rows):
    return {
        "lives": len(rows),
        "returns": sum(row["own_returns"] for row in rows),
        "opportunities": sum(row["own_opportunities"] for row in rows),
        "losses": sum(row["life_lost"] for row in rows),
        "clears": sum(row["cleared"] for row in rows),
        "bricks": sum(row["score"] for row in rows),
        "post_return_bricks": sum(row["post_own_bricks"] for row in rows),
        "first_five_per_life": _slot_summary(rows, "first_five"),
    }


def summarize_courses(courses):
    lives = [life for course in courses for life in course["lives"]]
    cold = [course["first_opportunity"] for course in courses]
    first_returns = [course["first_return"] for course in courses if course["first_return"] is not None]
    sustained = [course["sustained"] for course in courses if course["sustained"] is not None]
    return {
        **_ratio_summary(lives),
        "courses": len(courses),
        "cold_first": {"returns": _returns(cold), "reached": sum(event is not None for event in cold),
                       "scheduled": len(courses)},
        "first_five_across_lives": _slot_summary(courses, "first_five"),
        "courses_with_first_return": len(first_returns),
        "mean_observed_frames_to_first_return": mean(event["total_frames"] for event in first_returns)
                                                if first_returns else None,
        "courses_with_sustained_rally": len(sustained),
        "mean_capped_sustained_ticks": mean(course["capped_sustained_ticks"] for course in courses),
        "mean_observed_ticks_to_sustained": mean(event["total_ticks"] for event in sustained) if sustained else None,
        "mean_observed_frames_to_sustained": mean(event["total_frames"] for event in sustained) if sustained else None,
        "by_life_ordinal": [_ratio_summary(list(life_rows)) for life_rows in zip(*(c["lives"] for c in courses))],
        "by_condition": {
            condition: _ratio_summary([life for life in lives if life["condition"] == condition])
            for condition in sorted({life["condition"] for life in lives})
        },
    }


def _paired_difference(differences):
    estimate = mean(differences)
    rng = random.Random(82631728)
    resamples = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(8192))
    signs = itertools.product((-1, 1), repeat=len(differences))
    extreme = sum(abs(mean(value * sign for value, sign in zip(differences, pattern))) >= abs(estimate) - 1e-12
                  for pattern in signs)
    return {
        "mean": estimate,
        "paired_course_bootstrap_95": [resamples[204], resamples[7987]],
        "two_sided_sign_flip_p": extreme / (2 ** len(differences)),
        "courses_ahead": sum(value > 0 for value in differences),
        "courses_tied": sum(value == 0 for value in differences),
        "courses_behind": sum(value < 0 for value in differences),
    }


def paired_comparison(challenger, reference):
    pairs = list(zip(challenger, reference, strict=True))
    if any(a["id"] != b["id"] for a, b in pairs):
        raise ValueError("course identifiers differ across paired arms")
    first_five = [(_returns(a["first_five"]) - _returns(b["first_five"])) / FIRST_OPPORTUNITIES for a, b in pairs]
    time_saved = [b["capped_sustained_ticks"] - a["capped_sustained_ticks"] for a, b in pairs]
    attainment = [(a["sustained"] is not None) - (b["sustained"] is not None) for a, b in pairs]
    return {
        "course_ids": [a["id"] for a, _ in pairs],
        "first_five_fraction_gain": dict(_paired_difference(first_five), by_course=first_five),
        "capped_ticks_saved_to_sustained": dict(_paired_difference(time_saved), by_course=time_saved),
        "sustained_courses_gained": sum(attainment),
    }


def _summarize(rows):
    summary = {}
    for cohort in ("ordinary", "stress"):
        data = {arm: [course for course in rows[arm] if course["cohort"] == cohort] for arm in ARMS}
        comparisons = {
            f"{a}_vs_{b}": paired_comparison(data[a], data[b])
            for a, b in (("full", "fresh"), ("full", "passive"), ("motor_arrival", "fresh"),
                         ("motor_arrival", "passive"), ("full", "motor_arrival"), ("full", "frozen"))
        }
        summary[cohort] = {"by_arm": {arm: summarize_courses(courses) for arm, courses in data.items()},
                           "paired": comparisons}
    ordinary = summary["ordinary"]["paired"]
    full_fresh = ordinary["full_vs_fresh"]
    full_passive = ordinary["full_vs_passive"]
    half = len(full_fresh["course_ids"]) / 2
    summary["rapid_transfer_criteria"] = {
        "first_five_over_fresh": full_fresh["first_five_fraction_gain"]["mean"] > 0,
        "first_five_over_passive": full_passive["first_five_fraction_gain"]["mean"] > 0,
        "majority_of_courses_over_fresh": full_fresh["first_five_fraction_gain"]["courses_ahead"] > half,
        "majority_of_courses_over_passive": full_passive["first_five_fraction_gain"]["courses_ahead"] > half,
        "sustained_as_reliably_as_fresh": full_fresh["sustained_courses_gained"] >= 0,
        "sustained_sooner_than_fresh": full_fresh["capped_ticks_saved_to_sustained"]["mean"] > 0,
    }
    summary["rapid_transfer_criteria"]["all"] = all(summary["rapid_transfer_criteria"].values())
    return summary


def _controller_state(session):
    state = session.state()
    # Snapshot value content in case the returned mapping references live objects.
    content = canonical_json(state)
    return json.loads(content), _sha(content)


def _play_life(controller, arena, passive_request):
    if controller is not None:
        controller.reset(arena.frame())
    requests = bytearray()
    while not arena.needs_reset:
        request = controller.choose(baseline_action=1)[0] if controller is not None else passive_request
        requests.append(request)
        transition = arena.step(request)
        if controller is not None:
            controller.observe(replace(transition, hits=0))
    raw = arena.summary()
    contacts = [dict(contact, frame=contact["camera"], returned=bool(contact["success"]))
                for contact in raw["contacts"]]
    if len(contacts) != raw["own_opportunities"] or _returns(contacts) != raw["own_returns"]:
        raise ValueError("arena opportunity annotations do not agree with its summary")
    if any(not 1 <= c["tick"] <= raw["ticks"] or not 1 <= c["frame"] <= raw["cameras"] for c in contacts):
        raise ValueError("opportunity lies outside its life")
    if raw["life_lost"] and (not contacts or contacts[-1]["returned"]):
        raise ValueError("terminal loss must include its missed opportunity")
    if not raw["terminated"] and not raw["truncated"]:
        raise ValueError("life has neither terminated nor reached the horizon")
    return dict(raw, contacts=contacts, frames=raw["cameras"], first_five=_first_slots(contacts),
                action_sha256=_sha(requests))


def _run_course(arm, course, schedule, checkpoint, output):
    import torch
    from arcade_transfer_session import TransferSession
    from breakout_arena import BreakoutCamera

    torch.set_num_threads(1)
    session = TransferSession(arm, course["policy_seed"], checkpoint if arm in CARRIED_ARMS else None)
    initial_state, initial_sha = _controller_state(session)
    previous_sha = initial_sha
    lives = []
    for config in course["lives"]:
        condition = schedule["conditions"][config["condition"]]
        arena = BreakoutCamera(
            config["environment_seed"], layout_seed=config["layout_seed"], serve_side=config["serve_side"],
            layout_split=schedule["layout_split"], distractors=schedule["distractors"],
            max_episode_ticks=schedule["max_life_ticks"], **condition,
        )
        controller = session.begin_life()
        row = _play_life(controller if arm != "passive" else None, arena, condition["mapping"].index(1))
        session.end_life()
        _, final_sha = _controller_state(session)
        lives.append(dict(row, **config, initial_state_sha256=previous_sha, final_state_sha256=final_sha))
        previous_sha = final_sha

    final_state, final_sha = _controller_state(session)
    row = {"id": course["id"], "arm": arm, "cohort": course["cohort"],
           "policy_seed": course["policy_seed"], "lives": lives,
           "initial_state": initial_state, "initial_state_sha256": initial_sha,
           "final_state": final_state, "final_state_sha256": final_sha,
           **score_course(lives, schedule["max_life_ticks"])}
    (output / f"{arm}-{course['id']}.json").write_bytes(canonical_json(row))
    return row


def _audit_pairs(rows):
    for course_group in zip(*(rows[arm] for arm in ARMS), strict=True):
        if len({(course["id"], course["policy_seed"]) for course in course_group}) != 1:
            raise ValueError("controller arms did not play the same course and policy seed")
        for life_group in zip(*(course["lives"] for course in course_group), strict=True):
            keys = ("id", "environment_seed", "layout_seed", "serve_side", "condition", "initial_bricks")
            if len({tuple(life[key] for key in keys) for life in life_group}) != 1:
                raise ValueError("controller arms did not play the same life")
            if any(not life["contacts"] for life in life_group):
                raise ValueError("a scheduled descending serve did not reach its first incoming plane")
            first = [(life["contacts"][0]["tick"], life["contacts"][0]["ball_x"]) for life in life_group]
            if len(set(first)) != 1:
                raise ValueError("paired initial trajectory differed before the first incoming opportunity")
            passive = life_group[ARMS.index("passive")]
            opening = passive["contacts"][0]
            if opening["paddle_x"] != .5 or opening["success"] or opening["offset"] * passive["serve_side"] <= 0:
                raise ValueError("the passive first opportunity did not require movement in the scheduled direction")


def _candidate_receipt(candidate):
    frozen_commit = _git("rev-parse", f"{candidate}^{{commit}}")
    _git("merge-base", "--is-ancestor", frozen_commit, "HEAD")
    roots = ("scripts/experimental", "src")
    if _git("status", "--porcelain", "--untracked-files=all", "--", *roots):
        raise ValueError("commit all candidate source files before evaluation")
    if _git("diff", "--name-only", frozen_commit, "--", *roots):
        raise ValueError("current source differs from the frozen candidate")
    paths = _git("ls-tree", "-r", "--name-only", frozen_commit, "--", *roots).splitlines()
    hashes = {path: _sha((ROOT / path).read_bytes()) for path in paths}
    return {"candidate_commit": frozen_commit, "candidate_tree": _git("rev-parse", f"{frozen_commit}^{{tree}}"),
            "evaluation_commit": _git("rev-parse", "HEAD"), "source_sha256": hashes}


def run(schedule_path, output, candidate, workers):
    schedule, schedule_sha = verify_schedule(schedule_path, "breakout")
    checkpoint_content = CHECKPOINT.read_bytes()
    checkpoint_sha = _sha(checkpoint_content)
    if schedule["checkpoint_sha256"] != checkpoint_sha or checkpoint_sha != CHECKPOINT_SHA256:
        raise ValueError("Pong checkpoint does not match the precommitted byte digest")
    checkpoint = json.loads(checkpoint_content)
    receipt = dict(_candidate_receipt(candidate), schedule_sha256=schedule_sha, checkpoint_sha256=checkpoint_sha,
                   checkpoint_components_sha256={key: _sha(canonical_json(checkpoint[key])) for key in
                                                   ("motor", "timing", "interception", "score", "shot_memory")},
                   arms=ARMS)
    output.mkdir(parents=True, exist_ok=False)
    (output / "receipt.json").write_bytes(canonical_json(receipt))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {arm: [pool.submit(_run_course, arm, course, schedule, checkpoint, output)
                      for course in schedule["courses"]] for arm in ARMS}
        rows = {arm: [job.result() for job in jobs[arm]] for arm in ARMS}
    _audit_pairs(rows)
    summary = _summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["rapid_transfer_criteria"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", required=True, help="frozen candidate commit before schedule reveal")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.schedule, args.output, args.candidate, args.workers)


if __name__ == "__main__":
    main()
