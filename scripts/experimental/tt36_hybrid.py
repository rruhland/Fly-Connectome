"""Route individual approaches to staged or broad direct feedback using live camera and encoder input."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time

import numpy as np

from . import tt36_local as local
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, resources, verify
from .tt34_results import paired
from .tt35_replies import varied_opponent
from .tt36_nearby import displaced
from .tt36_reach import BODY_BANK, OUTPUT, PUPIL, Body, Reach, Relation
from .tt36_staged import body_model, choose_from, movement


EARLY = OUTPUT / "local-checkpoint.json"
LATE = OUTPUT / "staged-checkpoint.json"
CHECK = OUTPUT / "hybrid-recognition.json"
CALIBRATION = OUTPUT / "paired-practice-fixed.json.gz"
REACH = OUTPUT / "reach-checkpoint.json"
RELATIONS = OUTPUT / "reach-observations.npz"
PLAN = OUTPUT / "local-plan.json.gz"
MAX_RADIUS = .55
MARGIN = .50


def signature_inputs():
    return {p.name: _hash(p) for p in (EARLY, LATE, BODY_BANK, REACH, RELATIONS, PUPIL)}


def first_signature(arena, renderer, rig, card, row):
    arena.launch(row["shot"], card=card)
    arena.partner.reset(seed=row["opponent_seed"])
    visual = Vision(rig, arena.control_dt, raw=True)
    result = None
    for tick, action in enumerate(row["actions"]):
        q = arena.joints
        optical = visual.update(tick, renderer.capture() if tick % 2 == 0 else None, q)
        if optical is not None and optical[1] < -.22:
            result = np.r_[optical, q[:5]].tolist()
            break
        arena.exchange(action, FAR_COMMAND)
    if digest_json(arena.events) != digest_json(row["events"][:len(arena.events)]):
        raise ValueError("Calibration replay changed the physical opening prefix")
    return result


def calibrate(output):
    prior = _read(EARLY)
    late = _read(LATE)
    stage_centers = [r["context"] for r in prior["observations"]]
    if stage_centers != [r["context"] for r in late["states"]]:
        raise ValueError("The two staged checkpoints have different approach centers")
    centers = np.array(stage_centers)
    scale = np.asarray(prior["scale"])
    raw = _read(PUPIL)
    reference = _read(CALIBRATION)
    if raw["ready"] != reference["ready"] or reference["pupil_sha256"] != _hash(PUPIL):
        raise ValueError("Calibration must use the original frozen raw policy")
    if {item["id"] for item in prior["observations"]}.intersection(c["id"] for c in reference["cards"]):
        raise ValueError("Recognition calibration contains a staged practice shot")
    negatives = []
    with centered(raw["ready"]):
        arena = replay_arena([])
        try:
            renderer, rig, *_ = resources(arena)
            try:
                for card, row in zip(reference["cards"], reference["rows"], strict=True):
                    x = first_signature(arena, renderer, rig, card, row)
                    if x is not None:
                        distance = np.min(np.linalg.norm((centers - x) / scale, axis=1))
                        negatives.append(dict(id=card["id"], x=x, distance=float(distance)))
                        print("calibration", card["id"], round(distance, 3), flush=True)
            finally:
                renderer.close()
        finally:
            arena.close()
    if not negatives:
        raise ValueError("No recognized sensor observations in calibration")
    nearest = min(item["distance"] for item in negatives)
    radius = min(MAX_RADIUS, MARGIN * nearest)
    _write(output, dict(
        schema="tt36-hybrid-1", sources=signature_inputs(), calibration_sha256=_hash(CALIBRATION),
        practice_ids=[r["id"] for r in prior["observations"]], purpose="calibration only, no held data",
        centers=stage_centers, scale=scale.tolist(), radius=radius, cap=MAX_RADIUS, margin=MARGIN,
        unrelated=negatives,
    ))
    print("frozen radius", radius, "closest unrelated", nearest, flush=True)


class Hybrid(Reach):
    """Keep an approach in its selected mode until the original staged observer resets."""

    def __init__(self, body, relation, pupil, rig, authority, recognition, staged_body, early, late):
        super().__init__(body, relation, pupil, rig, authority, 1)
        self.centers = np.asarray(recognition["centers"])
        self.scale = np.asarray(recognition["scale"])
        self.radius = recognition["radius"]
        self.approaches = []
        self.current_route = None
        self.tick = None

        def selection(obs):
            distances = np.linalg.norm((self.centers - obs) / self.scale, axis=1)
            position = int(np.argmin(distances))
            closest = float(distances[position])
            self.current_route = "staged" if closest <= self.radius else "reach"
            self.approaches.append(dict(tick=self.tick, selected=self.current_route,
                                        distance=closest, context=position,
                                        observation=obs.tolist()))
            return early(obs) if self.current_route == "staged" else np.zeros(5)

        # Construct exactly the same approach state machine and late correction as the staged control.
        with movement(staged_body, late):
            self.staged = local.Approach(selection)

    def bias(self, tick, optical, joints, frames):
        self.tick = tick
        command = self.staged.update(optical, joints)
        if self.staged.current is not None and self.current_route == "staged":
            return command
        return super().bias(tick, optical, joints, frames)


def schedule(cohort):
    if cohort in ("new", "confirm", "retained"):
        path = OUTPUT / f"course-{cohort}.json.gz"
        source = _read(path)
        variants = source.get("reply_seeds", [None] * len(source["cards"]))
        return path, [(card, seed, None) for card, seed in zip(source["cards"], variants, strict=True)]
    if cohort == "rehearsed":
        source = _read(PLAN)
        return PLAN, [(case["original"], seed, None)
                      for case in source["cases"] for seed in case["reply_seeds"]]
    path = OUTPUT / "nearby-plan.json"
    source = _read(path)
    return path, [(entry["card"], entry["reply_seed"], entry["delta"]) for entry in source["points"]]


def play(arena, renderer, rig, raw, stride, lo, hi, shot, card, *,
         body, relation, authority, recognition, staged_body, early, late, cap):
    arena.launch(shot, card=card)
    arena.partner.reset(seed=card["seed"])
    pupil = JointPolicy(raw, stride, lo, hi)
    sense = Vision(rig, arena.control_dt, raw=True)
    actor = Hybrid(body, relation, pupil, rig, authority, recognition, staged_body, early, late)
    actions = []
    started = process_time()
    staged_ticks = 0
    for tick in range(arena.max_steps):
        q = arena.joints
        frames = renderer.capture() if tick % 2 == 0 else None
        optical = sense.update(tick, frames, q)
        offset = actor.bias(tick, optical, q, frames)
        if optical is not None and actor.staged.current is not None and actor.current_route == "staged":
            staged_ticks += 1
        action = pupil.command(optical, q, offset=offset)
        _, _, done, truncated, _ = arena.exchange(action, FAR_COMMAND)
        actions.append(action.tolist())
        if done or truncated or arena.legal_returns + arena.rival_returns - 1 >= cap:
            break
    return dict(
        shot=shot, opponent_seed=card["seed"], actions=actions, events=arena.events,
        placements=arena.point_placements(), score=audit(arena.events, opening_side=1),
        routes=actor.approaches, staged_ticks=staged_ticks, reach_ticks=len(actor.active),
        motor_audit=arena.motor_audit(), runtime_cpu_s=process_time() - started,
    )


def run(feeds, cohort, output, state):
    selected = _read(state)
    if selected["sources"] != signature_inputs():
        raise ValueError("Recognition and frozen actuator components differ")
    path, cards = schedule(cohort)
    raw = _read(PUPIL)
    shots = measured_launches(feeds)
    early = local.choose_from(_read(EARLY), "dense")
    late = choose_from(_read(LATE)["states"], "dense")
    staged_body = body_model()
    with np.load(BODY_BANK) as source:
        body = Body(source)
    with np.load(RELATIONS) as source:
        relation = Relation(source)
    authority = _read(REACH)["authority_rad"]
    cap = 16 if cohort in ("rehearsed", "nearby") else 24
    result = dict(
        ready=raw["ready"], opening_side=1, cards=[card for card, _, _ in cards], rows=[],
        cohort=cohort, recognition_sha256=_hash(state), course_sha256=_hash(path),
        cap=cap, schedule=[dict(seed=seed, delta=delta) for _, seed, delta in cards],
    )
    with centered(raw["ready"]):
        arena = replay_arena([])
        renderer, rig, stride, lo, hi = resources(arena)
        try:
            for card, seed, delta in cards:
                with ExitStack() as stack:
                    if delta is not None:
                        stack.enter_context(displaced(card, delta))
                    if seed is not None:
                        stack.enter_context(varied_opponent(card["index"], seed))
                    row = play(arena, renderer, rig, raw, stride, lo, hi, shots[card["id"]], card,
                               body=body, relation=relation, authority=authority, recognition=selected,
                               staged_body=staged_body, early=early, late=late, cap=cap)
                result["rows"].append(row)
                _write(output, result)
                print(cohort, card["id"], seed, row["score"]["learner"], row["score"]["returns"],
                      [r["selected"] for r in row["routes"]], flush=True)
        finally:
            renderer.close()
            arena.close()



CONTROLS = {
    "new": {
        "raw": OUTPUT / "paired-new-fixed.json.gz",
        "reach": OUTPUT / "reach-new-aligned.json.gz",
    },
    "confirm": {
        "raw": OUTPUT / "paired-confirm-fixed.json.gz",
        "reach": OUTPUT / "reach-confirm-aligned.json.gz",
    },
    "rehearsed": {
        "raw": OUTPUT / "local-held-fixed.json.gz",
        "staged": OUTPUT / "staged-held-dense.json.gz",
    },
    "nearby": {
        "raw": OUTPUT / "nearby-fixed.json.gz",
        "staged": OUTPUT / "nearby-both.json.gz",
    },
    "retained": {
        "raw": OUTPUT.parent / "tt35" / "replies-nominal.json.gz",
        "reach": OUTPUT / "reach-retained-aligned.json.gz",
    },
}


def control(path, cohort, used):
    document = _read(path)
    if cohort == "rehearsed":
        replay = [i for i, schedule in enumerate(document["schedule"]) if schedule["kind"] == "reply"]
        document = dict(document, cards=[document["cards"][i] for i in replay],
                        rows=[document["rows"][i] for i in replay])
    if cohort in ("rehearsed", "nearby", "retained"):
        seeds = document.get("reply_seeds")
        if seeds is None and all("reply_seed" in row for row in document["rows"]):
            seeds = [row["reply_seed"] for row in document["rows"]]
        if seeds is not None and seeds != [row["seed"] for row in used["schedule"]]:
            raise ValueError(f"Changed continuation schedule for {path.name}")
    return document


def counts(document):
    first = [next(iter(r["placements"]["delivered"]), None) for r in document["rows"]]
    return dict(
        points=len(document["rows"]), own=sum(r["score"]["learner"] for r in document["rows"]),
        alternating=sum(r["score"]["returns"] for r in document["rows"]),
        later=sum(max(0, r["score"]["learner"] - 1) for r in document["rows"]),
        opening_contacts=sum(x is not None and x["near_contact"] is not None for x in first),
        process_cpu_s=sum(r["runtime_cpu_s"] for r in document["rows"]),
    )


def first_approach_match(hybrid, staged):
    recognized = matches = 0
    for a, b in zip(hybrid["rows"], staged["rows"], strict=True):
        if not a["routes"] or a["routes"][0]["selected"] != "staged":
            continue
        recognized += 1
        cut = min(len(a["actions"]), len(b["actions"]))
        if len(a["routes"]) > 1:
            cut = min(cut, a["routes"][1]["tick"])
        if len(b["picks"]) > 1:
            cut = min(cut, b["picks"][1]["tick"])
        matches += a["actions"][:cut] == b["actions"][:cut]
    return dict(recognized=recognized, exact_commands_through_first_approach=matches)


def summarize(folder):
    recognition = _read(folder / "hybrid-recognition.json")
    result = dict(recognition_sha256=_hash(folder / "hybrid-recognition.json"),
                  radius=recognition["radius"], cohorts={})
    for name, controls in CONTROLS.items():
        source = folder / f"hybrid-{name}.json.gz"
        used = _read(source)
        if (used["recognition_sha256"] != result["recognition_sha256"]
                or len(used["cards"]) != len(used["rows"])):
            raise ValueError("Incomplete or mismatched hybrid evaluation")
        rows = used["rows"]
        total = dict(
            hybrid=counts(used), source_sha256=_hash(source),
            approach_routes={mode: sum(s["selected"] == mode for row in rows for s in row["routes"])
                             for mode in ("staged", "reach")},
            first_routes={mode: sum(bool(r["routes"]) and r["routes"][0]["selected"] == mode for r in rows)
                          for mode in ("staged", "reach")},
            staged_ticks=sum(r["staged_ticks"] for r in rows),
            reach_ticks=sum(r["reach_ticks"] for r in rows),
            max_measured_speed_rad_s=max(max(r["motor_audit"]["max_measured_rate_rad_s"]) for r in rows),
            controls={},
        )
        for label, path in controls.items():
            reference = control(path, name, used)
            total["controls"][label] = dict(
                scores=counts(reference), paired=paired(reference, used), control_sha256=_hash(path),
                exact_full_actions=sum(a["actions"] == b["actions"]
                                       for a, b in zip(used["rows"], reference["rows"], strict=True)),
                exact_full_events=sum(digest_json(a["events"]) == digest_json(b["events"])
                                      for a, b in zip(used["rows"], reference["rows"], strict=True)),
            )
            if label == "staged":
                total["controls"][label]["first_approach"] = first_approach_match(used, reference)
        result["cohorts"][name] = total
        print(name, total["hybrid"], total["approach_routes"],
              {key: v["scores"]["own"] for key, v in total["controls"].items()}, flush=True)
    _write(folder / "hybrid-results.json", result)


def verify_hybrid(folder):
    """Replay the recognized unfamiliar point and scoring rehearsal or displacement cases."""
    verified = []
    hashes = {}
    with TemporaryDirectory(prefix="tt36-hybrid-") as scratch:
        tape = Path(scratch) / "point.json.gz"
        receipt = Path(scratch) / "audit.json"
        for cohort in ("new", "confirm", "rehearsed", "nearby"):
            source = folder / f"hybrid-{cohort}.json.gz"
            hashes[cohort] = _hash(source)
            doc = _read(source)
            for card, row, context in zip(doc["cards"], doc["rows"], doc["schedule"], strict=True):
                if cohort in ("rehearsed", "nearby"):
                    if not row["score"]["learner"]:
                        continue
                elif not any(pick["selected"] == "staged" for pick in row["routes"]):
                    continue
                _write(tape, dict(ready=doc["ready"], opening_side=1, cards=[card], rows=[row]))
                with ExitStack() as stack:
                    if context["delta"] is not None:
                        stack.enter_context(displaced(card, context["delta"]))
                    if context["seed"] is not None:
                        stack.enter_context(varied_opponent(card["index"], context["seed"]))
                    verify(tape, receipt, cad=True)
                point, = _read(receipt)["points"]
                verified.append(dict(point, cohort=cohort, schedule=context))
    _write(folder / "hybrid-audit.json", dict(sources=hashes, points=verified))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    ops = parser.add_subparsers(dest="op", required=True)
    calibration = ops.add_parser("calibrate")
    calibration.add_argument("--output", type=Path, default=CHECK)
    trial = ops.add_parser("run")
    trial.add_argument("feeds", type=Path)
    trial.add_argument("cohort", choices=("new", "confirm", "rehearsed", "nearby", "retained"))
    trial.add_argument("--recognition", type=Path, default=CHECK)
    trial.add_argument("--output", type=Path)
    report = ops.add_parser("summary")
    report.add_argument("--folder", type=Path, default=OUTPUT)
    replay = ops.add_parser("audit")
    replay.add_argument("--folder", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.op == "calibrate":
        calibrate(args.output)
    elif args.op == "run":
        run(args.feeds, args.cohort, args.output or OUTPUT / f"hybrid-{args.cohort}.json.gz", args.recognition)
    elif args.op == "summary":
        summarize(args.folder)
    else:
        verify_hybrid(args.folder)


if __name__ == "__main__":
    main()
