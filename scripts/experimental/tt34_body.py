"""Learn a near-arm transmission from encoder consequences, then hold it fixed in play."""

import argparse
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_contacts import record
from .tt31_curriculum import card, neutral_delivery, opening_candidates
from .tt31_curriculum_opponent import replay_arena
from .tt31_curriculum_summary import compare_delivery, first_deliveries
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, resources


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
OUTPUT = ARTIFACTS / "tt34"
CHECKPOINT = ARTIFACTS / "tt33" / "joint-raw.json.gz"
PRIOR_COURSE = ARTIFACTS / "tt32" / "fresh-plan.json.gz"
CHANGE = np.array([.80, 1.20, .80, 1.22, .83, 1.])
STOCK = np.ones(6)
PULSE_TICKS = 9
IDLE_TICKS = 27
TRAIN_AMPLITUDES = [[s] * 5 for s in (.026, .037, .045)]
TEST_AMPLITUDES = [[.032, -.032, .033, -.028, .030],
                   [-.030, -.029, .027, .034, -.034]]
PROBE_CARD = card(12330, "fixed", 0, 341003)


def plan(feeds, output, *, count=16, seed=341127, exclude_ids=()):
    """Select new openings by neutral delivery only, before evaluating either learner."""
    shots = measured_launches(feeds)
    memory = _read(CHECKPOINT)
    exclude = set(memory["training_release_ids"]).union(exclude_ids)
    archives = [PRIOR_COURSE, ARTIFACTS / "tt31" / "course-plan.json.gz",
                ARTIFACTS / "tt31" / "recovery-full.json.gz",
                ARTIFACTS / "tt31" / "recovery-periodic.json.gz",
                ARTIFACTS / "tt31" / "confirm-recovery.json.gz"]
    for source in archives:
        archived = _read(source)
        for field in ("cards", "held", "staged"):
            exclude.update(entry["id"] for entry in archived.get(field, []))
    candidates = [shot for shot in opening_candidates(shots) if shot["id"] not in exclude][:300]
    prng = np.random.default_rng(seed)
    counts = Counter()
    quota = dict(left=3 * count // 8, middle=count // 4, right=3 * count // 8)
    selections = []
    with centered(memory["ready"]):
        arena = replay_arena([])
        try:
            for i, index in enumerate(prng.permutation(len(candidates))):
                trial = card(candidates[index]["id"], "full", i, seed + i)
                opening = neutral_delivery(arena, shots, trial)
                side = opening["observed_lane"] if opening is not None else None
                if side is not None and counts[side] < quota[side]:
                    selections.append(dict(card=trial, delivery=opening))
                    counts[side] += 1
                    if counts == quota:
                        break
        finally:
            arena.close()
    if sum(counts.values()) != count:
        raise ValueError(f"Could not complete the predeclared roster: {counts}")
    _write(output, dict(cards=[item["card"] for item in selections], first_deliveries=selections,
                        exclude_ids=sorted(exclude), counts=dict(counts), feed_sha256=_hash(feeds)))


class BodyMemory:
    """Diagonal local response estimated from commanded displacement and observed travel."""

    def __init__(self, response):
        self.response = np.asarray(response)

    @classmethod
    def fit(cls, pairs):
        requested = np.array([row["requested_delta"][:5] for row in pairs])
        observed = np.array([row["observed_delta"][:5] for row in pairs])
        gain = np.sum(requested * observed, axis=0) / np.sum(requested**2, axis=0)
        return cls(np.r_[gain, 1.])

    def predict(self, nominal_change):
        return self.response * nominal_change

    def compensate(self, issued, nominal):
        return np.clip(issued * (nominal.response / self.response), -1., 1.)


def pulse(arena, shot, stride, nominal_displacement):
    """Use a safe probe before the far arm has returned the ball."""
    command = nominal_displacement / (PULSE_TICKS * stride)
    arena.launch(shot, card=PROBE_CARD)
    arena.partner.reset(seed=PROBE_CARD["seed"])
    before = arena.joints
    history = []
    for tick in range(PULSE_TICKS + IDLE_TICKS):
        _, _, end, capped, _ = arena.exchange(command if tick < PULSE_TICKS else np.zeros(6), FAR_COMMAND)
        if end or capped:
            raise ValueError("Encoder probe ended early")
        history.append(arena.joints)
    if arena.contacts:
        raise ValueError("Ball arrived during the encoder probe")
    measured = np.mean(history[-5:], axis=0) - before
    return dict(command=command.tolist(), initial=before.tolist(), encoders=np.asarray(history).tolist(),
                expected_displacement=nominal_displacement.tolist(), displacement=measured.tolist())


def sense(arena, shot, stride, amplitudes):
    """Symmetric excitation cancels stationary servo and gravity offsets."""
    pairs = []
    for value in amplitudes:
        displacement = np.r_[value, 0.]
        pos = pulse(arena, shot, stride, displacement)
        neg = pulse(arena, shot, stride, -displacement)
        observed = np.asarray(pos["displacement"]) - neg["displacement"]
        pairs.append(dict(requested_delta=(2 * displacement).tolist(), observed_delta=observed.tolist(),
                          plus=pos, minus=neg))
    return pairs


def errors(memory, pairs):
    delta = np.array([row["requested_delta"] for row in pairs])
    observed = np.array([row["observed_delta"] for row in pairs])
    return float(np.sqrt(np.mean(np.square(memory.predict(delta)[:, :5] - observed[:, :5]))))


def drive_profile(arena, nominal_increment, multipliers):
    """Evaluator-only change to the control increments applied by the plant."""
    arena._joint_increments[arena._arm_ids[1][2]] = nominal_increment * multipliers


def calibration(feeds, output):
    shots = measured_launches(feeds)
    raw = _read(CHECKPOINT)
    calibrations = []
    memories = []
    with centered(raw["ready"]):
        arena = replay_arena([])
        axes = arena._arm_ids[1][2]
        increments = arena._joint_increments[axes].copy()
        stride = increments * (arena.control_dt / arena.model.opt.timestep)
        try:
            for name, gains in [("stock", STOCK), ("asymmetric", CHANGE)]:
                drive_profile(arena, increments, gains)
                pairs = sense(arena, shots[PROBE_CARD["id"]], stride, TRAIN_AMPLITUDES)
                memory = BodyMemory.fit(pairs)
                memories.append(memory)
                calibrations.append(dict(condition=name, evaluator_gain=gains.tolist(),
                                         fitted_response=memory.response.tolist(), training=pairs))

            validation = []
            for name, gains in [("asymmetric", CHANGE), ("stock_restored", STOCK)]:
                drive_profile(arena, increments, gains)
                test = sense(arena, shots[PROBE_CARD["id"]], stride, TEST_AMPLITUDES)
                identify = test[:1]
                validate = test[1:]
                rank = [errors(model, identify) for model in memories]
                selected = int(np.argmin(rank))
                residuals = [errors(model, validate) for model in memories]
                validation.append(dict(condition=name, chosen_memory=selected, probe=test,
                                       identification_rms_rad=rank, held_probe_rms_rad=residuals))
        finally:
            arena.close()
    _write(output, dict(schema="tt34-body-1", checkpoint_sha256=_hash(CHECKPOINT),
                        calibration_card=PROBE_CARD, calibrations=calibrations, validation=validation))
    print("multipliers from encoders", (memories[1].response / memories[0].response).round(4), flush=True)
    for row in validation:
        print(row["condition"], "chosen", row["chosen_memory"], "held RMS", row["held_probe_rms_rad"], flush=True)


def point(arena, stereo, rig, model, stride, lo, hi, shot, entry, nominal, body):
    arena.launch(shot, card=entry)
    arena.partner.reset(seed=entry["seed"])
    policy = JointPolicy(model, stride, lo, hi)
    vision = Vision(rig, arena.control_dt, raw=True)
    actions, pupil_actions, states = [], [], []
    start = process_time()
    for tick in range(arena.max_steps):
        q = arena.joints
        optical = vision.update(tick, stereo.capture() if tick % 2 == 0 else None, q)
        issued = policy.command(optical, q)
        applied = issued if body is None else body.compensate(issued, nominal)
        _, _, done, capped, _ = arena.exchange(applied, FAR_COMMAND)
        actions.append(applied.tolist())
        pupil_actions.append(issued.tolist())
        states.append(q.tolist())
        if done or capped or arena.legal_returns + arena.rival_returns - 1 >= 16:
            break
    motor = np.asarray(actions)
    prior = np.asarray(pupil_actions)
    return dict(shot=shot, opponent_seed=entry["seed"], score=audit(arena.events, opening_side=1),
                placements=arena.point_placements(), events=arena.events, actions=actions,
                raw_pupil_actions=pupil_actions, encoder_trajectory=states, motor_audit=arena.motor_audit(),
                active_steps=int(np.count_nonzero(np.max(abs(motor), axis=1) > .01)),
                rescaled_steps=int(np.count_nonzero(np.max(abs(motor - prior), axis=1) > 1e-8)),
                saturated_commands=int(np.count_nonzero(np.max(abs(motor), axis=1) >= 1.)),
                runtime_cpu_s=process_time() - start)


def trial(feeds, plan_path, calibration_path, output, mode):
    doc = _read(plan_path)
    model = _read(CHECKPOINT)
    evidence = _read(calibration_path)
    if evidence["checkpoint_sha256"] != _hash(CHECKPOINT) or not model["options"]["features"].startswith("raw"):
        raise ValueError("Body calibration requires its matching raw-input policy")
    if doc["feed_sha256"] != _hash(feeds):
        raise ValueError("The course and measured feeds disagree")
    memory = [BodyMemory(item["fitted_response"]) for item in evidence["calibrations"]]
    gains = CHANGE if mode in {"changed", "adapted"} else STOCK
    shots = measured_launches(feeds)
    report = dict(mode=mode, ready=model["ready"], opening_side=1, rows=[], cards=doc["cards"],
                  evaluator_gain=gains.tolist(), checkpoint_sha256=_hash(CHECKPOINT),
                  course_sha256=_hash(plan_path), calibration_sha256=_hash(calibration_path))
    with centered(model["ready"]):
        arena = replay_arena([])
        increments = arena._joint_increments[arena._arm_ids[1][2]].copy()
        stereo, rig, stride, lo, hi = resources(arena)
        try:
            drive_profile(arena, increments, gains)
            body = None
            if mode in {"adapted", "restored"}:
                probe = sense(arena, shots[PROBE_CARD["id"]], stride, TEST_AMPLITUDES[:1])
                losses = [errors(candidate, probe) for candidate in memory]
                selection = int(np.argmin(losses))
                body = memory[selection]
                report["identification"] = dict(memory_index=selection, rms_rad=losses, probe=probe)
            for entry in doc["cards"]:
                row = point(arena, stereo, rig, model, stride, lo, hi, shots[entry["id"]], entry, memory[0], body)
                report["rows"].append(row)
                _write(output, report)
                print(mode, entry["id"], row["score"]["learner"], row["score"]["returns"], flush=True)
        finally:
            stereo.close()
            arena.close()


def replay(source_path, output, *, cad=False):
    source = _read(source_path)
    rows = []
    with centered(source["ready"]):
        arena = replay_arena([])
        drive_profile(arena, arena._joint_increments[arena._arm_ids[1][2]].copy(), source["evaluator_gain"])
        full_body = None
        try:
            if cad:
                from .tt29_arm_clearance import ArmClearance
                from .tt29_rally_physics import mujoco

                full_body = replay_arena([], detail="cad")
                guard = ArmClearance(full_body.model, tolerance=0.)
                shadow = mujoco.MjData(full_body.model)
            for entry, row in zip(source["cards"], source["rows"], strict=True):
                arena.launch(row["shot"], card=entry)
                arena.partner.reset(seed=entry["seed"])
                if cad:
                    arena.sample_clearance(True)
                    arena._cad_reader = guard, shadow
                contacts = record(arena, row, opening_side=1)
                after = arena.point_placements()
                if digest_json(after) != digest_json(row["placements"]):
                    raise ValueError("Command-only arm replay changed deliveries")
                clearance = arena.clearance_result() if cad else None
                if cad and clearance["samples"] != round(arena.data.time / arena.model.opt.timestep):
                    raise ValueError("Full-CAD scan missed a physics step")
                rows.append(dict(card=entry, events_sha256=digest_json(arena.events), contacts=contacts,
                                 learner=row["score"]["learner"], returns=row["score"]["returns"],
                                 clearance=clearance))
        finally:
            arena.close()
            if full_body is not None:
                full_body.close()
    _write(output, dict(source_sha256=_hash(source_path), rows=rows))


def trajectory_errors(stock, alternatives):
    """Compare encoders over the same active ticks before any arm disturbs the ball."""
    chunks = {name: [] for name in alternatives}
    for i, baseline in enumerate(stock):
        rows = {name: trial[i] for name, trial in alternatives.items()}
        rows["stock"] = baseline
        traces = {name: np.asarray(row["encoder_trajectory"])[:, :5] for name, row in rows.items()}
        endpoint = min(
            min(len(trace) for trace in traces.values()),
            min((e["step"] for row in rows.values() for e in row["events"]
                 if e["kind"] == "contact" and e["side"] == 1), default=np.inf),
        )
        active = np.flatnonzero(np.max(abs(np.asarray(baseline["actions"])), axis=1) > .01)
        if len(active) and active[0] < endpoint:
            region = slice(active[0], endpoint)
            for name, group in chunks.items():
                group.append(traces["stock"][region] - traces[name][region])
    result = {}
    for name, group in chunks.items():
        data = np.concatenate(group)
        result[name] = dict(ticks=len(data), rms_rad=float(np.sqrt(np.mean(data**2))))
    return result


def summarize(directory, output):
    prior_path = ARTIFACTS / "tt33" / "joint-fresh-raw.json.gz"
    before = _read(prior_path)
    modes = {name: _read(directory / f"body-novel-{name}.json.gz")
             for name in ("stock", "changed", "adapted")}
    modes["restored"] = _read(directory / "body-retained.json.gz")
    for name, doc in modes.items():
        if len(doc["cards"]) != len(doc["rows"]):
            raise ValueError(f"Incomplete {name} assay")
    retained = modes["restored"]
    if before["cards"] != retained["cards"]:
        raise ValueError("Prior retention cards changed")
    for original, new in zip(before["rows"], retained["rows"], strict=True):
        for field in ("actions", "events", "placements", "score"):
            if digest_json(original[field]) != digest_json(new[field]):
                raise ValueError(f"Stock retention did not reproduce {field}")
    bodies = {}
    results = {}
    stock = modes["stock"]["rows"]
    for name, doc in modes.items():
        rows = doc["rows"]
        bodies[name] = compare_delivery(first_deliveries(stock), first_deliveries(rows)) if name != "restored" else None
        results[name] = dict(
            points=len(rows), near_legal=sum(r["score"]["learner"] for r in rows),
            alternating=sum(r["score"]["returns"] for r in rows),
            contacts=sum(p["near_contact"] is not None for r in rows for p in r["placements"]["delivered"]),
            max_measured_rate_rad_s=max(max(r["motor_audit"]["max_measured_rate_rad_s"]) for r in rows),
            active_steps=sum(r["active_steps"] for r in rows), rescaled_steps=sum(r["rescaled_steps"] for r in rows),
            ticks=sum(len(r["actions"]) for r in rows),
            saturated_commands=sum(r["saturated_commands"] for r in rows),
            cpu_s=sum(r["runtime_cpu_s"] for r in rows),
        )
    moved = trajectory_errors(stock, {name: modes[name]["rows"] for name in ("changed", "adapted")})
    _write(output, dict(scores=results, matched_first_deliveries=bodies, encoder_divergence_from_stock=moved,
                        reproduced_prior_points=len(before["rows"]), prior_sha256=_hash(prior_path)))
    print(results, moved, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    roster = actions.add_parser("plan")
    roster.add_argument("feeds", type=Path)
    roster.add_argument("output", type=Path)
    roster.add_argument("--count", type=int, choices=(8, 16, 24), default=16)
    roster.add_argument("--seed", type=int, default=341127)
    roster.add_argument("--exclude-ids", type=int, nargs="*", default=[])
    calibrate = actions.add_parser("calibrate")
    calibrate.add_argument("feeds", type=Path)
    calibrate.add_argument("output", type=Path)
    run = actions.add_parser("trial")
    for field in ("feeds", "plan", "calibration", "output"):
        run.add_argument(field, type=Path)
    run.add_argument("mode", choices=("stock", "changed", "adapted", "restored"))
    verify = actions.add_parser("replay")
    verify.add_argument("source", type=Path)
    verify.add_argument("output", type=Path)
    verify.add_argument("--cad", action="store_true")
    report = actions.add_parser("summarize")
    report.add_argument("output", type=Path)
    report.add_argument("--directory", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "plan":
        plan(args.feeds, args.output, count=args.count, seed=args.seed, exclude_ids=args.exclude_ids)
    elif args.action == "calibrate":
        calibration(args.feeds, args.output)
    elif args.action == "trial":
        trial(args.feeds, args.plan, args.calibration, args.output, args.mode)
    elif args.action == "replay":
        replay(args.source, args.output, cad=args.cad)
    else:
        summarize(args.directory, args.output)


if __name__ == "__main__":
    main()
