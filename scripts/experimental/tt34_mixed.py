"""Test independently learned camera and actuator corrections under their combined shifts."""

import argparse
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_curriculum_summary import compare_delivery, first_deliveries
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, resources
from .tt34_body import (
    CHANGE, CHECKPOINT, OUTPUT, PROBE_CARD, TEST_AMPLITUDES,
    BodyMemory, drive_profile, errors, replay, sense,
)
from .tt34_latency import Map


MODES = ("frozen", "clock", "both")
IMAGE_DELAY = 3
FRAME_PERIOD = 2


def point(arena, renderer, rig, model, stride, lower, upper, shot, entry, *,
          forward, lag, nominal, body):
    arena.launch(shot, card=entry)
    arena.partner.reset(seed=entry["seed"])
    pupil = JointPolicy(model, stride, lower, upper)
    eye = Vision(rig, arena.control_dt, raw=True)
    queue = deque()
    actions, original = [], []
    started = process_time()
    compensated_ticks = 0
    for tick in range(arena.max_steps):
        encoders = arena.joints
        delivered = None
        if tick % FRAME_PERIOD == 0:
            queue.append(renderer.capture())
            if len(queue) > IMAGE_DELAY:
                delivered = queue.popleft()
        # Only the evaluator sees the queue length or capture time.
        sensed = eye.update(tick, delivered, encoders)
        if sensed is not None and lag:
            sensed = sensed + forward(sensed).reshape(-1, 6)[lag]
            compensated_ticks += 1
        requested = pupil.command(sensed, encoders)
        applied = requested if body is None else body.compensate(requested, nominal)
        _, _, done, capped, _ = arena.exchange(applied, FAR_COMMAND)
        original.append(requested.tolist())
        actions.append(applied.tolist())
        if done or capped or arena.legal_returns + arena.rival_returns - 1 >= 16:
            break
    commands = np.asarray(actions)
    return dict(
        shot=shot, opponent_seed=entry["seed"], actions=actions, raw_pupil_actions=original,
        events=arena.events, placements=arena.point_placements(), score=audit(arena.events, opening_side=1),
        motor_audit=arena.motor_audit(), runtime_cpu_s=process_time() - started,
        clock_compensated_ticks=compensated_ticks,
        active_ticks=int(np.count_nonzero(np.max(abs(commands), axis=1) > .01)),
        rescaled_ticks=int(np.count_nonzero(np.max(abs(commands - original), axis=1) > 1e-8)),
    )


def assess(feeds, course, output_dir, modes=MODES):
    shots = measured_launches(feeds)
    plan = _read(course)
    if plan["feed_sha256"] != _hash(feeds):
        raise ValueError("Course and measured launch file differ")
    model = _read(CHECKPOINT)
    bank_path = OUTPUT / "lag-bank.npz"
    age_path = OUTPUT / "lag-fit.json.gz"
    calibration_path = OUTPUT / "body-calibration.json.gz"
    camera = _read(age_path)
    body = _read(calibration_path)
    model_hash = _hash(CHECKPOINT)
    if camera["pupil_sha256"] != model_hash or body["checkpoint_sha256"] != model_hash:
        raise ValueError("Calibration and direct pupil checksums differ")
    if camera["bank_sha256"] != _hash(bank_path):
        raise ValueError("Camera age and observation model checksums differ")
    with np.load(bank_path) as bank:
        motion = bank["next"] - bank["flight"][:, None, :]
        forward = Map(bank["flight"], motion.reshape(len(motion), -1),
                      np.maximum(bank["flight"].std(axis=0), .05))
    learned_age = int(camera["lag"])
    if not 0 <= learned_age < motion.shape[1]:
        raise ValueError("Learned age exceeds the forward model")
    memories = [BodyMemory(item["fitted_response"]) for item in body["calibrations"]]
    common = dict(
        ready=model["ready"], opening_side=1, cards=plan["cards"],
        course_sha256=_hash(course), pupil_sha256=model_hash, evaluator_gain=CHANGE.tolist(),
        evaluator_camera_delay_exposures=IMAGE_DELAY, bank_sha256=_hash(bank_path),
        clock_fit_sha256=_hash(age_path), body_fit_sha256=_hash(calibration_path),
    )
    with centered(model["ready"]):
        arena = replay_arena([])
        try:
            renderer, rig, stride, lower, upper = resources(arena)
            axes = arena._arm_ids[1][2]
            drive_profile(arena, arena._joint_increments[axes].copy(), CHANGE)
            try:
                for mode in modes:
                    report = dict(common, mode=mode, rows=[], body_probe=None)
                    selected = None
                    if mode == "both":
                        probe = sense(arena, shots[PROBE_CARD["id"]], stride, TEST_AMPLITUDES[:1])
                        residuals = [errors(memory, probe) for memory in memories]
                        index = int(np.argmin(residuals))
                        selected = memories[index]
                        report["body_probe"] = dict(selected_index=index, rms_rad=residuals, data=probe)
                    report["carried_age_exposures"] = 0 if mode == "frozen" else learned_age
                    for entry in plan["cards"]:
                        row = point(arena, renderer, rig, model, stride, lower, upper,
                                    shots[entry["id"]], entry, forward=forward,
                                    lag=report["carried_age_exposures"], nominal=memories[0], body=selected)
                        report["rows"].append(row)
                        _write(output_dir / f"mixed-{mode}.json.gz", report)
                        print(mode, entry["id"], row["score"]["learner"], row["score"]["returns"], flush=True)
            finally:
                renderer.close()
        finally:
            arena.close()


def positive_audit(source, output):
    report = _read(source)
    pairs = [(card, row) for card, row in zip(report["cards"], report["rows"], strict=True)
             if row["score"]["learner"] > 0]
    if not pairs:
        _write(output, dict(source_sha256=_hash(source), rows=[]))
        return
    with TemporaryDirectory(prefix="tt34-mixed-") as temp:
        filtered = Path(temp) / "positive.json.gz"
        audit_path = Path(temp) / "audit.json"
        cards, rows = map(list, zip(*pairs, strict=True))
        _write(filtered, dict(report, cards=cards, rows=rows))
        replay(filtered, audit_path, cad=True)
        result = _read(audit_path)
    _write(output, dict(result, source_sha256=_hash(source)))


def summarize(directory):
    files = {mode: directory / f"mixed-{mode}.json.gz" for mode in MODES}
    trials = {mode: _read(file) for mode, file in files.items()}
    cards = trials["frozen"]["cards"]
    if any(trial["cards"] != cards or len(trial["rows"]) != len(cards) for trial in trials.values()):
        raise ValueError("The combined test suite is incomplete or has different rosters")
    scores, pairwise = {}, {}
    for mode, trial in trials.items():
        rows = trial["rows"]
        scores[mode] = dict(
            points=len(rows), near_legal=sum(row["score"]["learner"] for row in rows),
            alternating=sum(row["score"]["returns"] for row in rows),
            later_near_legal=sum(max(0, row["score"]["learner"] - 1) for row in rows),
            positive=[row["shot"]["id"] for row in rows if row["score"]["learner"]],
            contacts=sum(p["near_contact"] is not None for row in rows for p in row["placements"]["delivered"]),
            active_ticks=sum(row["active_ticks"] for row in rows),
            rescaled_ticks=sum(row["rescaled_ticks"] for row in rows),
            clock_compensated_ticks=sum(row["clock_compensated_ticks"] for row in rows),
            max_measured_rate_rad_s=max(max(row["motor_audit"]["max_measured_rate_rad_s"]) for row in rows),
            cpu_s=sum(row["runtime_cpu_s"] for row in rows),
        )
        pairwise[mode] = compare_delivery(first_deliveries(trials["frozen"]["rows"]), first_deliveries(rows))
    paired = [
        dict(id=card["id"], scores={mode: [trial["rows"][i]["score"][key] for key in ("learner", "returns")]
                                   for mode, trial in trials.items()})
        for i, card in enumerate(cards)
    ]
    _write(directory / "mixed-results.json", dict(
        scores=scores, matches_frozen=pairwise, per_game=paired,
        sources={mode: _hash(file) for mode, file in files.items()},
    ))
    print(scores, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="action", required=True)
    trial = operations.add_parser("trial")
    trial.add_argument("feeds", type=Path)
    trial.add_argument("--course", type=Path, default=OUTPUT / "body-novel-plan.json.gz")
    trial.add_argument("--output", type=Path, default=OUTPUT)
    trial.add_argument("--modes", choices=MODES, nargs="+", default=MODES)
    verify = operations.add_parser("audit")
    verify.add_argument("source", type=Path)
    verify.add_argument("output", type=Path)
    report = operations.add_parser("summarize")
    report.add_argument("--directory", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == "trial":
        args.output.mkdir(parents=True, exist_ok=True)
        assess(args.feeds, args.course, args.output, args.modes)
    elif args.action == "audit":
        positive_audit(args.source, args.output)
    else:
        summarize(args.directory)


if __name__ == "__main__":
    main()
