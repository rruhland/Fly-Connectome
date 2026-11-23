"""Measure dose and transfer in two sequential, camera-driven solo rebound courses."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce_ident import _ellipse
from .tt24_carried_wrist import BUNDLE
from .tt24_metric_stereo import triangulate
from .tt24_public_body import stationary, stock
from .tt25_pose_demonstrations import episode
from .tt26_bounce_optics import OUTPUT as NOMINAL
from .tt26_bounce_optics import impact_candidates, rotation_between, separation, serialized, unit
from .tt26_grip_course import Grip, ReferenceReceiver, load_match
from .tt26_optical_bouncing import OpticalWrist
from .tt26_optical_transfer import decoder, transport
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT as BANK_DIR
from .tt27_flight_feedback import load_bank
from .tt27_outcome_receiver import independent

OUTPUT = Path("artifacts/tt28")
DOSES = (0, 4, 16, 64, 192)
MOUNTS = {
    "handle": (0.235, (0.10, 0.0, 1.0)),
    "canted": (0.235, (1.0, 0.0, -0.29)),
}
SCALES = np.array([0.008, 0.025, 0.055, 0.10])
PRACTICE_CAP = 180
HELD_CAP = 360
FIRST_FRAMES = 31
REFERENCE_PRIORITY = [0, 2, 3, 1]
SOURCE_SEEDS = {"handle": 28150370, "canted": 28153320}
HELD_SEED = 28410065
RECORD_SEED = 2876521
PERTURB_SEED = 28104817


class SoloLearner:
    """Keep the relative blade observation and a scalar return estimate per tolerance."""

    def __init__(self, reference):
        self.reference = reference
        self.directions = []
        self.rewards = np.zeros(len(SCALES))
        self.attempts = np.zeros(len(SCALES), dtype=int)

    def select(self, count, rng):
        if count < len(REFERENCE_PRIORITY):
            return REFERENCE_PRIORITY[count]
        if rng.random() < 0.24:
            return int(rng.integers(len(SCALES)))
        return self.best()

    def best(self):
        # A one-drop prior keeps a never-tried option available after several poor outcomes.
        return int(np.argmax((self.rewards + 0.7) / (self.attempts + 1)))

    def update(self, action, reward, direction):
        self.attempts[action] += 1
        self.rewards[action] += np.sqrt(reward)
        if direction is not None:
            self.directions.append(direction)

    def snapshot(self):
        direction = unit(np.median(self.directions, axis=0)) if self.directions else self.reference
        return dict(rotation=rotation_between(self.reference, direction).tolist(),
                    selected=self.best(), samples=len(self.directions),
                    attempts=self.attempts.tolist(), rewards=self.rewards.tolist())


def restore_learner(receipt, count):
    """Reconstruct the online state and its exploration stream after a saved prefix."""
    record = _read(receipt)
    if not 0 <= count <= len(record["rows"]):
        raise ValueError("The source receipt does not contain that many drops")
    if record["inherited_nominal_optics_sha256"] != _hash(NOMINAL):
        raise ValueError("Inherited nominal optics changed")
    nominal = np.asarray(_read(NOMINAL)["mounts"]["nominal"]["normal_link"])
    learner = SoloLearner(nominal)
    rng = np.random.default_rng(record["source_seed"] ^ 1761209)
    for i, row in enumerate(record["rows"][:count]):
        action = learner.select(i, rng)
        if action != row["action"]:
            raise ValueError(f"Exploration stream changed at drop {i + 1}")
        impact = row["first_impact"]
        direction = None if impact is None else impact["raw_normal_link"]
        learner.update(action, row["reward"], direction)
    expected = record["checkpoints"].get(str(count))
    if expected is not None and any(expected[k] != v for k, v in learner.snapshot().items()):
        raise ValueError(f"The restored checkpoint at drop {count} differs")
    return learner, rng


def optical_ball(images):
    points = []
    for rgb in images:
        ball = _ellipse(rgb)
        points.append(ball[:2] if ball is not None else (np.nan, np.nan))
    return points


def release(arena, policy, seed, regime, split, *, eyes=None, rig=None, acceleration=None, tape=None):
    render = tape is None
    obs = arena.reset(seed=seed, split=split, regime=regime, render=render)
    if render:
        policy.begin_release(obs)
    commands, encoders, pixels = [], [], []
    total = 0.
    for t in range(arena.max_steps):
        if eyes is not None and t < FIRST_FRAMES:
            pixels.append(optical_ball(arena.fixed_images()))
            encoders.append(obs["joints"].copy())
        action = policy.act(obs) if render else tape[t]
        commands.append(np.asarray(action).tolist())
        obs, reward, done = arena.step(action, render=render)
        total += reward
        if done:
            break
    metrics = arena.judge_metrics()
    detail = dict(actions=commands, reward=total, steps=metrics["steps"],
                  rebounds=metrics["rebounds"], streak=metrics["longest_consecutive_rebounds"],
                  referee=digest_json(metrics))
    if eyes is not None:
        ball, gap, _ = triangulate(eyes, np.asarray(pixels), tolerance=0.018)
        observations = dict(ball=ball, ray_gap=gap, pixels=pixels, joint_angles=np.asarray(encoders))
        candidates = impact_candidates(observations, acceleration, rig)
        detail["optics"] = serialized(observations)
        detail["first_impact"] = candidates[0] if candidates else None
    return detail


def wrist(arena, snap, action=None):
    bounds = arena._control_low[:5], arena._control_high[:5]
    selected = snap["selected"] if action is None else action
    return OpticalWrist(np.asarray(snap["rotation"]), bounds, position_scale=SCALES[selected])


def mkplan(seed, count):
    rng = np.random.default_rng(seed)
    return [(int(s), "changed" if i % 2 else "standard")
            for i, s in enumerate(rng.choice(2**31, count, replace=False))]


def course(args):
    label, max_dose, held_count, output = args
    began = perf_counter()
    arena = stock()
    arena.max_steps = PRACTICE_CAP
    fixture = Grip(arena._model)
    fixture.set(*MOUNTS[label])
    rig = decoder()
    prior = _read(NOMINAL)
    nominal = np.asarray(prior["mounts"]["nominal"]["normal_link"])
    acceleration = np.asarray(prior["free_flight_acceleration_m_s2"])
    learner = SoloLearner(nominal)
    seeds = mkplan(SOURCE_SEEDS[label], DOSES[-1])[:max_dose]
    held = mkplan(HELD_SEED, held_count)
    if {s for s, _ in seeds} & {s for s, _ in held}:
        raise ValueError("Reused practice seed")
    rng = np.random.default_rng(SOURCE_SEEDS[label] ^ 1761209)
    doses = [d for d in DOSES if d <= max_dose]
    snaps = {"0": learner.snapshot()}
    rows = []
    data = dict(schema="tt28-solo-dose-1", tool=label, source_seed=SOURCE_SEEDS[label],
                mount_evaluator=MOUNTS[label], practice_cap=PRACTICE_CAP, held_cap=HELD_CAP,
                optical_window=FIRST_FRAMES, inherited_bounce_sha256=_hash(BUNDLE),
                inherited_nominal_optics_sha256=_hash(NOMINAL), solo_tolerances_m=SCALES.tolist(),
                initial_exploration_order=REFERENCE_PRIORITY,
                policy_inputs="wrist RGB and own encoders",
                fit_inputs="first visually observed flight reversal from fixed stereo RGB and encoders",
                tolerance_inputs="sqrt of sum of the played release's scalar rebound rewards",
                camera_spec=[dict(lookat=e.lookat.tolist(), azimuth=e.azimuth,
                                  elevation=e.elevation, distance=e.distance) for e in arena.eyes],
                rows=rows, checkpoints=snaps, held=[])
    path = output / f"dose-solo-{label}.json.gz"
    try:
        arena.reset(seed=seeds[0][0], render=False)
        eyes = stationary(arena)
        for i, (seed, regime) in enumerate(seeds):
            before = learner.snapshot()
            action = learner.select(i, rng)
            result = release(arena, wrist(arena, before, action), seed, regime, "practice",
                             eyes=eyes, rig=rig, acceleration=acceleration)
            impact = result["first_impact"]
            direction = None if impact is None else impact["raw_normal_link"]
            learner.update(action, result["reward"], direction)
            rows.append(dict(seed=seed, regime=regime, attempt=i + 1, action=action, **result))
            if i + 1 in doses:
                snap = learner.snapshot()
                snaps[str(i + 1)] = snap
                print(label, "solo checkpoint", i + 1, "optical", snap["samples"],
                      "tolerance", SCALES[snap["selected"]],
                      "rolling streak", np.mean([r["streak"] for r in rows[-16:]]).round(2),
                      "elapsed", round(perf_counter() - began, 1), flush=True)
                _write(path, data)
        state = output / f"dose-checkpoints-{label}.npz"
        np.savez_compressed(
            state, schema="tt28-solo-dose-checkpoint-1", doses=doses,
            rotation=[s["rotation"] for s in snaps.values()],
            tolerance_m=[SCALES[s["selected"]] for s in snaps.values()],
            optical_samples=[s["samples"] for s in snaps.values()],
            attempts=[s["attempts"] for s in snaps.values()],
            rewards=[s["rewards"] for s in snaps.values()],
            wrist_sha256=_hash(BUNDLE), reference_sha256=_hash(NOMINAL),
        )
        # Reveal the tool geometry strictly as a scoring diagnostic after training.
        nominal_geom = Rotation.from_quat(np.roll(fixture.original, -1)).apply((0., 0., 1.))
        axis = unit(np.asarray(MOUNTS[label][1]))
        true_geom = Rotation.from_rotvec(MOUNTS[label][0] * axis).apply(nominal_geom)
        for snap in snaps.values():
            correction = np.asarray(snap["rotation"])
            snap["face_error_deg"] = separation(correction @ nominal_geom, true_geom)
        data["train_wall_seconds"] = perf_counter() - began
        data["held"].extend(held_solo(arena, held, snaps))
        data["checkpoint_sha256"] = _hash(state)
        data["wall_seconds"] = perf_counter() - began
        _write(path, data)
        print(label, "held solo", solo_summary(data["held"]), flush=True)
    finally:
        arena.close()
    return label


def held_solo(arena, held, snaps):
    arena.max_steps = HELD_CAP
    for seed, regime in held:
        results = {}
        for dose, config in snaps.items():
            result = release(arena, wrist(arena, config), seed, regime, "held")
            replay = release(arena, None, seed, regime, "held", tape=result["actions"])
            if replay != result:
                raise ValueError(f"Physical rebound replay differed for {seed}/{dose}")
            results[dose] = result
        yield dict(seed=seed, regime=regime, results=results)


def solo_summary(rows):
    if not rows:
        return {}
    base = [r["results"]["0"]["streak"] for r in rows]
    result = {}
    for dose in rows[0]["results"]:
        cases = [r["results"][dose] for r in rows]
        streaks = np.asarray([r["streak"] for r in cases])
        result[dose] = dict(rebounds=sum(r["rebounds"] for r in cases), longest=int(streaks.max()),
                            median=float(np.median(streaks)), at_least_five=int((streaks >= 5).sum()),
                            wins=int((streaks > base).sum()), losses=int((streaks < base).sum()))
    return result


def roster(shots, bank):
    rng = np.random.default_rng(PERTURB_SEED)
    ids = [r["id"] for r in bank["rows"]]
    fixed = independent(shots, seed=RECORD_SEED, near_count=4, unrestricted_count=96)
    shifted = [("perturbed", draw(shots, ids, rng, i)) for i in range(48)]
    return shifted + fixed


def target_banks(arena, goals, snaps):
    rig = decoder()
    axes = arena._arm_ids[1][2][:5]
    bounds = arena._joint_low[axes], arena._joint_high[axes]
    return {dose: np.stack([transport(rig, q, np.asarray(v["rotation"]), bounds) for q in goals])
            for dose, v in snaps.items()}


def target(args):
    label, feeds, output = args
    began = perf_counter()
    src = output / f"dose-solo-{label}.json.gz"
    shots, (_, _, scales) = load_match(feeds)
    bank = load_bank()
    keys = np.array([r["key"] for r in bank["rows"]])
    goals = np.array([r["goal"] for r in bank["rows"]])
    snaps = _read(src)["checkpoints"]
    arena = ReferenceReceiver()
    fixture = Grip(arena.model)
    fixture.set(*MOUNTS[label])
    banks = target_banks(arena, goals, snaps)
    rows = []
    replays = 0
    try:
        for group, shot in roster(shots, bank):
            results = {}
            for dose, options in banks.items():
                def choose(key):
                    return options[np.argmin(np.linalg.norm((keys - key) / scales, axis=1))]

                perf, key = episode(arena, shot, choose_goal=choose)
                if perf["legal"]:
                    if perf != episode(arena, shot, choose_goal=choose)[0]:
                        raise ValueError(f"Physical receiving replay differed for {shot['id']}/{dose}")
                    replays += 1
                results[dose] = perf
            rows.append(dict(group=group, shot=shot, key=None if key is None else key.tolist(),
                             results=results))
    finally:
        arena.close()
    report = dict(schema="tt28-regulation-dose-1", tool=label, source_sha256=_hash(src),
                  recorded_launch_sha256=_hash(feeds), bank_sha256=_hash(BANK_DIR / "demonstrations.json.gz"),
                  prior_receiver_scale=scales.tolist(), deck_seeds=[RECORD_SEED, PERTURB_SEED],
                  target_inputs="first four stereo RGB frames and own issued joint target history",
                  successful_replays=replays, rows=rows, wall_seconds=perf_counter() - began)
    report["summary"] = target_summary(rows)
    _write(output / f"dose-target-{label}.json.gz", report)
    print(label, "target", report["summary"], flush=True)


def target_summary(rows):
    result = {}
    for group in dict.fromkeys(r["group"] for r in rows):
        cases = [r["results"] for r in rows if r["group"] == group]
        result[group] = {}
        for dose in cases[0]:
            stats = [r[dose] for r in cases]
            result[group][dose] = dict(n=len(stats), legal=sum(r["legal"] for r in stats),
                                       contact=sum(r["contact"] for r in stats),
                                       gains_vs_zero=sum(r[dose]["legal"] > r["0"]["legal"] for r in cases),
                                       losses_vs_zero=sum(r[dose]["legal"] < r["0"]["legal"] for r in cases))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("solo", "target", "all"))
    p.add_argument("--feeds", type=Path)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--max-dose", type=int, choices=DOSES[1:], default=DOSES[-1])
    p.add_argument("--held", type=int, default=12)
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    if args.part != "solo" and args.feeds is None:
        p.error("receiving requires --feeds")
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
        if args.part in ("all", "solo"):
            list(pool.map(course, [(label, args.max_dose, args.held, args.output) for label in MOUNTS]))
        if args.part in ("all", "target"):
            list(pool.map(target, [(label, args.feeds, args.output) for label in MOUNTS]))


if __name__ == "__main__":
    main()
