"""Compare cold and bounce-registered wrist receiving on a matched opening family."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt25_reference_receiver import ReferenceReceiver
from .tt29_bounce import CHECKPOINT, COURSE, OUTPUT, cold_snapshot, held_score, load_checkpoint
from .tt29_vision import Rig, ball_pixels

DEMONSTRATION = OUTPUT / "tt29_bounce_receive_demo.json.gz"
RECEIPT = OUTPUT / "tt29_bounce_receive.json.gz"
PRACTICE_SEED = 291040503
HELD_SEED = 291180551
ACTION_SEED = 29682537
OBSERVE = 5
PRACTICE = 32
HELD = 24
CAP = 120
SETTINGS = np.array([0., -.05, .05, -.10, .10])
FIELDS = tuple(f"{kind}_{axis}" for kind in ("pos", "vel", "w_vel") for axis in "xyz")
DEVIATIONS = np.array([.004, .004, .004, .008, .012, .015, .40, .40, .40])


@dataclass
class WristAdjustment:
    attempts: np.ndarray = field(default_factory=lambda: np.zeros(len(SETTINGS), dtype=int))
    wins: np.ndarray = field(default_factory=lambda: np.zeros(len(SETTINGS), dtype=int))

    def best(self):
        probability = (self.wins + np.array([2., 1., 1., 1., 1.])) / (self.attempts + 3.)
        return int(np.argmax(probability))

    def update(self, action, legal):
        self.attempts[action] += 1
        self.wins[action] += int(legal)

    def snapshot(self):
        return dict(attempts=self.attempts.tolist(), legal=self.wins.tolist(), selected=self.best())


def paired_deck(donor, seed, count):
    rng = np.random.default_rng(seed)
    original = np.array([donor[field] for field in FIELDS])
    ranges = rng.uniform(-1., 1., (count, len(FIELDS))) * DEVIATIONS
    return [donor | dict(zip(FIELDS, original + difference, strict=True)) for difference in ranges]


def carried_mount(rig, checkpoint, focal):
    transform = checkpoint["registration"]
    return rig.wrist.remount(transform[3:6], Rotation.from_rotvec(transform[:3]).as_matrix(), focal_px=focal)


def capture(arena, shot):
    arena.launch(shot)
    exposures = []
    for tick in range(OBSERVE + 1):
        found = ball_pixels(arena.render_wrist())
        if found is not None:
            exposures.append((found.center, arena.joints.copy()))
        if tick < OBSERVE:
            arena.step(np.zeros(arena.action_size), render=False)
    return exposures


def wrist_key(exposures, camera):
    rays = [camera.at(q).ray(pixel) for pixel, q in exposures]
    bearings = [ray[[0, 2]] / ray[1] for ray in rays if ray[1] > .20]
    return (np.median(bearings, axis=0) if len(bearings) >= 3 else None), len(bearings)


def one(arena, shot, demonstration, *, action=None, replay=None, teacher=False):
    """Commit a remembered rate pattern using only the first six wrist exposures."""
    arena.launch(shot)
    actions = []
    for tick in range(arena.max_steps):
        if replay is not None:
            command = replay[tick]
        else:
            command = np.zeros(arena.action_size)
            if (teacher or action is not None) and tick < len(demonstration):
                command = np.array(demonstration[tick])
                if action is not None and 24 <= tick < 34:
                    command[4] = np.clip(command[4] + SETTINGS[action], -1., 1.)
        _, _, done, cap, _ = arena.step(command, render=False)
        actions.append(np.asarray(command).tolist())
        if done or cap or arena.legal_returns:
            break
    return dict(actions=actions, contact=arena.contacts, legal=arena.legal_returns, incoming=arena.rival_returns,
                steps=arena.steps, reason=arena.reason, referee=digest_json(arena.events), events=arena.events.copy())


def summarize(rows):
    outcomes = [row["results"] for row in rows]
    return {name: dict(returns=sum(row[name]["legal"] for row in outcomes),
                       contacts=sum(row[name]["contact"] for row in outcomes),
                       accepted=sum(row[name]["attempted"] is not None for row in outcomes),
                       first_return=next((i for i, row in enumerate(outcomes, 1) if row[name]["legal"]), None),
                       trials=len(outcomes)) for name in outcomes[0]}


def run(output=OUTPUT, *, checkpoint_path=CHECKPOINT, course_path=COURSE):
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    warmed = load_checkpoint(checkpoint_path)
    cold = cold_snapshot()
    before_hash = _hash(checkpoint_path)
    demo = _read(DEMONSTRATION)
    source_shot = demo["shot"]
    source_tape = demo["actions"]
    arena = ReferenceReceiver(fixture="table-edge", view_mode="wrist", width=640, height=480,
                              frameskip=4, max_steps=CAP)
    modes = {"cold": cold, "bounce": warmed, "fixed_bounce": warmed}
    learners = {"cold": WristAdjustment(), "bounce": WristAdjustment()}
    practice, held = [], []
    verified = 0
    rng = np.random.default_rng(ACTION_SEED)
    try:
        if not np.allclose(np.asarray(source_tape[:OBSERVE]), 0):
            raise ValueError("The demonstration has a command before the camera decision")
        rig = Rig.from_arena(arena)
        focal = arena.lens_intrinsics()["wrist"]["focal"] * 144 / arena.height
        cameras = {label: carried_mount(rig, snap, focal) for label, snap in modes.items()}
        expected = one(arena, source_shot, source_tape, teacher=True)
        if not expected["legal"] or expected["referee"] != digest_json(demo["events"]):
            raise ValueError("Archived receiving demonstration changed on the updated fixture")
        reference, _ = wrist_key(capture(arena, source_shot), cameras["bounce"])
        if reference is None:
            raise ValueError("Receiving demonstration has no plausible wrist bearing")
        for dest, schedule, update in ((practice, paired_deck(source_shot, PRACTICE_SEED, PRACTICE), True),
                                       (held, paired_deck(source_shot, HELD_SEED, HELD), False)):
            for index, shot in enumerate(schedule):
                common_action = None
                if update and index < len(SETTINGS):
                    common_action = index
                elif update and rng.random() < .22:
                    common_action = int(rng.integers(len(SETTINGS)))
                outcomes = {}
                # Every branch issues zero for these first five command intervals.
                # Compute one common set of native RGB detections at decision time.
                exposures = capture(arena, shot)
                for label, camera in cameras.items():
                    learner = learners.get(label)
                    key, seen = wrist_key(exposures, camera)
                    chosen = None
                    if key is not None and np.linalg.norm(key - reference) < .055:
                        chosen = 0 if learner is None else learner.best()
                        if learner is not None and common_action is not None:
                            chosen = common_action
                    result = one(arena, shot, source_tape, action=chosen)
                    again = one(arena, shot, source_tape, replay=result["actions"])
                    if again != result:
                        raise ValueError(f"Receiving command replay changed for {label}/{index}")
                    verified += 1
                    if update and learner is not None and chosen is not None:
                        learner.update(chosen, result["legal"])
                    outcomes[label] = result | dict(key=None if key is None else key.tolist(), attempted=chosen, seen=seen)
                dest.append(dict(attempt=index + 1, shot=shot, exploration=common_action,
                                 wrist_samples=[dict(pixel=px.tolist(), encoders=q.tolist()) for px, q in exposures],
                                 results=outcomes))
    finally:
        arena.close()
    summary = dict(practice=summarize(practice), held=summarize(held))
    receipt = dict(schema="tt29-wrist-receive-transfer-1", solo_checkpoint_sha256=before_hash,
                   demonstration_sha256=_hash(DEMONSTRATION), dataset_sha256=demo["source_sha256"],
                   opening_family_id=demo["selected_launch"], practice_seed=PRACTICE_SEED, held_seed=HELD_SEED,
                   action_seed=ACTION_SEED, jitter=dict(zip(FIELDS, DEVIATIONS, strict=True)),
                   observation_ticks=OBSERVE + 1, camera_size=[640, 480], reference_bearing=reference.tolist(),
                   controller_inputs="wrist RGB and capture-time encoders", online_reward="legal physical return",
                   settings=SETTINGS.tolist(), contacts_clock_hz=1 / arena.control_dt,
                   practice=practice, held=held, verified_receiving=verified, summary=summary,
                   table_update={key: learner.snapshot() for key, learner in learners.items()},
                   table_wall_seconds=perf_counter() - start)
    _write(output / RECEIPT.name, receipt)
    print(summary, flush=True)
    finish_retention(receipt, output, checkpoint_path, course_path)


def finish_retention(receipt, output, checkpoint_path, course_path):
    start = perf_counter()
    baseline = _read(course_path)
    checkpoint = load_checkpoint(checkpoint_path)
    before_hash = receipt["solo_checkpoint_sha256"]
    if baseline["published_checkpoint_sha256"] != before_hash or _hash(checkpoint_path) != before_hash:
        raise ValueError("Solo and receiving checkpoint hashes disagree")
    count = len(baseline["held"])
    label = str(checkpoint["drops"])
    post_table_held = held_score({label: load_checkpoint(checkpoint_path)}, held_count=count, seed=baseline["held_seed"])
    original = [{**row, "results": {label: row["results"][label]}} for row in baseline["held"]]
    if post_table_held != original or _hash(checkpoint_path) != before_hash:
        raise ValueError("The frozen bounce checkpoint changed after table practice")
    receipt.update(post_table_held=post_table_held, post_table_matches=count, retention_wall_seconds=perf_counter() - start)
    _write(output / RECEIPT.name, receipt)
    print("post-table held", count, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--course", type=Path, default=COURSE)
    parser.add_argument("--retention-only", action="store_true")
    args = parser.parse_args()
    if args.retention_only:
        finish_retention(_read(args.output / RECEIPT.name), args.output, args.checkpoint, args.course)
    else:
        run(args.output, checkpoint_path=args.checkpoint, course_path=args.course)
