"""Compare retained and inherited wrist calibration in complete live regulation points."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_action_memory import run as run_point
from .tt29_arm_clearance import ArmClearance
from .tt29_bounce import OUTPUT, cold_snapshot, load_checkpoint
from .tt29_bounce_promotion import RETAINED
from .tt29_endurance import audit
from .tt29_partner import DEEP_LANES, FOREHAND_LANES, PartnerArena
from .tt29_vision import FlightPractice, OpticalFlight, Rig, ball_pixels

MEMORY = OUTPUT / "camera-action-memory.json.gz"
RESULT = OUTPUT / "tt29_warm_rallies.json.gz"
POINTS = (
    ("historical_opportunity", 7271, False, "wrist"),
    ("historical_opportunity", 7271, "lateral", "left+wrist"),
    ("held_for_motor_memory", 4230, False, "wrist"),
    ("held_for_motor_memory", 15489, "lateral", "left+wrist"),
)


def replay(row, pose, view, profile):
    """Play applied commands against the same independently resetting physical partner."""
    settings = dict(strike_ready=pose, width=384, height=288, clearance_guard=True,
                    fast_search=True, max_steps=9000, guard_table=True,
                    targets=FOREHAND_LANES if pose == "lateral" else DEEP_LANES)
    arena = PartnerArena(**settings)
    cad = PartnerArena(**settings, detail="cad")
    try:
        arena.launch(row["shot"])
        arena.partner.reset(seed=row["opponent_seed"])
        arena.sample_clearance(True)
        arena._cad_reader = (ArmClearance(cad.model), mujoco.MjData(cad.model))
        arena.streams = tuple(view.split("+"))
        rig = Rig.from_arena(arena)
        transform = profile["registration"]
        rig.wrist = rig.wrist.remount(transform[3:6], Rotation.from_rotvec(transform[:3]).as_matrix(),
                                     focal_px=profile["focal_px"])
        tracker = OpticalFlight(rig, FlightPractice(), learn=True)
        readings = dict.fromkeys(arena.streams, 0)
        error = []
        completed = False
        for index, command in enumerate(row["actions"]):
            if index < 12:
                rgb = arena.render_cameras()
                for name, image in rgb.items():
                    readings[name] += int(ball_pixels(image) is not None)
                now = index * arena.control_dt
                tracker.observe(now, rgb, arena.joints)
                if tracker.latest is not None and tracker.latest.time == now:
                    error.append(float(np.linalg.norm(tracker.latest.position - arena.ball)))
            _, _, done, cap, _ = arena.exchange(command, np.zeros(5))
            completed = done or cap or arena.legal_returns + arena.rival_returns >= 101
        statistics = audit(arena.events)
        if not completed or digest_json(arena.events) != digest_json(row["events"]):
            raise ValueError(f"The continuously played point did not replay for {row['shot']['id']}/{pose}")
        return dict(referee=digest_json(arena.events), complete=completed, steps=arena.steps,
                    reason=arena.reason, score=statistics["returns"], motor=arena.motor_audit(),
                    clearance=arena.clearance_result(), opening_vision=dict(ticks=min(12, arena.steps),
                        segmented_frames=readings, valid_estimates=len(error),
                        median_position_error_m=float(np.median(error)) if error else None))
    finally:
        arena.close()
        cad.close()


def run(feeds, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = _read(MEMORY)
    old_point = next(row for row in reference["rows"] if row["shot"]["id"] == 7271)
    n = old_point["initial_examples"]
    if n != 7 or old_point["score"]["returns"] != 66:
        raise ValueError("The initial demonstration-memory boundary changed")
    retained = load_checkpoint(RETAINED)
    cold = cold_snapshot() | dict(focal_px=retained["focal_px"])
    report = dict(schema="tt29-continuous-wrist-warmth-1", source_sha256=_hash(feeds),
                  memory_sha256=_hash(MEMORY), retained_sha256=_hash(RETAINED),
                  starting_examples=n, example_cut="prefix saved before opening 7271",
                  benchmark_7271_two_fixed_camera_returns=old_point["score"]["returns"],
                  plan=[list(row) for row in POINTS], rows=[], verified=0)
    if output.exists():
        prior = _read(output)
        for key in ("source_sha256", "memory_sha256", "retained_sha256", "plan"):
            if report[key] != prior[key]:
                raise ValueError("Existing comparison used different inputs")
        report = prior
    with TemporaryDirectory(prefix="tt29_warm_rallies_") as scratch:
        session = Path(scratch)
        initial = session / "memory.json.gz"
        cold_file = session / "legacy_wrist.json"
        _write(initial, dict(checkpoint=reference["checkpoint"][:n]))
        _write(cold_file, cold)
        for cohort, sid, pose, view in POINTS:
            for label, profile in (("retained", RETAINED), ("cold", cold_file)):
                record = next((r for r in report["rows"] if (r["id"], r["pose"], r["view"], r["mode"])
                               == (sid, pose, view, label)), None)
                if record is None:
                    physical = session / "point.json.gz"
                    run_point(feeds, physical, [sid], view, pose, checkpoint=initial, bounce=profile)
                    point = _read(physical)["rows"][0]
                    if point["initial_examples"] != n or point["opponent_seed"] != sid:
                        raise ValueError("Paired point did not start from the shared memory and opponent seed")
                    record = dict(cohort=cohort, id=sid, pose=pose, view=view, mode=label, point=point)
                    report["rows"].append(record)
                    _write(output, report)
                if "replay" in record:
                    continue
                point = record["point"]
                verified = replay(point, pose, view, load_checkpoint(profile))
                record["replay"] = verified
                report["verified"] += 1
                _write(output, report)
                print(cohort, sid, pose, view, label, point["score"]["returns"], verified["reason"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    run(args.feeds, args.output)
