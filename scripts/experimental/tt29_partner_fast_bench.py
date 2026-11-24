"""Benchmark native and default far planning on matched physical incoming releases."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_partner import FOREHAND_LANES, PartnerArena
from .tt29_partner_fast import FastPartnerArena
from .tt29_partner_validation import play
from .tt29_rally_physics import OUTPUT, measured_launches

COACH = OUTPUT / "rally-physics-100.json.gz"
RESULT = OUTPUT / "tt29_partner_fast.json.gz"
TAPE = OUTPUT / "tt29_partner_fast_tape.json"
HELD = (5906, 6971, 9681, 13160, 14221, 9339, 3233, 10831)


def recorded_outputs(report):
    count = 0
    for event in report["events"]:
        if event["kind"] != "separation" or event["collider"] != "learner_racket":
            continue
        count += 1
        x, y, z = event["ball"]
        vx, vy, vz = event["velocity"]
        wx, wy, wz = event["spin"]
        # The shared admission helper rotates the launch once, including spin.
        yield dict(id=29054700 + count, pos_x=-x, pos_y=-y, pos_z=z - TABLE_HEIGHT,
                   vel_x=-vx, vel_y=-vy, vel_z=vz, w_vel_x=-wx, w_vel_y=-wy, w_vel_z=wz)
        if count == 8:
            break


def timed_attempt(arena, source):
    chooser = arena.partner._plan
    calls = []

    def measured(*args):
        cpu, wall = process_time(), perf_counter()
        result = chooser(*args)
        calls.append(dict(cpu_s=process_time() - cpu, wall_s=perf_counter() - wall,
                          candidate_windows=result[1]["windows"], verified=result[1]["verified"]))
        return result

    arena.partner._plan = measured
    clock = process_time()
    try:
        result = play(arena, source)
    finally:
        arena.partner._plan = chooser
    return dict(result, planning=calls, total_cpu_s=process_time() - clock)


def summarize(rows):
    values = {}
    for key in ("default", "native"):
        cases = [r[key] for r in rows]
        plans = [c for r in cases for c in r["planning"]]
        errors = []
        for r in cases:
            if r["landing"] is not None and r["receiving_state"] is not None:
                state, target = r["receiving_state"], r["target"]
                errors.append([abs(state[1] - target[0]), abs(state[3] - target[2]), abs(state[5] + target[3])])
        values[key] = dict(planning_cpu_s=sum(p["cpu_s"] for p in plans), planning_wall_s=sum(p["wall_s"] for p in plans),
                           total_cpu_s=sum(r["total_cpu_s"] for r in cases), plan_calls=len(plans),
                           far_contacts=sum(r["result"]["rival_contacts"] for r in cases),
                           far_legal=sum(r["landing"] is not None for r in cases),
                           received_legal=sum(r["received_legal"] for r in cases),
                           at_target_depth=len(errors), median_target_errors=np.median(errors, axis=0).tolist() if errors else None)
    return dict(values, planning_speedup=values["default"]["planning_cpu_s"] / values["native"]["planning_cpu_s"],
                end_to_end_speedup=values["default"]["total_cpu_s"] / values["native"]["total_cpu_s"],
                exact_event_matches=sum(r["default"]["event_sha256"] == r["native"]["event_sha256"] for r in rows))


def run(feeds, output):
    coach = _read(COACH)
    shots = measured_launches(feeds)
    deck = [("physical_near_outputs", s) for s in recorded_outputs(coach)]
    deck.extend(("measured_rotated_held", shots[sid]) for sid in HELD)
    rows = []
    settings = dict(max_steps=145, targets=FOREHAND_LANES, guard_table=True)
    baseline, native = PartnerArena(**settings), FastPartnerArena(**settings)
    try:
        for index, (group, shot) in enumerate(deck):
            order = [("default", baseline), ("native", native)]
            if index % 2:
                order.reverse()
            results = {name: timed_attempt(arena, shot) for name, arena in order}
            rows.append(dict(cohort=group, launch_id=shot["id"], **results))
            print(group, shot["id"], {mode: (bool(results[mode]["landing"]), round(results[mode]["total_cpu_s"], 3))
                                      for mode in ("default", "native")}, flush=True)
    finally:
        baseline.close()
        native.close()
    results = {group: summarize([r for r in rows if r["cohort"] == group]) for group in dict.fromkeys(r["cohort"] for r in rows)}
    results["all"] = summarize(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, dict(schema="tt29-partner-native-perf-1", measured_source_sha256=_hash(feeds),
                       coach_source_sha256=_hash(COACH), results=results, rows=rows))
    print(results, flush=True)


def full_tape(output):
    """Apply the same archived near side actions while the faster partner plays live."""
    coach = _read(COACH)
    settings = dict(coach["configuration"])
    settings["targets"] = settings.pop("partner_lanes")
    arena = FastPartnerArena(**settings)
    try:
        arena.launch(coach["launch"])
        state = sha256()
        began = process_time()
        for near, far in coach["actions"]:
            _, _, done, capped, _ = arena.exchange(near, far)
            state.update(arena.data.qpos.astype("<f8").tobytes())
            state.update(arena.data.qvel.astype("<f8").tobytes())
            if done or capped:
                break
        receipt = dict(coach_sha256=_hash(COACH), steps=arena.steps, returns=audit(arena.events)["returns"],
                       far_returns=arena.rival_returns - 1, events_sha256=digest_json(arena.events),
                       original_events_sha256=digest_json(coach["events"]),
                       state_sha256=state.hexdigest(), original_state_sha256=coach["replay"]["state_sha256"],
                       cpu_s=process_time() - began, simulated_s=arena.steps * arena.control_dt)
        _write(output, receipt)
        print(receipt, flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-tape", action="store_true")
    arguments = parser.parse_args()
    if arguments.full_tape:
        full_tape(arguments.output or TAPE)
    else:
        run(arguments.feeds, arguments.output or RESULT)
