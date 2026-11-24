"""Replay archived motor tapes through the optional slow net-deflection rescue."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_conditions import cooperative_targets, table_settings
from .tt29_endurance import audit
from .tt29_partner_fast import FastPartnerArena
from .tt29_rally_physics import OUTPUT, mujoco


def station(archive, rescue):
    return FastPartnerArena(
        strike_ready=archive["pose"], width=archive.get("size", (384, 288))[0],
        height=archive.get("size", (384, 288))[1], clearance_guard=True, fast_search=True,
        rate_scale=.8 if archive.get("rated") else 1., velocity_guard=archive.get("rated", False),
        motor_envelope=archive.get("motor_envelope", False), guard_table=True,
        targets=cooperative_targets(archive["pose"], archive.get("spread", "practice")),
        plant=table_settings(archive.get("condition", "stock")), net_rescue=rescue,
    )


def verify(archive, row, rescue, aim):
    arena = station(archive, rescue)
    try:
        arena.launch(row["shot"])
        arena.partner.reset(seed=row["opponent_seed"])
        far = np.zeros(5)
        follow = np.zeros(arena.action_size)
        end = aim["time"] if aim else np.inf
        diagnostics = dict(samples=0, min_table_m=.25, min_net_m=.25, max_speed_mps=0., max_drive_force_n=0.)
        original_air = arena._aerodynamics
        model = arena.model
        shadow = mujoco.MjData(model)
        table, net, blade = [model.geom(g).id for g in ("table", "net", "rival_racket")]
        expected = row["score"]["returns"]
        received = arena.rival_returns
        for event in row["events"]:
            received += int(event["kind"] == "legal_return" and event.get("side") == -1)

        def observe():
            original_air()
            if arena.data.time < end:
                return
            p = arena.partner
            shadow.qpos[:] = arena.data.qpos
            mujoco.mj_kinematics(model, shadow)
            diagnostics["samples"] += 1
            diagnostics["min_table_m"] = min(diagnostics["min_table_m"], mujoco.mj_geomDistance(model, shadow, table, blade, .25, None))
            diagnostics["min_net_m"] = min(diagnostics["min_net_m"], mujoco.mj_geomDistance(model, shadow, net, blade, .25, None))
            diagnostics["max_speed_mps"] = max(diagnostics["max_speed_mps"], float(np.linalg.norm(arena.data.qvel[p.qvel:p.qvel+3])))
            diagnostics["max_drive_force_n"] = max(diagnostics["max_drive_force_n"], float(np.linalg.norm(arena.data.xfrc_applied[p.body, :3])))

        if rescue and aim:
            arena._aerodynamics = observe
        tape = row["actions"]
        for tick in range(len(tape) + (120 if rescue and aim else 0)):
            command = tape[tick] if tick < len(tape) else follow
            _, _, done, cap, _ = arena.exchange(command, far)
            if done or cap or (rescue and aim and arena.rival_returns > received):
                break
        events = arena.events.copy()
        score = audit(events)
        original = row["events"]
        if rescue and aim:
            prefix = original[:-2]
            if digest_json(events[:len(prefix)]) != digest_json(prefix):
                raise ValueError("Changed a native event before the archived second bounce")
            if score["returns"] != expected + 1 or score["opponent"] != row["score"]["opponent"] + 1:
                raise ValueError("The far rescue did not make the next legal alternating return")
            if diagnostics["min_table_m"] < 0 or diagnostics["min_net_m"] < 0:
                raise ValueError("The rescue blade penetrated the table or net")
        else:
            prefix = original
            if digest_json(events) != digest_json(original):
                raise ValueError("The control point did not reproduce its archived events")
        return dict(opening=row["shot"]["id"], rescue=rescue, returns=score["returns"],
                    old_tape_ticks=len(tape), replayed_ticks=arena.steps,
                    continuation_ticks=max(0, arena.steps-len(tape)), prefix_events=len(prefix),
                    prefix_sha256=digest_json(prefix), events_sha256=digest_json(events),
                    endings=events[-14:], result=arena.point_result(),
                    rescue_plans=[x for x in arena.partner.reports if x["stage"] == "net_rescue"],
                    carriage_diagnostics=diagnostics if rescue and aim else None)
    finally:
        arena.close()


def run(varied, envelope, output):
    rows = []
    for path, sid, should_rescue in ((envelope, 7345, True), (varied, 11292, True), (varied, 7271, False)):
        tape = _read(path)
        source = next(row for row in tape["rows"] if row["shot"]["id"] == sid)
        aim = None
        if should_rescue:
            aim = next(e for e in reversed(source["events"]) if e.get("collider") == "net" and e["kind"] == "separation")
            if source["events"][-1].get("reason") != "second_bounce":
                raise ValueError("Expected an archived missed short ball")
        baseline = verify(tape, source, False, aim)
        modified = verify(tape, source, True, aim)
        if not should_rescue and modified["rescue_plans"]:
            raise ValueError("Unneeded rescue on the control")
        rows.append(dict(source=str(path.name), source_sha256=_hash(path), control=baseline, opt_in=modified))
        _write(output, dict(schema="tt29-net-rescue-1", rows=rows))
        print(sid, baseline["returns"], modified["returns"], modified["carriage_diagnostics"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--varied", type=Path, default=OUTPUT / "varied-opponent.json.gz")
    p.add_argument("--envelope", type=Path, default=OUTPUT / "envelope-carry.json.gz")
    p.add_argument("--output", type=Path, default=OUTPUT / "tt29_partner_short.json.gz")
    args = p.parse_args()
    run(args.varied, args.envelope, args.output)
