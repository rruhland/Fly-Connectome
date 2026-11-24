"""Paired counterfactual retries of an archived rally leg, scored by the live referee."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _read, _write, _hash
from .tt29_conditions import CONDITIONS, apply, cooperative_targets, table_settings
from .tt29_partner_fast import FastPartnerArena


def retry(arena, row, handover, shift, expected, trace=None):
    arena.launch(row["shot"])
    arena.partner.reset(seed=row["opponent_seed"])
    actions = np.asarray(row["actions"])
    neutral = np.zeros(actions.shape[1])
    far = np.zeros(5)
    state = None
    response = []
    for t in range(len(actions) + 120):
        if t == handover:
            state = sha256(arena.data.qpos.tobytes() + arena.data.qvel.tobytes()
                           + arena.data.ctrl.tobytes()).hexdigest()
        idx = t if t < handover else t - shift
        command = actions[idx] if 0 <= idx < len(actions) and (t < handover or idx >= handover) else neutral
        _, _, done, capped, info = arena.exchange(command, far)
        if t >= handover:
            if trace is not None:
                trace.append([arena.data.time, *arena.ball.tolist()])
            response.extend(info["events"])
            landed = any(e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"])
            if landed or done or capped:
                if shift == 0 and digest_json(response) != digest_json(expected[:len(response)]):
                    raise ValueError(f"Unmodified retry diverged: {response[:1]}, expected {expected[:1]}")
                hit = next((e for e in response if e["kind"] == "contact" and e["side"] == 1), None)
                return dict(shift=shift, state=state, landed=landed, contact=hit is not None,
                            reason=arena.point_result()["reason"], events=response)
        elif done or capped:
            raise ValueError("Prefix ended before the incoming shot")
    raise ValueError("Retry never reached a decision")


def run(path, output, shifts):
    archive = _read(path)
    condition = archive["condition"]
    if not archive["motor_envelope"] or not archive["fast_partner"] or not archive["rated"]:
        raise ValueError("Expected a rated, coupled-motor, native-partner archive")
    arena = FastPartnerArena(strike_ready=archive["pose"], fast_search=True, clearance_guard=True,
                             motor_envelope=True, velocity_guard=True, rate_scale=.8, guard_table=True,
                             max_steps=9000, targets=cooperative_targets(archive["pose"], archive["spread"]),
                             net_rescue=archive.get("net_rescue", False), plant=table_settings(condition))
    source_hash = _hash(path)
    rows = []
    if output.exists():
        saved = _read(output)
        if saved["source_sha256"] != source_hash or saved["shifts"] != shifts:
            raise ValueError("Existing retry receipt used different inputs")
        rows = saved["rows"]
    completed = {(r["id"], r["opponent_seed"], r["start"]) for r in rows}
    try:
        if condition not in ("stock", "two_surface", "narrow_table"):
            apply(arena, CONDITIONS[condition])
            arena.partner = type(arena.partner)(arena, targets=arena.partner.targets, guard_table=True)
        for row in archive["rows"]:
            final = row["events"][-1]
            if final["kind"] != "point" or final["winner"] != -1:
                continue
            far_contacts = [e for e in row["events"] if e["kind"] == "contact" and e["side"] == -1]
            # One prior successful leg per point is the specificity control. An
            # archived far contact positions the replay fixture, outside the policy.
            starts = [0] + [e["step"] for e in far_contacts]
            for start in starts[-2:]:
                leg = "terminal" if start == starts[-1] else "previous"
                if (row["shot"]["id"], row["opponent_seed"], start) in completed:
                    continue
                expected = [v for v in row["events"] if v["step"] > start]
                trials = [retry(arena, row, start, shift, expected) for shift in shifts]
                if len({r["state"] for r in trials}) != 1:
                    raise ValueError("Retiming changed the geometry or command state before the retry")
                rows.append(dict(id=row["shot"]["id"], opponent_seed=row["opponent_seed"],
                                 leg=leg, start=start, trials=trials))
                print(rows[-1]["id"], leg, [(t["shift"], t["landed"]) for t in trials], flush=True)
                _write(output, dict(source_sha256=source_hash, shifts=shifts, rows=rows))
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shifts", type=int, nargs="+", default=[0, -1, 1, -2, 2])
    args = parser.parse_args()
    run(args.source, args.output, args.shifts)
