"""Verify a paired physical readout comparison from bounded live command tapes."""

import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt43_motor_modes as run
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_opponent import draw


def score(row):
    return row["score"]["learner"]


def report(args):
    motor = _read(args.motor)
    stock = _read(args.stock)
    if motor["stock_sha256"] != _hash(args.stock):
        raise ValueError("Different paired stock tape")
    paired = len(motor["rows"])
    cards = motor["cards"][:paired]
    if not cards or cards != stock["cards"][:paired]:
        raise ValueError("Missing paired release")
    pairs = list(zip(cards, stock["rows"][:paired], motor["rows"], strict=True))
    changed = [i for i, (_, _, new) in enumerate(pairs) if new["first_action_diff"] is not None]
    # Verify CAD for the longest changed point and the largest gain; both are
    # selected after paired testing for audit, not used to choose the policy.
    cad = {max(changed, key=lambda i: score(pairs[i][2])),
           max(changed, key=lambda i: score(pairs[i][2]) - score(pairs[i][1]))} if changed else set()
    result = {
        "protocol_sha256": motor["protocol_sha256"],
        "stock_sha256": _hash(args.stock), "motor_sha256": _hash(args.motor),
        "paired_games": paired, "complete": motor["complete"], "rows": [],
        "score": {
            "stock_total": sum(score(old) for _, old, _ in pairs),
            "motor_total": sum(score(new) for _, _, new in pairs),
            "stock_peak": max(score(old) for _, old, _ in pairs),
            "motor_peak": max(score(new) for _, _, new in pairs),
            "improved": sum(score(new) > score(old) for _, old, new in pairs),
            "declined": sum(score(new) < score(old) for _, old, new in pairs),
            "unchanged": sum(score(new) == score(old) for _, old, new in pairs),
        },
    }
    contacts = 0
    with mpc.centered(motor["ready"]):
        table = mpc.arena()
        try:
            for ix, (c, old, new) in enumerate(pairs):
                if old["shot"] != new["shot"] or old["opponent_seed"] != new["opponent_seed"]:
                    raise ValueError("Both arms must open on the same seed and measured ball")
                a, b = old["placements"]["requested"], new["placements"]["requested"]
                prefix_draws = min(len(a), len(b))
                if any(p["target"] != q["target"] for p, q in zip(a, b)):
                    raise ValueError("Common independent opponent placement outcomes differ")
                t = new["first_action_diff"]
                if t is not None and digest_json([e for e in old["events"] if e["step"] < t]) != digest_json([e for e in new["events"] if e["step"] < t]):
                    raise ValueError("Physical events differ before any changed movement")
                row = {
                    "id": c["id"], "first_action_diff": t, "common_prefix_draws": prefix_draws,
                    "legal_returns": [score(old), score(new)],
                    "alternating_returns": [old["score"]["returns"], new["score"]["returns"]],
                    "attempts": new["planner"].get("mode_attempts", 0),
                    "selected": new["planner"].get("mode_selected", 0),
                    "last_searches": new["searches"][-4:],
                    "stock_event_sha256": digest_json(old["events"]),
                    "motor_event_sha256": digest_json(new["events"]),
                    "cpu_seconds": [old["runtime_cpu_s"], new["runtime_cpu_s"]],
                }
                for tag, tape in (("stock", old), ("motor", new)):
                    u = np.asarray(tape["actions"])
                    if not np.all(np.isfinite(u)) or np.max(abs(u)) > .8 + 1e-12:
                        raise ValueError("Control tape exceeds the direct rate cap")
                    if any(move["target"] != list(draw(c, leg, "random", None)[0])
                           for leg, move in enumerate(tape["placements"]["requested"])):
                        raise ValueError("Tape's opponent draw differs from the frozen schedule")
                    if args.replay:
                        table.sample_clearance(live=bool(tag == "motor" and ix in cad))
                        starts = table.launches
                        table.launch(tape["shot"], card=c)
                        table.partner.reset(seed=c["seed"])
                        for cmd in u:
                            table.exchange(cmd, mpc.FAR_COMMAND)
                        if table.launches - starts != 1 or digest_json(table.events) != digest_json(tape["events"]):
                            raise ValueError("Command replay changed tape or launches")
                        res = audit(table.events, opening_side=1)
                        if digest_json(res) != digest_json(tape["score"]):
                            raise ValueError("Independent referee result differs")
                        row[tag + "_exact_replay"] = True
                        row[tag + "_launches"] = 1
                        if tag == "motor" and ix in cad:
                            row["clearance"] = table.clearance_result()
                result["rows"].append(row)
                contacts += row["selected"]
        finally:
            table.close()
    result["mode_choices"] = contacts
    _write(args.out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("motor", type=Path)
    p.add_argument("stock", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--replay", action="store_true")
    report(p.parse_args())
