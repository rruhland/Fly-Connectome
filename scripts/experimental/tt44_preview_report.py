"""Verify paired preview policy tapes and report physical practice and transfer."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt44_preview as preview
from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance


def phase_report(rows, controls, phase):
    if len(rows) != len(controls):
        raise ValueError(f"Unequal {phase} releases")
    stats = [Counter(), Counter()]
    pairs = []
    for i, (old, new) in enumerate(zip(controls, rows, strict=True)):
        if old["card"] != new["card"] or old["shot"] != new["shot"]:
            raise ValueError("Different measured launch or opponent seeds")
        c = old["card"]
        p, q = (row["placements"]["requested"] for row in (old, new))
        if any(a["target"] != b["target"] for a, b in zip(p, q)):
            raise ValueError("Common opponent choices changed")
        x, y = old["actions"], new["actions"]
        t = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v),
                 None if len(x) == len(y) else min(len(x), len(y)))
        if t is None and digest_json(old["events"]) != digest_json(new["events"]):
            raise ValueError("Equal command tapes changed physics")
        if t is not None and digest_json([e for e in old["events"] if e["step"] < t]) != digest_json([e for e in new["events"] if e["step"] < t]):
            raise ValueError("Physical delivery diverged before the first differing command")
        for count, row in zip(stats, (old, new), strict=True):
            u = np.asarray(row["actions"])
            if not np.isfinite(u).all() or np.abs(u).max() > .8 + 1e-12:
                raise ValueError("Out-of-range command on physical tape")
            count["joint_ticks"] += len(u)
            count["preview_ticks"] += sum(e["effect_ticks"] for e in row["episodes"])
            count["learner_returns"] += row["score"]["learner"]
            count["alternating"] += row["score"]["returns"]
            count["updates"] += row["learner_updates"]
            count["peak"] = max(count["peak"], row["score"]["learner"])
            for leg, move in enumerate(row["placements"]["requested"]):
                if move["target"] != list(draw(c, leg, "random", None)[0]):
                    raise ValueError("Independent broad random draw differs from tape")
            for e in row["episodes"]:
                delay = e["delay_ticks"]
                if phase == "train":
                    forced, arm = preview.exploration(i, e["offer"])
                    if forced != e["forced_exploration"] or (forced and preview.DELAYS[arm] != delay):
                        raise ValueError("Exploration assignment differs from frozen schedule")
                elif e["forced_exploration"] or row["learner_updates"]:
                    raise ValueError("Held games must not explore or update")
                count[f"delay_{delay}_offered"] += 1
                count["censored"] += e["earned"] is None
                if e["earned"] is not None:
                    count[f"delay_{delay}_physical_outcomes"] += 1
                    count[f"delay_{delay}_legal_hit"] += e["earned"]
        first_hits = [next((e for e in item["events"] if e["kind"] == "contact" and e["side"] == 1), None)
                      for item in (old, new)]
        gap = float(np.linalg.norm(np.asarray(first_hits[0]["location"]) - first_hits[1]["location"])) if all(first_hits) else None
        pairs.append(dict(id=c["id"], first_action_diff=t, common_random_draws=min(len(p), len(q)),
                          learner_returns=[old["score"]["learner"], new["score"]["learner"]],
                          first_arm_contact_offset_m=gap,
                          control_event_sha256=digest_json(old["events"]),
                          treatment_event_sha256=digest_json(new["events"])))
    return {"pairs": pairs, "order": ["control", "treatment"], "totals": stats}


def report(a):
    trained, base = _read(a.treatment), _read(a.control)
    if a.archived_late:
        base = archived_late(a, trained, base)
    if trained["plan_sha256"] != base["plan_sha256"]:
        raise ValueError("Compared branches used different frozen plans")
    rows = {}
    for phase in ("train", "held"):
        n = min(len(trained["records"][phase]), len(base["records"][phase]))
        rows[phase] = phase_report(trained["records"][phase][:n], base["records"][phase][:n], phase)
    result = {
        "plan_sha256": base["plan_sha256"],
        "control_sha256": _hash(a.control), "treatment_sha256": _hash(a.treatment),
        "complete": base["complete"] and trained["complete"],
        "archived_late_control": a.archived_late,
        "records": rows,
    }
    if a.replay_strongest and rows["held"]["pairs"]:
        pairs = rows["held"]["pairs"]
        changed = [i for i, rec in enumerate(pairs) if rec["first_action_diff"] is not None]
        ranked = changed or list(range(len(pairs)))
        i = max(ranked, key=lambda i: (pairs[i]["learner_returns"][1] - pairs[i]["learner_returns"][0],
                                       pairs[i]["learner_returns"][1]))
        result["native_witness"] = replay(trained["ready"], trained["records"]["held"][i])
    _write(a.out, result)


def archived_late(a, trained, base):
    plan = _read(a.plan)
    if (trained["kind"] != "transfer" or trained["plan_sha256"] != _hash(a.plan)
            or base["source_sha256"] != plan["bank_sha256"]
            or base["optical_sha256"] != plan["optics_sha256"]
            or base["carry_sha256"] != trained["late_carry_sha256"]
            or base["mode"] != "frozen" or base["variant"] != "elastic" or base["budget"] != 28):
        raise ValueError("Archived late control differs from the frozen transfer conditions")
    source = {c["id"]: (c, row) for c, row in zip(base["cards"], base["rows"])}
    rows = []
    for record in trained["records"]["held"]:
        q, data = source[record["card"]["id"]]
        if (q != record["card"] or data["score"]["returns"] >= plan["held_cap"]
                or data["memory"]["before"] != data["memory"]["after"]):
            raise ValueError("Archived card, nominal cap or memory differs")
        rows.append(dict(data, card=q, episodes=[], learner_updates=0))
    return dict(plan_sha256=_hash(a.plan), complete=True, records={"train": [], "held": rows})


def replay(ready, tape):
    mpc = preview.mpc
    c = tape["card"]
    with mpc.centered(ready):
        table = mpc.arena()
        shadow = mpc.arena(detail="cad")
        try:
            clear = CourtClearance(shadow.model, articulated=True)
            table.launch(tape["shot"], card=c)
            table.partner.reset(seed=c["seed"])
            guarded(table, clear)
            touches = []
            with contact_observer(table, touches):
                for u in tape["actions"]:
                    table.exchange(np.asarray(u), mpc.FAR_COMMAND)
            h = digest_json(table.events)
            expected = digest_json(tape["events"])
            score = preview.audit(table.events, opening_side=1)
            if table.launches != 1 or h != expected or digest_json(score) != digest_json(tape["score"]):
                diff = next(((i, a, b) for i, (a, b) in enumerate(zip(table.events, tape["events"])) if a != b), None)
                raise ValueError(f"Replay {c['id']} changed point or referee: {table.launches=} "
                                 f"events {len(table.events)} vs {len(tape['events'])}, "
                                 f"score {score['learner']} vs {tape['score']['learner']}, {diff=}")
            verify_clearance(table)
            return dict(id=c["id"], event_sha256=h, launches=1, native_racket_contacts=len(touches),
                        learner=tape["score"]["learner"], clearance=table.clearance_result())
        finally:
            table.close()
            shadow.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("treatment", type=Path)
    p.add_argument("control", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--replay-strongest", action="store_true")
    p.add_argument("--archived-late", action="store_true")
    p.add_argument("--plan", type=Path, default=preview.PLAN)
    report(p.parse_args())
