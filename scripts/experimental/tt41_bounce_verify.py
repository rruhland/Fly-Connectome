"""Compare matched first intervention and independently replay a scored course tape."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena
from .tt40_low_lift import SCHOOL
from .tt41_bounce_effect import OUT


def match(stock, changed):
    first = next(
        (
            i
            for i, (x, y) in enumerate(zip(stock["actions"], changed["actions"]))
            if x != y
        ),
        None,
    )
    if first is None:
        if (
            stock["actions"] != changed["actions"]
            or stock["events"] != changed["events"]
        ):
            raise ValueError("Equal commands stopped at unequal physical histories")
        return {"first_changed_tick": None}
    same_events = lambda row: [e for e in row["events"] if e["step"] <= first]
    if digest_json(same_events(stock)) != digest_json(same_events(changed)):
        raise ValueError("The matched pre-intervention physical history differs")
    begin = first / 60
    locate = lambda row: next(
        (
            i
            for i, x in enumerate(row["placements"]["delivered"])
            if x["far_contact"]["time"] < begin
            and (not x.get("near_contact") or x["near_contact"]["time"] > begin)
        ),
        None,
    )
    i = locate(stock)
    if i is None or i != locate(changed):
        raise ValueError(
            "The first new action occurs outside a shared received delivery"
        )
    d0 = stock["placements"]["delivered"][i]
    d1 = changed["placements"]["delivered"][i]
    for key in ("far_contact", "near_bounce", "receive_line"):
        if d0[key] != d1[key]:
            raise ValueError(f"The first different command did not face the same {key}")
    lim = stock["placements"]["requested"][: i + 1]
    if lim != changed["placements"]["requested"][: i + 1]:
        raise ValueError("The opponent changed the placement request before the action")

    def departure(row, leg):
        touch = leg["near_contact"]
        if touch is None:
            return None
        return next(
            (
                e["velocity"]
                for e in row["events"]
                if e["kind"] == "separation"
                and e.get("collider") == "learner_racket"
                and touch["time"] <= e["time"] < touch["time"] + 0.065
            ),
            None,
        )

    obs = lambda row: [
        r["impact_margin"]
        for r in row["records"]
        if r["tick"] == first and "impact_margin" in r
    ]
    return {
        "first_changed_tick": first,
        "delivery_index": i,
        "far_contact": d0["far_contact"],
        "near_bounce": d0["near_bounce"],
        "prefix_action_sha256": digest_json(stock["actions"][:first]),
        "prefix_event_sha256": digest_json(same_events(stock)),
        "requested_sha256": digest_json(lim),
        "stock_legal": bool(d0["near_landed"]),
        "changed_legal": bool(d1["near_landed"]),
        "changed_answered": bool(
            len(changed["placements"]["delivered"]) > i + 1
            and changed["placements"]["delivered"][i + 1]["near_bounce"]
        ),
        "stock_departure_m_s": departure(stock, d0),
        "changed_departure_m_s": departure(changed, d1),
        "observation": obs(changed),
    }


def recheck(ready, card, row):
    with centered(ready):
        a = arena()
        try:
            a.sample_clearance(live=True)
            a.launch(row["shot"], card=card)
            a.partner.reset(seed=card["seed"])
            for u in row["actions"]:
                a.exchange(u, FAR_COMMAND)
            if digest_json(a.events) != digest_json(row["events"]):
                raise ValueError("Native CAD shadow reproduced different events")
            if digest_json(audit(a.events, opening_side=1)) != digest_json(
                row["score"]
            ):
                raise ValueError("Independent point score differs")
            return {
                "events_sha256": digest_json(a.events),
                "command_sha256": digest_json(row["actions"]),
                "score": row["score"],
                "ticks": len(row["actions"]),
                "cad": a.clearance_result(),
                "motor": a.motor_audit(),
            }
        finally:
            a.close()


def verify(tape, output, cad):
    source = _read(tape)
    ready = _read(SCHOOL)["ready"]
    modes = source["modes"]
    if "fixed" not in modes:
        raise ValueError("Need the matched frozen stock control")
    out = {
        "schema": "tt41-bounce-restored-matched-proof-v1",
        "source_sha256": _hash(tape),
        "assigned": len(source["cards"]),
        "completed": len(source["rows"]),
        "modes": modes,
        "rows": [],
        "compute": {},
    }
    scores = {m: [] for m in modes}
    for r in source["rows"]:
        trials = r["results"]
        c = r["card"]
        orig = trials["fixed"]
        row = {"release": c["id"], "variants": {}}
        for mode in modes:
            tr = trials[mode]
            count = tr["score"]["learner"]
            scores[mode].append(count)
            if audit(tr["events"], opening_side=1) != tr["score"]:
                raise ValueError("Tape score disagrees with actual referee")
            row["variants"][mode] = dict(
                own_legal=count,
                opponent_legal=tr["score"]["opponent"],
                simulated_s=tr["simulation_s"],
                cpu_s=tr["cpu_s"],
                **match(orig, tr),
            )
            if (
                cad
                and c["id"] == source["rows"][0]["card"]["id"]
                and mode in ("fixed", "cold", "solo")
            ):
                row["variants"][mode]["recheck"] = recheck(ready, c, tr)
        out["rows"].append(row)
    for mode, s in scores.items():
        sec = sum(r["variants"][mode]["cpu_s"] for r in out["rows"])
        sim = sum(r["variants"][mode]["simulated_s"] for r in out["rows"])
        out["compute"][mode] = {
            "total_returns": sum(s),
            "max_returns": max(s, default=0),
            "actual_cpu_s": sec,
            "physical_time_s": sim,
            "legal_returns_per_cpu_minute": 60 * sum(s) / max(0.01, sec),
        }
    _write(output, out)
    print(out["compute"], flush=True)


def verify_supplement(tape, supplemental, output, cad):
    """A separately executed control must face the same preassigned rollout."""
    a, b = (_read(path) for path in (tape, supplemental))
    for key in ("plan_sha256", "group", "batch", "rounds", "cap", "law_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"Assessment supplements differ in {key}")
    ready = _read(SCHOOL)["ready"]
    rows = []
    for test in b["rows"]:
        card = test["card"]
        anchor = next(row for row in a["rows"] if row["card"] == card)
        for mode, tr in test["results"].items():
            r = {
                "mode": mode,
                "release": card["id"],
                "own_legal": tr["score"]["learner"],
                "physical_time_s": tr["simulation_s"],
                "cpu_s": tr["cpu_s"],
                **match(anchor["results"]["fixed"], tr),
            }
            if cad:
                r["recheck"] = recheck(ready, card, tr)
            rows.append(r)
    _write(
        output,
        {
            "matched_sha256": _hash(tape),
            "supplement_sha256": _hash(supplemental),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tape", type=Path, default=OUT / "held.json.gz")
    p.add_argument("--out", type=Path, default=OUT / "held-audit.json")
    p.add_argument("--cad", action="store_true")
    p.add_argument("--supplement", type=Path)
    q = p.parse_args()
    if q.supplement:
        verify_supplement(q.tape, q.supplement, q.out, q.cad)
    else:
        verify(q.tape, q.out, q.cad)
