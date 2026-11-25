"""Compare each sequential student arm against the same regulation game releases."""

import argparse
from ast import literal_eval
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt42_commit_teacher import OUT

BASE_CONDITIONS = ("stock", "pilot", "all")


def draw_list(placements):
    return [(o["draw"], o["target"]) for o in placements["requested"]]


def same_received(a, b):
    return all(
        digest_json(a[k]) == digest_json(b[k]) for k in ("far_contact", "near_bounce")
    )


def actual_departure(row, ball):
    contact = ball.get("near_contact")
    if contact is None:
        return None
    return next(
        (
            e["velocity"]
            for e in row["events"]
            if e["kind"] == "separation"
            and e.get("collider") == "learner_racket"
            and contact["time"] < e["time"] < contact["time"] + 0.07
        ),
        None,
    )


def matched_repairs(stock, treatment):
    shared = []
    for i, (a, b) in enumerate(
        zip(stock["placements"]["delivered"], treatment["placements"]["delivered"])
    ):
        if not same_received(a, b):
            break
        shared.append((i, a, b))
    repairs = []
    for i, a, b in shared:
        if a["near_landed"] or not b["near_landed"]:
            continue
        before, after = actual_departure(stock, a), actual_departure(treatment, b)
        landing = next(
            (
                e["location"]
                for e in treatment["events"]
                if e["kind"] == "legal_return"
                and e["side"] == 1
                and e["time"] > b["near_contact"]["time"]
            ),
            None,
        )
        repairs.append(
            {
                "legal_offer_index": i,
                "same_far_contact_and_near_bounce": True,
                "incoming_receive_line": a["receive_line"],
                "stock_departure_eval_only_mps": before,
                "student_departure_eval_only_mps": after,
                "outgoing_effect_eval_only_mps": (np.asarray(after) - before).tolist()
                if before is not None and after is not None
                else None,
                "stock_terminal_fault": stock["events"][-1].get("reason"),
                "student_legal_landing_eval_only_m": landing,
            }
        )
    return len(shared), repairs


def verify(out, arm, *, cad, baseline="stock", conditions=None):
    started = process_time()
    docs = {name: _read(path) for name, path in arm.items()}
    stock = docs[baseline]
    expected = conditions or {name: name for name in arm}
    if any(
        d["condition"] != expected[name]
        or not d.get("complete")
        or len(d["rows"]) != len(d["cards"])
        or d["cards"] != stock["cards"]
        or d["ready"] != stock["ready"]
        or d["cap"] != stock["cap"]
        or d["budget"] != stock["budget"]
        or {
            key: val
            for key, val in d["source_sha256"].items()
            if "earned-camera" not in key
        }
        != {
            key: val
            for key, val in stock["source_sha256"].items()
            if "earned-camera" not in key
        }
        for name, d in docs.items()
    ):
        raise ValueError("Use completed, matched course tapes")
    if (
        len({d["rows"][0]["memory"]["before"]["state_sha256"] for d in docs.values()})
        != 1
    ):
        raise ValueError("Different initial own learned-memory samples")
    if "frozen" in docs and (
        docs["frozen"].get("own_updates") is not False
        or docs["updating"].get("own_updates") is not True
        or any(
            r["memory"]["before"] != r["memory"]["after"]
            or r["miss_memory"]["before"] != r["miss_memory"]["after"]
            for r in docs["frozen"]["rows"]
        )
    ):
        raise ValueError("Frozen arm changed its own learned history")
    rows = []
    with mpc.centered(stock["ready"]):
        arena = mpc.arena(protocol="random")
        try:
            for i, c in enumerate(stock["cards"]):
                r = stock["rows"][i]
                base = r["placements"]
                trials = {}
                for name, d in docs.items():
                    row = d["rows"][i]
                    p = row["placements"]
                    if digest_json(row["shot"]) != digest_json(r["shot"]):
                        raise ValueError("Changed opening physical coordinates")
                    x, y = draw_list(base), draw_list(p)
                    count = min(len(x), len(y))
                    if x[:count] != y[:count] or not all(s == "random" for s, _ in y):
                        raise ValueError(
                            "Changed common sequence of random opponent choices"
                        )
                    aa, bb = base["delivered"], p["delivered"]
                    if bool(aa) != bool(bb) or (aa and not same_received(aa[0], bb[0])):
                        raise ValueError(
                            "Changed first physical far contact/near bounce"
                        )
                    common_received, repairs = matched_repairs(r, row)
                    first = next(
                        (
                            t
                            for t, (a, b) in enumerate(
                                zip(r["actions"], row["actions"])
                            )
                            if a != b
                        ),
                        None,
                    )
                    prefix = len(r["actions"]) if first is None else first

                    def events_before(s, end=prefix):
                        return [e for e in s["events"] if e["step"] <= end]

                    if digest_json(events_before(r)) != digest_json(events_before(row)):
                        raise ValueError(
                            "Body or ball outcome changed before the issued commands"
                        )
                    if np.max(np.abs(row["actions"])) > 0.8000000001:
                        raise ValueError("Action rate limit violated")
                    arena.sample_clearance(live=cad)
                    arena.launch(row["shot"], card=c)
                    arena.partner.reset(seed=c["seed"])
                    for action in row["actions"]:
                        arena.exchange(action, mpc.FAR_COMMAND)
                    if digest_json(arena.events) != digest_json(row["events"]):
                        raise ValueError(
                            "Independent physical referee did not reproduce"
                        )
                    score = mpc.audit(arena.events, opening_side=1)
                    if digest_json(score) != digest_json(row["score"]):
                        raise ValueError("Independent away landing count changed")
                    logs = row["recall_attempts"]
                    options = Counter(e[0] for s in logs for e in s["query"])
                    admitted_source = Counter(
                        int(literal_eval(e[1])[0])
                        for s in logs
                        for e in s["query"]
                        if e[0] == "admitted"
                    )
                    n = sum(s["substituted"] for s in logs)
                    if (
                        n != len(row["teacher_admissions"])
                        or options["admitted"] != n
                        or any(s["substituted"] > 2 for s in logs)
                    ):
                        raise ValueError(
                            "Teacher search slots or logged admissions changed"
                        )
                    limits = arena.clearance_result()
                    if limits["penetration_samples"]:
                        raise ValueError(
                            "CAD replay detected a physical collision penetration"
                        )
                    trials[name] = {
                        "own": score["learner"],
                        "opponent": score["opponent"],
                        "alternating": score["returns"],
                        "recorded_ticks": len(row["actions"]),
                        "first_action_difference": first,
                        "matched_aim_draw_prefix": count,
                        "same_first_incoming": True if aa else None,
                        "common_physically_delivered_offers": common_received,
                        "same_incoming_repairs": repairs,
                        "terminated": arena.terminated,
                        "physical_s": float(arena.data.time),
                        "cpu_s": row["runtime_cpu_s"],
                        "exact_referee_replay": True,
                        "cad": limits,
                        "lookups": len(logs),
                        "teacher_lookups": dict(options),
                        "candidate_admissions": n,
                        "candidate_admissions_by_teacher_point": dict(admitted_source),
                        "stock_declines": sum(
                            len(s["stock_miss_declines"]) for s in logs
                        ),
                        "own_hit_updates": row["planner"].get("updates", 0),
                        "own_hit_fragments": row["planner"].get(
                            "credited_fragments", 0
                        ),
                        "own_miss_updates": row["planner"].get(
                            "physical_miss_updates", 0
                        ),
                        "own_miss_fragments": row["planner"].get(
                            "observed_miss_fragments", 0
                        ),
                    }
                rows.append({"release": c["id"], "conditions": trials})
                print(
                    c["id"],
                    {
                        k: (
                            v["own"],
                            v["candidate_admissions"],
                            v["own_miss_fragments"],
                        )
                        for k, v in trials.items()
                    },
                    flush=True,
                )
        finally:
            arena.close()
    _write(
        out,
        {
            "arms_sha256": {k: _hash(v) for k, v in arm.items()},
            "control": baseline,
            "expanded_bank_retained_exact_pilot_samples": compare_banks(),
            "rows": rows,
            "time_cpu_s": process_time() - started,
        },
    )


def compare_banks():
    prior = OUT / "earned-camera.npz"
    expanded = OUT / "earned-camera-41.npz"
    with np.load(prior) as p, np.load(expanded) as e:
        if set(p.files) != set(e.files) or any(
            not np.array_equal(p[k], e[k][: len(p[k])]) for k in p.files
        ):
            raise ValueError("Expanded bank changed the existing pilot labels")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--cad", action="store_true")
    p.add_argument(
        "--comparison", choices=("base", "learning", "coverage"), default="base"
    )
    a = p.parse_args()
    if a.comparison == "learning":
        names = {"frozen": "fresh-frozen-pilot", "updating": "fresh-live-pilot"}
        verify(
            a.out or OUT / "fresh-pilot-audit.json",
            {k: OUT / f"{v}.json.gz" for k, v in names.items()},
            cad=a.cad,
            baseline="frozen",
            conditions={"frozen": "pilot", "updating": "pilot"},
        )
    elif a.comparison == "coverage":
        names = {"all": "coverage-nearest-control", "coverage": "coverage-diverse"}
        verify(
            a.out or OUT / "coverage-audit.json",
            {k: OUT / f"{v}.json.gz" for k, v in names.items()},
            cad=a.cad,
            baseline="all",
        )
    else:
        verify(
            a.out or OUT / "course-audit.json",
            {name: OUT / f"course-{name}.json.gz" for name in BASE_CONDITIONS},
            cad=a.cad,
        )
