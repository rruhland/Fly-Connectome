"""Independent referee, actuator and CAD check of committed teacher action tapes."""

import argparse
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit

BASE = Path("artifacts/tt42/learning-12.json.gz")


def metrics(row):
    plans = row["teacher_searches"]
    accepted = [p for p in plans if p.get("committed")]
    scored = [p for p in accepted if p.get("physical_legal")]
    answered = [p for p in scored if p.get("answered")]
    pending = sum(p.get("capped_pending", False) for p in accepted)
    native = sum(p["virtual_legal"] for p in plans)
    total = sum(round_["trials"] for p in plans for round_ in p["rounds"])
    future_hits = []
    landing_error = []
    departure = []
    for q in accepted:
        k = q["tick"]
        ran = q["executed_ticks"]
        past = q["planned_actions"]
        length = min(len(past), ran)
        if not length or row["actions"][k : k + length] != past[:length]:
            raise ValueError("A new choice interrupted a committed native prefix")
        if ran > length and np.any(
            np.asarray(row["actions"][k + length : k + ran]) != 0.0
        ):
            raise ValueError(
                "An unexpected additional stroke interrupted the planned hold"
            )
        if not q.get("physical_legal"):
            continue
        hit = next(
            e
            for e in row["events"]
            if e["kind"] == "contact"
            and e["side"] == 1
            and k < e["step"] <= q["landing"]["step"]
        )
        sep = next(
            (
                e
                for e in row["events"]
                if e["kind"] == "separation"
                and e.get("collider") == "learner_racket"
                and hit["time"] < e["time"] < hit["time"] + 0.07
            ),
            None,
        )
        native_hit = q["selection"]["native"][1] + (k * mpc.DT)
        native_land = q["selection"]["native"][2] + (k * mpc.DT)
        future_hits.append(abs(hit["time"] - native_hit) * 1000.0)
        landing_error.append(abs(q["landing"]["time"] - native_land) * 1000.0)
        if sep is not None:
            departure.append(
                float(
                    np.linalg.norm(
                        np.asarray(sep["velocity"]) - q["selection"]["departure"]
                    )
                )
            )
    return {
        "searches": len(plans),
        "attempted_windows_with_nominal_legal": sum(
            p["virtual_legal"] > 0 for p in plans
        ),
        "native_candidates": total,
        "legal_native_candidates": native,
        "distinct_successful_rates": sum(p.get("unique_legal", 0) for p in plans),
        "committed": len(accepted),
        "committed_resolved": len(accepted) - pending,
        "committed_censored": pending,
        "actual_legal": len(scored),
        "actual_answered": len(answered),
        "selected_modes": dict(
            Counter(p["selection"]["family"].split("-centered")[0] for p in accepted)
        ),
        "physical_legal_modes": dict(
            Counter(p["selection"]["family"].split("-centered")[0] for p in scored)
        ),
        "median_abs_contact_clock_error_ms": float(np.median(future_hits))
        if future_hits
        else None,
        "median_abs_landing_error_ms": float(np.median(landing_error))
        if landing_error
        else None,
        "median_departure_error_m_s": float(np.median(departure))
        if departure
        else None,
        "times_waited_cpu_s": float(sum(p["cpu_s"] for p in plans)),
    }


def run(source, output, *, only=None, full_cad=True, stage=False):
    data = _read(source)
    old = _read(BASE)
    started = process_time()
    anchored = {
        (c["id"], c["seed"], c["reply_seed"]): r
        for c, r in zip(old["cards"][: len(old["rows"])], old["rows"], strict=True)
    }
    if not stage and not data.get("complete"):
        raise ValueError("Refuse to report an unfinished teacher cohort as complete")
    rows = []
    with mpc.centered(data["ready"]):
        a = mpc.arena()
        try:
            for c, r in zip(
                data["cards"][: len(data["rows"])], data["rows"], strict=True
            ):
                if only is not None and c["id"] not in only:
                    continue
                stats = metrics(r)
                a.sample_clearance(live=full_cad)
                a.launch(r["shot"], card=c)
                a.partner.reset(seed=c["seed"])
                for u in r["actions"]:
                    if max(abs(float(x)) for x in u) > 0.8000000001:
                        raise ValueError(
                            "Teacher exceeded the stock normalized target-rate bound"
                        )
                    a.exchange(u, mpc.FAR_COMMAND)
                if digest_json(a.events) != digest_json(r["events"]) or digest_json(
                    audit(a.events, opening_side=1)
                ) != digest_json(r["score"]):
                    raise ValueError(
                        "Teacher referee result differs from independent replay"
                    )
                requests = r["placements"]["requested"]
                if any(o["draw"] != "random" for o in requests) or len(
                    {tuple(o["target"]) for o in requests}
                ) != len(requests):
                    raise ValueError(
                        "Request sequence is not the unmodified continuous opponent"
                    )
                k = (c["id"], c["seed"], c["reply_seed"])
                matched = anchored.get(k)
                ref = None
                if matched is not None:
                    target = lambda o: [
                        (e["draw"], e["target"]) for e in o["placements"]["requested"]
                    ]
                    before, after = target(matched), target(r)
                    start = min(len(before), len(after))
                    left, right = (
                        s["placements"]["delivered"][0] for s in (matched, r)
                    )
                    ref = {
                        "control_returns": matched["score"]["learner"],
                        "same_aim_draws": before[:start] == after[:start],
                        "same_open_far_contact": digest_json(left["far_contact"])
                        == digest_json(right["far_contact"]),
                        "same_open_near_bounce": digest_json(left["near_bounce"])
                        == digest_json(right["near_bounce"]),
                    }
                    if not all(
                        ref[t]
                        for t in (
                            "same_aim_draws",
                            "same_open_far_contact",
                            "same_open_near_bounce",
                        )
                    ):
                        raise ValueError(
                            "The matched fixture or incoming delivery changed"
                        )
                stats.update(
                    id=c["id"],
                    own=r["score"]["learner"],
                    opponent=r["score"]["opponent"],
                    alternating=r["score"]["returns"],
                    ticks=len(r["actions"]),
                    cpu_s=r["cpu_s"],
                    physical_s=r["physical_time_s"],
                    launches=r["launches"],
                    end=a.reason,
                    guard=a.clearance_result(),
                    score_and_referee_exact=True,
                    action_sha256=digest_json(r["actions"]),
                    event_sha256=digest_json(a.events),
                    matched_control=ref,
                )
                rows.append(stats)
                print(c["id"], stats, flush=True)
        finally:
            a.close()
    _write(
        output,
        {
            "source_sha256": _hash(source),
            "controlled_stock_sha256": _hash(BASE),
            "rows": rows,
            "audit_cpu_s": process_time() - started,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--ids", nargs="+", type=int)
    p.add_argument("--no-cad", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    run(a.source, a.out, only=a.ids, full_cad=not a.no_cad, stage=a.allow_partial)
