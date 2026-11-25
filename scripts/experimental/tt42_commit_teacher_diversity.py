"""Physically replay diverse teacher movements on the exact same hard received ball."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt33_joint import FAR_COMMAND
from .tt40_low_feasibility import replay_prefix
from .tt42_commit_teacher import OUT, PRACTICE, SOURCE, ZERO, Teacher


def signature(plan):
    values = plan["native"]
    v = plan["departure"]
    return np.r_[
        values[1] / 0.042,
        values[3] / 0.16,
        values[4] / 0.23,
        np.asarray(v) / [0.55, 0.65, 0.60],
    ]


def diverse(plans, maximum):
    if not plans:
        return []
    sort = sorted(plans, key=lambda p: p["score"])
    ideas = [
        p for p in sort if p["answerability"] < max(7.2, sort[0]["answerability"] + 2.5)
    ]
    order = [ideas.pop(0)]
    while ideas and len(order) < maximum:
        distance = lambda p: min(
            np.linalg.norm(signature(p) - signature(x)) for x in order
        )
        j = int(np.argmax([distance(p) for p in ideas]))
        p = ideas.pop(j)
        if distance(p) < 0.36:
            break
        order.append(p)
    return order


def practice(a, teacher, card, old, tick, leg, plan):
    replay_prefix(a, old, card, tick)
    before = len(a.events)
    rates = plan["rates"]
    prior = old["actions"][:tick]
    actions = list(prior)
    legal = False
    for j in range(190):
        u = teacher.recover() if legal else rates[j] if j < len(rates) else ZERO
        _, _, done, cap, e = a.exchange(u, FAR_COMMAND)
        actions.append(np.asarray(u).tolist())
        legal = legal or any(
            p["kind"] == "legal_return" and p["side"] == 1 for p in e["events"]
        )
        if done or cap or a.rival_returns > leg + 1:
            break
    p = a.point_placements()
    q = p["delivered"][leg]
    original = old["placements"]["delivered"][leg]
    if any(
        digest_json(q[x]) != digest_json(original[x])
        for x in ("far_contact", "near_bounce")
    ):
        raise ValueError(
            "Physical matched preparation received a changed opponent ball"
        )
    if q["near_landed"] != legal:
        raise ValueError("Actual point referee disagrees with anchored landing")
    prior_targets = [o["target"] for o in old["placements"]["requested"][: leg + 1]]
    targets = [o["target"] for o in p["requested"][: leg + 1]]
    if targets != prior_targets or any(o["draw"] != "random" for o in p["requested"]):
        raise ValueError("An evaluation used a different realized opponent target")
    actual_hit = q["near_contact"]
    separation = next(
        (
            e
            for e in a.events[before:]
            if e["kind"] == "separation" and e.get("collider") == "learner_racket"
        ),
        None,
    )
    return {
        "family": plan["family"],
        "native": plan["native"],
        "native_departure": plan["departure"],
        "predicted_answerability": plan["answerability"],
        "legal": legal,
        "answered": a.rival_returns > leg + 1,
        "touch": actual_hit,
        "physical_separation": separation,
        "actions": actions,
        "events": a.events,
        "placements": p,
        "score": audit(a.events, opening_side=1),
        "physics_clearance": a.clearance_result(),
        "motor": a.motor_audit(),
    }


def run(args):
    d = _read(args.source)
    i, c = next((i, c) for i, c in enumerate(d["cards"]) if c["id"] == args.id)
    old = d["rows"][i]
    leg = len(old["placements"]["delivered"]) - 1
    point = old["placements"]["delivered"][leg]
    event = next(
        e
        for e in old["events"]
        if e["kind"] == "separation"
        and e.get("collider") == "rival_racket"
        and point["far_contact"]["time"] <= e["time"]
    )
    tick = int(np.ceil((event["time"] + 1e-8) / mpc.DT))
    report = {
        "role": "training-only physically replayed same-offer alternatives",
        "source_sha256": _hash(args.source),
        "ready": d["ready"],
        "card": c,
        "shot": old["shot"],
        "leg_index": leg,
        "fork_tick": tick,
        "variants": [],
        "original_fault": old["events"][-1].get("reason"),
        "same_recorded_prefix_sha256": digest_json(old["actions"][:tick]),
        "original_returned": bool(point["near_landed"]),
        "batch": args.batch,
        "rounds": args.rounds,
        "max_actual": args.max,
        "independent_native_cad_and_referee_confirmation": True,
    }
    with mpc.centered(d["ready"]):
        a = mpc.arena()
        t = Teacher(
            a,
            d["ready"],
            args.batch,
            args.rounds,
            PRACTICE,
            stop_quality=-1.0,
            fragments_only=False,
        )
        try:
            a.sample_clearance(live=False)
            replay_prefix(a, old, c, tick)
            _, log = t.search()
            options = diverse(t.last_valid, args.max)
            report["proposal_screen"] = log
            report["physical_proposals"] = len(options)
            for q in options:
                a.sample_clearance(live=True)
                z = practice(a, t, c, old, tick, leg, q)
                report["variants"].append(z)
                _write(args.out, report)
                print(
                    c["id"],
                    q["family"],
                    z["legal"],
                    z["answered"],
                    q["native"][1],
                    q["native"][3:5],
                    z["physics_clearance"]["penetration_samples"],
                    flush=True,
                )
        finally:
            a.close()
    report["complete"] = True
    _write(args.out, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "out", type=Path, nargs="?", default=OUT / "same-offer-diversity.json.gz"
    )
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--id", type=int, default=8519)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--max", type=int, default=5)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = process_time()
    run(args)
    print("time_cpu_s", process_time() - t0, flush=True)
