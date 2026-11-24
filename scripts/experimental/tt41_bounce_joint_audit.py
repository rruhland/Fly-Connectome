"""Audit full direct transfers and compare precontact optical forecasts on terminal touches."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BladeIK
from .tt36_impact import rgb_point
from .tt39_opportunity_map import projected_height
from .tt41_bounce_effect import forecast
from .tt41_bounce_joint import NOMINAL, OUT, SOLO, SRC


def arc(time, points, at):
    t = np.asarray(time) - at
    p = np.asarray(points)
    fit = np.c_[np.ones(len(t)), t]
    coeff = np.linalg.lstsq(fit, p, rcond=None)[0]
    vel = coeff[1]
    acc = np.array([0.0, 0.0, -9.81]) - 0.13 * np.linalg.norm(vel) * vel
    y = p - 0.5 * np.outer(t**2, acc)
    coeff = np.linalg.lstsq(fit, y, rcond=None)[0]
    err = float(np.sqrt(np.mean(np.sum((y - fit @ coeff) ** 2, axis=1))))
    return coeff[0], coeff[1], err


def spectrum(res, law):
    rows = []
    source = _read(SRC)
    started = process_time()
    with mpc.centered(source["ready"]):
        a = mpc.arena()
        s, rig, *_ = mpc.resources(a)
        k = BladeIK(a)
        try:
            for card, row in zip(source["cards"], source["rows"], strict=True):
                offer = row["placements"]["delivered"][-1]
                rx = offer["receive_line"]
                touch = offer["near_contact"]
                if (
                    rx is None
                    or touch is None
                    or not 0.99 <= projected_height(rx) < 1.04
                ):
                    continue
                time = touch["time"]
                first = max(0, int((time - 0.13) / mpc.DT))
                last = min(len(row["actions"]), round(time / mpc.DT) + 4)
                a.launch(row["shot"], card=card)
                a.partner.reset(seed=card["seed"])
                offsets = []
                positions = []
                q = []
                clocks = []
                for tick, u in enumerate(row["actions"]):
                    if first <= tick < last:
                        t = tick * mpc.DT
                        q.append(a.joints[:5].copy())
                        clocks.append(t)
                        if offer["near_bounce"]["time"] + 0.032 < t < time - 0.008:
                            p, _ = rgb_point(s.capture(), rig)
                            if p is not None:
                                offsets.append(t)
                                positions.append(p)
                    a.exchange(u, mpc.FAR_COMMAND)
                if digest_json(a.events) != digest_json(row["events"]):
                    raise ValueError("Archived direct arm did not reproduce")
                truth = next(
                    (
                        e["velocity"]
                        for e in a.events
                        if e["kind"] == "separation"
                        and e.get("collider") == "learner_racket"
                        and time <= e["time"] < time + 0.07
                    ),
                    None,
                )
                rec = {
                    "id": card["id"],
                    "projection_height_m": projected_height(rx),
                    "contact_time": time,
                    "positions": np.asarray(positions).tolist(),
                    "times": offsets,
                    "known_departure": truth,
                }
                if len(positions) >= 4 and truth is not None:
                    p, v, err = arc(offsets[-6:], positions[-6:], time)
                    dt = np.asarray(clocks) - time
                    before = (dt <= 0) & (dt > -0.065)
                    t = dt[before]
                    qs = np.asarray(q)[before]
                    fit = np.linalg.lstsq(
                        np.c_[np.ones(len(t)), t, t * t], qs, rcond=None
                    )[0]
                    center, n, j, _ = k.pose(fit[0], jacobian=True)
                    offset = p - center
                    if offset @ n < 0:
                        n = -n
                    offset -= n * (offset @ n)
                    velocity = j @ fit[1] + np.cross(
                        k.angular[:, k.dofs] @ fit[1], offset
                    )
                    estimates = {
                        m: forecast(
                            v,
                            velocity if m != "still" else np.zeros(3),
                            n,
                            law if m == "solo" else NOMINAL,
                        ).tolist()
                        for m in ("still", "elastic", "solo")
                    }
                    rec.update(
                        p=p.tolist(),
                        incoming=v.tolist(),
                        err=err,
                        normal=n.tolist(),
                        radius=float(np.linalg.norm(offset)),
                        blade_velocity=velocity.tolist(),
                        estimated=estimates,
                    )
                rows.append(rec)
                print(
                    card["id"],
                    len(positions),
                    rec.get("radius"),
                    rec.get("estimated"),
                    truth,
                    flush=True,
                )
        finally:
            s.close()
            a.close()
    stats = {}
    for m in ("still", "elastic", "solo"):
        err = np.array(
            [
                np.asarray(r["estimated"][m]) - r["known_departure"]
                for r in rows
                if "estimated" in r
            ]
        )
        if len(err):
            stats[m] = {
                "n": len(err),
                "median_absolute_xyz": np.median(abs(err), axis=0).tolist(),
                "median_3d": float(np.median(np.linalg.norm(err, axis=1))),
                "median_forward_up": float(
                    np.median(np.linalg.norm(err[:, 1:], axis=1))
                ),
            }
    _write(
        res,
        {
            "source_sha256": _hash(SRC),
            "effect_sha256": _hash(SOLO),
            "rows": rows,
            "summary": stats,
            "time_cpu_s": process_time() - started,
            "boundary": "Retrospective optical-and-encoder departure estimate before contact, aligned by evaluator contact time. Evaluator output is not a policy input.",
        },
    )


def compare(tape, output, *, cad=False):
    src = _read(SRC)
    d = _read(tape)
    results = []
    lookup = {c["id"]: r for c, r in zip(src["cards"], src["rows"], strict=True)}
    start = process_time()
    with mpc.centered(src["ready"]):
        a = mpc.arena()
        try:
            for r in d["rows"]:
                card = r["card"]
                cid = card["id"]
                fork = "fork_tick" in r
                ref = lookup[cid] if fork else r["modes"]["stock"]
                aim = r["fork_tick"] if fork else None
                base = {"id": cid, "modes": {}}
                for mode, q in r["modes"].items():
                    shot = ref["shot"]
                    limit = next(
                        (
                            i
                            for i, (x, y) in enumerate(
                                zip(ref["actions"], q["actions"])
                            )
                            if x != y
                        ),
                        None,
                    )
                    prefix = len(ref["actions"]) if limit is None else limit
                    if fork and prefix < aim:
                        raise ValueError(
                            "Branch changed earlier than the scheduled evaluation halt"
                        )
                    orig = [e for e in ref["events"] if e["step"] <= prefix]
                    recv = [e for e in q["events"] if e["step"] <= prefix]
                    if digest_json(orig) != digest_json(recv):
                        raise ValueError("Different pre-intervention events")
                    stamps = {}
                    if cad:
                        a.sample_clearance(live=True)
                        a.launch(shot, card=card)
                        a.partner.reset(seed=card["seed"])
                        for u in q["actions"]:
                            a.exchange(u, mpc.FAR_COMMAND)
                        if digest_json(a.events) != digest_json(q["events"]):
                            raise ValueError("CAD replay disagrees with physical game")
                        stamps = a.clearance_result()
                    base["modes"][mode] = {
                        "physical_returns": mpc.audit(q["events"], opening_side=1)[
                            "learner"
                        ],
                        "first_different_tick": limit,
                        "prefix_sha256": digest_json(orig),
                        "physics_sha256": digest_json(q["events"]),
                        "actions": len(q["actions"]),
                        "depart": q.get("depart"),
                        "contact": q.get("placement", {}).get("near_contact")
                        is not None
                        if fork
                        else None,
                        "answered": q.get("answered"),
                        "legal_repair": q.get("legal"),
                        "cpu_s": q.get("branch_cpu_s", q.get("runtime_cpu_s")),
                        "cad": stamps,
                    }
                results.append(base)
        finally:
            a.close()
    _write(
        output,
        {
            "source_sha256": _hash(tape),
            "cases": results,
            "cpu_s": process_time() - start,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("spectrum", "verify"))
    p.add_argument("output", type=Path)
    p.add_argument("--tape", type=Path, default=OUT / "live.json.gz")
    p.add_argument("--cad", action="store_true")
    c = p.parse_args()
    c.output.parent.mkdir(parents=True, exist_ok=True)
    if c.mode == "spectrum":
        spectrum(c.output, _read(SOLO)["pooled"])
    else:
        compare(c.tape, c.output, cad=c.cad)
