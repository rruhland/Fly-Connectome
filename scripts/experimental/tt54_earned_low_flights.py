"""Recover successful low-receiving movements from ordinary RGB-controlled play."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt46_continuous_season as season
from . import tt48_practice_transfer as transfer
from . import tt51_low_reverse_curriculum as school
from .tt19_view_full_game import digest_json
from .tt39_opportunity_map import projected_height
from .tt4_transfer_trial import _hash, _read, _write

mpc = season.course.mpc
ROOT = Path("artifacts/tt54_earned_low")
PLAN = ROOT / "sources.json"


def optical_position(notes):
    notes = [n for n in notes if n.get("bounced") and n.get("p") is not None]
    for b in reversed(notes):
        for a in reversed(notes):
            dt = (b["tick"] - a["tick"]) / 60
            if not 0.04 < dt < 0.18:
                continue
            p, q = np.asarray(a["p"]), np.asarray(b["p"])
            v = (q - p) / dt
            if v[1] > -1.4 or not -1.56 < q[1] < -0.85:
                continue
            eta = (-1.53 - q[1]) / (0.98 * v[1])
            return (
                float(q[0] + eta * v[0]),
                float(q[2] + eta * (v[2] - 4.905 * dt) - 4.905 * eta**2),
            )
    return None


def optical_height(notes):
    p = optical_position(notes)
    return None if p is None else p[1]


def plan(region, plan_path):
    reserved = {
        c["id"] for k in ("practice", "assessment") for c in _read(season.PLAN)[k]
    }
    reserved.update(c["id"] for c in _read(transfer.PLAN)["practice"])
    inventory = _read(Path("artifacts/tt52_matched_movements/pairs.json.gz"))["sources"]
    by_release = {}
    seen = set()
    counts = Counter()
    for path, h in inventory.items():
        p = Path(path)
        doc = _read(p)
        if _hash(p) != h:
            raise ValueError(f"Changed source {path}")
        for j, (c, r) in enumerate(zip(doc["cards"], doc["rows"])):
            if c["id"] in reserved or c.get("profile") != "reach":
                continue
            if any(z["draw"] != "random" for z in r["placements"]["requested"]):
                continue
            own = [
                e for e in r["events"] if e["kind"] == "legal_return" and e["side"] == 1
            ]
            offers = [o for o in r["placements"]["delivered"] if o.get("near_bounce")]
            # The first far-side legal bounce belongs to the externally
            # measured opening. It predates the first opponent delivery.
            if not offers:
                continue
            own = [e for e in own if e["step"] > offers[0]["near_bounce"]["step"]]
            if len(own) != r["score"]["learner"]:
                raise ValueError(
                    f"Receipt, reward count disagreement: {p}, {c}, {len(own)}, {r['score']['learner']}"
                )
            for e, o, following in zip(own, offers, offers[1:]):
                leg = o["leg"]
                if (
                    not o["near_landed"]
                    or not following["near_bounce"]
                    or not o.get("receive_line")
                ):
                    continue
                # The lesson begins at a scalar reward. The later physical
                # landing gates storage. Only optical readings rank difficulty.
                start = own[leg - 1]["step"] if leg else 0
                near_score = o["near_bounce"]["step"]
                notes = [
                    n for n in r["searches"] if near_score - 1 <= n["tick"] < e["step"]
                ]
                pos = optical_position(notes)
                if pos is None:
                    continue
                x, z = pos
                k = (c["id"], c["seed"], c["reply_seed"], start, digest_json(notes))
                if k in seen:
                    continue
                seen.add(k)
                counts["answered"] += 1
                if region == "central" and abs(x) >= 0.10:
                    continue
                if z >= (1.045 if region == "central" else 1.02):
                    continue
                counts["optically_low"] += 1
                item = dict(
                    source=path,
                    source_sha256=h,
                    index=j,
                    leg=leg,
                    release=c["id"],
                    start=start,
                    end=e["step"] + 1,
                    optical_z=z,
                    optical_x=x,
                    evaluator_z=projected_height(o["receive_line"]),
                    evaluator_x=o["receive_line"]["xyz"][0],
                )
                group = by_release.setdefault(c["id"], [])
                group.append(item)
    picked = [min(group, key=lambda r: r["optical_z"]) for group in by_release.values()]
    picked.sort(key=lambda r: r["optical_z"])
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        plan_path,
        dict(
            region=region,
            reserved=sorted(reserved),
            counts=counts,
            possible=len(picked),
            rows=picked[:40],
            selection="One lowest camera-estimated answered shot per prior release, 40 lowest releases. Native height and x recorded only after selection. All official releases excluded.",
        ),
    )


def export(plan_path, target):
    p = _read(plan_path)
    result = dict(plan=_hash(plan_path), rows=[], rejected=[])
    # Replay through the legal reward. A full-prefix RGB history establishes
    # the same camera dependencies a player has during a normal point.
    for case in p["rows"]:
        source = Path(case["source"])
        if _hash(source) != case["source_sha256"]:
            raise ValueError("Source changed")
        doc = _read(source)
        c, r = doc["cards"][case["index"]], doc["rows"][case["index"]]
        with mpc.centered(doc["ready"]):
            arena = mpc.arena()
            camera, rig, stride, lo, hi = mpc.resources(arena)
            ik = mpc.BladeIK(arena)
            try:
                arena.launch(r["shot"], card=c)
                arena.partner.reset(seed=c["seed"])
                eye = mpc.Vision(rig, arena.control_dt, raw=True)
                eye.eye = mpc.LearnedOptics(eye.eye, season.course.OPTIC)
                ref = np.asarray(doc["ready"])
                record = []
                for t, u in enumerate(r["actions"][: case["end"]]):
                    q = arena.joints.copy()
                    rgb = camera.capture() if t % 2 == 0 else None
                    x = mpc.observed(eye, t, rgb, q)
                    sent = np.clip(ref + np.asarray(u) * stride, lo, hi)
                    if t >= case["start"]:
                        record.append(
                            dict(
                                tick=t,
                                x=None if x is None else x.copy(),
                                q=q,
                                ref=ref.copy(),
                                next_ref=sent.copy(),
                            )
                        )
                    arena.exchange(u, mpc.FAR_COMMAND)
                    ref = sent
                expected = [e for e in r["events"] if e["step"] <= case["end"]]
                if digest_json(arena.events) != digest_json(expected):
                    result["rejected"].append(
                        dict(case=case, reason="native prefix differs")
                    )
                else:
                    learned = school.extract(
                        record,
                        ik,
                        f"own-{case['release']}-{case['leg']}",
                        case["start"],
                    )
                    if learned is None:
                        result["rejected"].append(
                            dict(case=case, reason="optical phase unavailable")
                        )
                    else:
                        phase = learned["source_phase_tick"]
                        approach = [
                            v for v in record if phase - 65 <= v["tick"] <= phase + 3
                        ]
                        result["rows"].append(
                            dict(
                                case=case,
                                template=school.plain(learned),
                                approach=school.plain(approach),
                            )
                        )
            finally:
                camera.close()
                arena.close()
        _write(target, result)
        print(
            case["release"],
            "accepted",
            len(result["rows"]),
            "rejected",
            len(result["rejected"]),
            flush=True,
        )
    result["complete"] = True
    _write(target, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "export"))
    p.add_argument("--region", choices=("all", "central"), default="all")
    a = p.parse_args()
    prefix = "" if a.region == "all" else "central-"
    plan_path = ROOT / (prefix + "sources.json")
    if a.action == "plan":
        plan(a.region, plan_path)
    else:
        export(plan_path, ROOT / (prefix + "templates.json"))
