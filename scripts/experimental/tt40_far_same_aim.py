"""Count missed far replies and test a pack of opt-in same-goal physical repairs."""

import argparse
from collections import Counter
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np


BASE = Path("artifacts")
OUT = BASE / "tt40-far-same-aim"
SOURCES = {
    "articulated_history": [
        "tt37/mix-rollin", "tt37/integrated-practice", "tt37/integrated-replication",
        "tt37/rally-funded", "tt37/value-closed", "tt37/replicate-carry",
        "tt37/opponent-baseline-random-long", "tt38/joint-mpc-development-raw",
        "tt38/joint-mpc-development-live", "tt38/joint-mpc-development-elastic",
        "tt39/prep-lead", "tt39/prep-half", "tt39/skill-replication",
        "tt39/pure-early-replication", "tt39/integrated-31-tape", "tt39/direct-32-tape",
    ],
    "endurance_practice": [f"tt39/endurance-{i:02d}" for i in range(1, 9)],
    "mirror_capacity": [
        "tt37/opponent-capacity-central-random", "tt37/opponent-capacity-wide-random",
    ],
}
TERMINAL = {"point", "timeout", "let"}


def tape(tag):
    path = BASE / (tag + ".json.gz")
    with gzip.open(path, "rt") as f:
        return path, json.load(f)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def card_key(card):
    return tuple(card[k] for k in ("id", "seed", "reply_seed"))


def missed_replies(events, score, requests):
    """A real own landing needs native contact. A later own miss is not a far miss."""
    contact, opening, returns = None, True, []
    for i, event in enumerate(events):
        kind = event["kind"]
        if kind == "contact":
            contact = event
        if kind == "legal_return":
            if contact is None:
                if not opening or event["side"] != 1:
                    raise ValueError("Unmatched legal landing after opening")
            else:
                if contact["side"] != event["side"] or (returns and returns[-1][1]["side"] == event["side"]):
                    raise ValueError("Nonalternating physical replies")
                returns.append((i, event))
            contact, opening = None, False
    own = [(k, i, event) for k, (i, event) in enumerate(returns) if event["side"] == 1]
    if len(own) != score["learner"]:
        raise ValueError("Physical learner count differs from tape's evaluator")
    for leg, (k, i, event) in enumerate(own, 1):
        if k + 1 < len(returns):
            state = "legal_reply"
        else:
            ending = next((e for e in events[i+1:] if e["kind"] in TERMINAL), {})
            if ending.get("kind") == "point" and ending.get("winner") == 1:
                hit = any(e["kind"] == "contact" and e["side"] == -1 for e in events[i+1:])
                state = "illegal_reply" if hit else "missed_contact"
            else:
                state = ending.get("kind", "censored")
        req = next((r for r in requests if r["leg"] == leg), None)
        yield state, leg, event, req


def audit(out, *, through=5):
    practice = {card_key(c) for c in json.loads(
        (BASE / "tt39/endurance-plan.json").read_text())["practice"]}
    report = {}
    for group, names in SOURCES.items():
        pool = {}
        tables = {}
        traces = {}
        for tag in names:
            if tag.startswith("tt39/endurance-") and int(tag[-2:]) > through:
                continue
            path, data = tape(tag)
            if len(data["rows"]) != len(data["cards"]):
                raise ValueError(f"Incomplete numbered tape {tag}")
            counts = Counter()
            for row, card in zip(data["rows"], data["cards"], strict=True):
                ck = card_key(card)
                if group == "endurance_practice" and ck not in practice:
                    raise ValueError(f"Assessment card in practice tape {tag}")
                for state, leg, landing, req in missed_replies(
                        row["events"], row["score"], row["placements"]["requested"]):
                    counts[state] += 1
                    # Repeated controllers can generate genuinely different opportunities on the
                    # same random draw. Identical physical bounces and requests are one opportunity.
                    key = (ck, leg, digest((req, landing)))
                    if key not in pool:
                        pool[key] = state
                    elif state != pool[key]:
                        if state == "censored":
                            continue
                        if pool[key] != "censored":
                            raise ValueError(f"Conflicting outcomes for one physical shot {tag}")
                        pool[key] = state
                    if state in ("missed_contact", "illegal_reply") and key not in traces:
                        traces[key] = dict(tape=tag, card=card, leg=leg, status=state,
                                           bounce=landing["location"], target=None if req is None else req["target"],
                                           reason=next(e["reason"] for e in row["events"] if e["kind"] == "point"))
            tables[tag] = dict(points=len(data["rows"]), **counts,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        counts = Counter(pool.values())
        report[group] = dict(offers=dict(counts), release_seed_pairs=len({key[0] for key in pool}),
                             misses=list(traces.values()), per_tape=tables)
    write(out, report)


def write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    msg = json.dumps(result, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".partial")
    if path.suffix == ".gz":
        tmp.write_bytes(gzip.compress(msg.encode(), mtime=0))
    else:
        tmp.write_text(msg)
    tmp.replace(path)


def pick_repair(partner, path, position, velocity, now):
    """Search the original motor envelope with a wider *contact* region and fixed aim."""
    from .tt29_partner import DRAG, TABLE_TOP, _clears_table, _flight_approx, _trajectory

    a, target = partner.arena, partner.target.copy()
    xyz = path[:, 1:4]
    enough_time = np.linalg.norm(xyz-position, axis=1) < 2.45*path[:, 0]
    mask = ((path[:, 0] > .34) & (np.abs(xyz[:, 0]) < 1.07) & (xyz[:, 1] > .27)
            & (xyz[:, 1] < 2.05) & (xyz[:, 2] > .772) & (xyz[:, 2] < 1.70) & enough_time)
    windows = path[mask]
    if not len(windows):
        return None, {"windows": 0, "verified": 0}
    windows = windows[np.unique(np.linspace(0, len(windows)-1, min(19, len(windows))).round().astype(int))]
    params = list(product((-.65, -.25, .15, .65), (-.09, 0., .08, .16, .26),
                          (-.065, 0., .065), (-.5, -.08, .34, .75), (.0, .75, 1.5)))
    states = np.repeat(windows, len(params), axis=0)
    pace_bias, lift, bearing, speed_bias, rise = np.tile(params, (len(windows), 1)).T
    p, v = states[:, 1:4], states[:, 4:7]
    length = p[:, 1] - target[1]
    pace = (target[3]+pace_bias)*np.exp(DRAG*length)/.93
    landing_y = target[1]+.60+.54*(target[2]-1.01)
    t = np.expm1(DRAG*(p[:, 1]-landing_y))/(DRAG*pace)
    drag_rate = DRAG*pace
    launch_z = (TABLE_TOP-p[:, 2]+9.81*t/drag_rate)*drag_rate/-np.expm1(-drag_rate*t)-9.81/drag_rate
    launch = np.c_[(target[0]-p[:, 0])*pace/length, -pace, launch_z]
    delta = launch-v
    norm = np.linalg.norm(delta, axis=1)
    guess = delta/norm[:, None]
    yaw = np.arctan2(guess[:, 0], -guess[:, 1])+bearing
    pitch = np.arcsin(guess[:, 2])+lift
    ny = np.cos(yaw)
    nx = np.sin(yaw)
    face = np.c_[nx*np.cos(pitch), -ny*np.cos(pitch), np.sin(pitch)]
    along = np.clip((v*face).sum(axis=1)+norm/1.82+speed_bias, -1.85, 2.76)
    tangent = np.c_[-nx*np.sin(pitch), ny*np.sin(pitch), np.cos(pitch)]
    drive = along[:, None]*face+rise[:, None]*tangent
    dep_p, dep_v, dep_w, centers, quats = partner.impact.solve(states, face, drive)
    scores = _flight_approx(dep_p, dep_v, dep_w, target)
    best, quality, verify = None, np.inf, 0
    note = {"windows": int(len(windows))}
    for i in np.argsort(scores):
        if not np.isfinite(scores[i]) or verify >= 55:
            break
        stroke = _trajectory(now, states[i, 0], position, velocity, centers[i], drive[i], quats[i], target)
        if stroke is None or not _clears_table(stroke, a.table_half_length, a.table_half_width):
            continue
        if min(stroke.center[1], stroke.center[1]+.083*drive[i, 1]) < .17:
            continue
        verify += 1
        j = np.searchsorted(path[:, 0], stroke.eta-.038)
        departure = partner.impact.verify_approach(path[j], stroke)
        if departure is None:
            continue
        solved = partner._outgoing(*departure, target[1]-.08)
        if not len(solved) or not solved[-1, 2] <= target[1] <= solved[0, 2]:
            continue
        actual_x, actual_z, actual_vy = [
            np.interp(target[1], solved[::-1, 2], solved[::-1, k]) for k in (1, 3, 5)
        ]
        err = np.abs([actual_x-target[0], actual_z-target[2], actual_vy+target[3]])
        if np.any(err > [.070, .075, .80]):
            continue
        value = err @ [7., 8., .60]
        if value < quality:
            quality, best = value, stroke
            note.update(eta_s=stroke.eta, xyz=p[i].tolist(), predicted_at_goal=[actual_x, actual_z, -actual_vy])
    return best, note | {"verified": verify}


def repair_gate(path, table_depth):
    """The extended causal forecast flags only outward edge shots or a very low rear bounce."""
    if not len(path):
        return None
    first, last = path[[0, -1]]
    if first[2] > 1.41 and np.max(path[:, 3]) < .849 and last[2] > table_depth+.25:
        return "low_trailing_edge"
    if (first[2] < .43 and abs(first[1]) > .68 and abs(last[1]) > .88
            and first[1]*first[4] > .14):
        return "short_outward"
    return None


def gate_audit(out, *, through=5):
    """Check the geometric trigger on achieved practice departures without applying it."""
    from .tt37_opponent import arena

    a = arena(width=64, height=48)
    practice = json.loads((BASE / "tt39/endurance-plan.json").read_text())["practice"]
    cohort = {card_key(c) for c in practice}
    trials = []
    total = Counter()
    try:
        for tag in SOURCES["endurance_practice"]:
            if int(tag[-2:]) > through:
                continue
            _, data = tape(tag)
            for card, row in zip(data["cards"], data["rows"], strict=True):
                if card_key(card) not in cohort:
                    raise ValueError("Non-practice departure used for trigger audit")
                events = row["events"]
                separations = [e for e in events if e.get("collider") == "learner_racket"]
                for state, leg, landing, req in missed_replies(events, row["score"], row["placements"]["requested"]):
                    e = next((e for e in reversed(separations) if e["time"] < landing["time"]), None)
                    if e is None or req is None:
                        raise ValueError("Reply opportunity missing recorded ball departure")
                    path = a.partner.flight.path(e["ball"], e["velocity"], e["spin"], -1,
                                                 bounced=False, horizon=1.62)
                    gate = repair_gate(path, a.table_half_length)
                    total[state] += 1
                    if gate is not None or state == "missed_contact":
                        trials.append(dict(source=tag, card=card, leg=leg, status=state,
                                           gate=gate, lead_s=landing["time"]-e["time"]))
    finally:
        a.close()
    write(out, dict(offers=total, trials=trials,
                    description="Geometry at observed native separation; runtime also needs no stock plan."))


def pilot(out):
    """Paired physical replay on identified training examples and nearby practice controls."""
    from unittest.mock import patch
    from .tt29_endurance import audit as score
    from .tt29_partner import _drive
    from .tt31_ready import centered
    from .tt33_joint import FAR_COMMAND
    from .tt37_opponent import arena
    from .tt37_opponent_audit import CourtClearance, guarded

    jobs = {
        "tt39/endurance-02": (12596,),
        "tt39/endurance-03": (14645,),
        "tt39/endurance-05": (13077, 15035),
    }
    practice = {card_key(c) for c in json.loads(
        (BASE / "tt39/endurance-plan.json").read_text())["practice"]}
    results = []
    for tag, ids in jobs.items():
        path, data = tape(tag)
        by_id = {c["id"]: (c, r) for c, r in zip(data["cards"], data["rows"], strict=True)}
        # Two declared neighboring cards for the false-activation dose, not chosen by success.
        controls = [c["id"] for c in data["cards"][:2] if c["id"] not in ids]
        with centered(data["ready"]):
            a = arena(width=64, height=48)
            cad = arena(width=64, height=48, detail="cad")
            clearance = CourtClearance(cad.model, articulated=True)
            try:
                for id_ in (*ids, *controls):
                    card, row = by_id[id_]
                    if card_key(card) not in practice:
                        raise ValueError("Non-practice repair trial")
                    pair = []
                    for active in (False, True):
                        a.launch(row["shot"], card=card)
                        a.partner.reset(seed=card["seed"])
                        a.sample_clearance(False)
                        checks, previous, goal_crossings = [], set(), {}
                        original = a.partner.advance
                        dynamics = a._aerodynamics
                        last_inbound = cad_from = None

                        def observe():
                            nonlocal last_inbound
                            if a._last_hitter == -1 and a._received_bounce and a.partner.requests:
                                req = a.partner.requests[-1]
                                leg, target = req["leg"], req["target"]
                                current = np.r_[a.ball, a.velocity]
                                if (last_inbound is not None and last_inbound[0] == leg
                                        and current[1] <= target[1] < last_inbound[1][1]):
                                    prev = last_inbound[1]
                                    frac = (target[1]-prev[1])/(current[1]-prev[1])
                                    crossing = prev+frac*(current-prev)
                                    goal_crossings[leg] = [crossing[0], crossing[1], crossing[2], -crossing[4]]
                                last_inbound = leg, current
                            else:
                                last_inbound = None
                            dynamics()

                        def inspect():
                            nonlocal cad_from
                            original()
                            p = a.partner
                            if (not active or p.reception in previous or len(p.requests) < 2
                                    or p.plan_stage != 1 or p.stroke is not None
                                    or a._last_hitter != 1 or a.velocity[1] <= .32):
                                return
                            previous.add(p.reception)
                            path = p.flight.path(a.ball, a.velocity, a.spin, -1,
                                                 bounced=a._received_bounce, horizon=1.62)
                            gate = repair_gate(path, a.table_half_length)
                            if gate is None:
                                return
                            q = a.data.qpos[p.qpos:p.qpos+7]
                            v = a.data.qvel[p.qvel:p.qvel+6]
                            now = float(a.data.time)
                            stroke, meta = pick_repair(p, path, q[:3], v[:3], now)
                            checks.append(dict(leg=p.requests[-1]["leg"], target=p.target.tolist(),
                                               gate=gate, time_s=now, issued=stroke is not None, **meta))
                            if stroke is not None:
                                if not np.array_equal(stroke.target, p.target):
                                    raise ValueError("Repair changed the random target")
                                if cad_from is None:
                                    cad_from = now
                                    guarded(a, clearance)
                                p.stroke = stroke
                                _drive(a.data.xfrc_applied[p.body], q, v, now, p.impact.dt,
                                       p.stroke, p.home, p.home_quat)

                        with patch.object(a.partner, "advance", inspect), patch.object(a, "_aerodynamics", observe):
                            for action in row["actions"]:
                                _, _, ended, truncated, _ = a.exchange(action, FAR_COMMAND)
                                if ended or truncated:
                                    break
                            if active and any(c["issued"] for c in checks):
                                last = [c for c in checks if c["issued"]][-1]["leg"]
                                for _ in range(85):
                                    if ended or truncated or last in goal_crossings:
                                        break
                                    _, _, ended, truncated, _ = a.exchange(np.zeros_like(row["actions"][0]), FAR_COMMAND)
                        if not active and digest(a.events) != digest(row["events"]):
                            raise ValueError(f"Archived stock replay diverged on {id_}")
                        ca = a.clearance_result()
                        if cad_from is not None and ca["samples"] != round((a.data.time-cad_from)/a.model.opt.timestep):
                            raise ValueError("Not every state from first issued repair was CAD checked")
                        pair.append(dict(score=score(a.events, opening_side=1), events=a.events.copy(),
                                         requests=a.partner.requests.copy(), plans=a.partner.reports.copy(),
                                         checks=checks, clearance=ca, cad_from_s=cad_from,
                                         goal_crossings=goal_crossings,
                                         motor=dict(speed=a.partner.max_speed, force=a.partner.max_force,
                                                    angular_speed=a.partner.max_angular_speed)))
                    u, v = pair
                    for req, changed in zip(u["requests"], v["requests"]):
                        if req["target"] != changed["target"]:
                            raise ValueError("Different random goals for corresponding receptions")
                    if not any(c["issued"] for c in v["checks"]) and u["events"] != v["events"]:
                        raise ValueError("A nonissued repair altered the referee tape")
                    stock, changed = ([e for e in outcome["events"]
                                       if e["kind"] == "legal_return" and e["side"] == -1]
                                      for outcome in pair)
                    if stock != changed[:len(stock)]:
                        raise ValueError("The repair altered a previously successful reply")
                    prefix = 0
                    for left, right in zip(u["events"], v["events"]):
                        if left != right:
                            break
                        prefix += 1
                    results.append(dict(source=tag, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                        id=id_, cohort="miss" if id_ in ids else "control", pair=pair,
                                        equal_event_prefix=prefix))
                    print(id_, [(x["score"]["learner"], x["score"]["opponent"], x["checks"]) for x in pair],
                          flush=True)
                    write(out, results)
            finally:
                a.close()
                cad.close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("audit", "gates", "pilot"))
    cli.add_argument("--out", type=Path)
    cli.add_argument("--through", type=int, default=5)
    args = cli.parse_args()
    if args.command == "audit":
        audit(args.out or OUT / "census.json", through=args.through)
    elif args.command == "gates":
        gate_audit(args.out or OUT / "gates.json", through=args.through)
    else:
        pilot(args.out or OUT / "pilot.json.gz")
