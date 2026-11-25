"""Training-only native search of diverse joint options, committed to physical landing."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, Rollout
from .tt33_joint import FAR_COMMAND
from .tt37_motor_clinic import landing_error
from .tt38_motorcoach import AIM, SIZES, answerable, schedules
from .tt39_centered_joint import adjustment
from .tt39_opportunity_map import projected_height
from .tt40_low_feasibility import Probe, replay_prefix

ROOT = Path("artifacts")
OUT = ROOT / "tt42_commit_teacher"
SOURCE = ROOT / "tt37/mix-rollin.json.gz"
PLAN = ROOT / "tt42/direct-course.json"
PRACTICE = tuple(
    ROOT / "tt37" / f"{name}.npz"
    for name in ("long-rgb", "self-combined", "camera-time-school", "camera-mix-rollin")
)
DT = mpc.DT
JITTER = np.array(
    [0.092, 0.105, 0.117, 0.142, 0.163, 0.78, 1.12, 1.17, 1.44, 1.28, 0.033]
)
ZERO = np.zeros(6)


def finite_rank(f, y):
    accepted = np.isfinite(y[:, 0])
    # Preserve many physically different options instead of optimizing the first
    # returning motor mode to exhaustion. Avoid marginal low, short, or wide landings.
    answer = answerable(y)
    direct = (
        1.45 * np.abs(y[:, 4] - 0.76)
        + 2.3 * np.maximum(np.abs(y[:, 3]) - 0.27, 0.0)
        + 3.0 * np.maximum(0.070 - y[:, 5], 0.0)
        + 0.75 * abs(y[:, 9] - 3.45)
    )
    loss = np.where(
        accepted,
        answer + 0.31 * direct - 12.0,
        np.where(
            f[:, 1] > 0.5,
            0.86 * landing_error(f) + 1.8 * f[:, 8],
            13.0 + 32.0 * f[:, 0],
        ),
    )
    return accepted, answer, loss


def jitter(parent, eta, move):
    n = len(parent)
    t = np.arange(n) * DT
    at = np.clip(t + move[10], 0.0, t[-1])
    xx = np.stack(
        [np.interp(at, t, parent[:, j]) for j in range(parent.shape[1])], axis=1
    )
    w = np.clip(t / max(0.036, eta * 0.50), 0.0, 1.0)
    ramp = w * w * (3.0 - 2.0 * w)
    arch = np.clip(t - eta + 0.042, -0.112, 0.091)
    xx[:, :5] += ramp[:, None] * (move[:5] + arch[:, None] * move[5:10])
    return xx


def unique_indices(actions, candidates, maximum, tol=0.050):
    accepted = []
    # Compare the sent-through-contact segment, not the arbitrary long frozen tail.
    for idx in candidates:
        a = actions[idx][::4, :5]
        if any(np.max(np.abs(a - actions[j][::4, :5])) < tol for j in accepted):
            continue
        accepted.append(int(idx))
        if len(accepted) == maximum:
            break
    return accepted


class MultiBank:
    def __init__(self, a, paths):
        # Only prior storage is used; physics remains the single native arm.
        class Adapter:
            twin = a
            actuators = a._arm_ids[1][2]
            predict = Rollout(a, 1)

        self.banks = [mpc.Fragments(p, Adapter) for p in paths]
        self.total = sum(len(b.rows) for b in self.banks)

    def candidates(self, x, q, reference, max_each):
        key = mpc.query(x, q, reference)
        for group, b in enumerate(self.banks):
            distances, rows = b.tree.query(
                key, min(len(b.rows), max(36, max_each * 14))
            )
            found = []
            seen = set()
            for distance, idx in zip(distances, rows, strict=True):
                _state, curves, label = b.rows[idx]
                if label in seen:
                    continue
                seen.add(label)
                found.append((curves.copy(), f"earned-{group}", float(distance)))
                if len(found) >= max_each:
                    break
            yield from found


class Teacher:
    def __init__(
        self,
        a,
        ready,
        batch,
        rounds,
        paths,
        *,
        stop_quality,
        fragments_only,
        phase_adjust=False,
    ):
        self.a = a
        self.ready = np.asarray(ready)
        self.batch = batch
        self.rounds = rounds
        self.stop_quality = stop_quality
        self.fragments_only = fragments_only
        self.probe = Probe(a)
        self.roll = self.probe.core
        self.axes = a._arm_ids[1][2]
        self.bank = MultiBank(a, paths)
        self.rng = np.random.default_rng(420188)
        self.last_valid = []
        # The geometric seed explicitly distinguishes blade faces. Feedback
        # needs the court-oriented incident normal even on the other face.
        self.phase_geometry = BladeIK(a) if phase_adjust else None

    def wave(self, curves, frames):
        return mpc.tail(curves, frames)

    def starts(self, path, bases, n):
        a = self.a
        q = a.joints.copy()
        ref = a.data.ctrl[self.axes].copy()
        goals = []
        tags = []
        time = []
        # Equal support for both signed faces and varied actual flight depths.
        banks = [[r for r in bases if r["face"] == f] for f in (-1, 1)]
        order = []
        for rows in banks:
            buckets = {}
            for k in rows:
                buckets.setdefault(round(k["depth"], 2), []).append(k)
            for i in range(3):
                order.extend(b[i] for b in buckets.values() if len(b) > i)
        if order and not self.fragments_only:
            z = np.zeros(len(SIZES))
            for i, base in enumerate(order[: self.batch // 2]):
                g = schedules(
                    np.asarray([base["q"]]),
                    np.asarray([base["dq"]]),
                    np.array([base["time"]]),
                    z[None],
                    DT,
                    n,
                )[0]
                g[:, 5] = ref[5]
                goals.append(g)
                time.append(base["time"])
                tags.append("geometry-plus" if base["face"] > 0 else "geometry-minus")
        # Retrieve at the actual state and at two reachable future approach phases.
        probes = [(0.0, np.r_[a.ball, a.velocity])]
        for phase in (0.10, 0.24, 0.38):
            m = path[(path[:, 0] >= phase - 0.035) & (path[:, 0] < phase + 0.045)]
            if len(m):
                y = m[np.argmin(abs(m[:, 0] - phase))]
                probes.append((float(y[0]), y[1:7]))
        for wait, x in probes:
            delay = max(0, round(wait / DT))
            for found, tag, d in self.bank.candidates(x, q, ref, 4):
                pad = np.tile(found[0], (delay, 1))
                g = self.wave(np.concatenate((pad, found)), n)
                g[:, 5] = ref[5]
                near = path[np.argmin(abs(path[:, 2] + 1.52)), 0]
                if d < 7.0 or wait == 0.0:
                    goals.append(g)
                    tags.append(tag)
                    time.append(float(near))
        # Explicit common recovery and cold no-op controls also participate.
        goals.extend(np.tile(row, (n, 1)) for row in (ref, self.ready))
        tags.extend(("hold-reference", "ready"))
        time.extend([0.2, 0.2])
        return goals, tags, time

    def search(self):
        self.last_valid = []
        a = self.a
        started = process_time()
        path = a.partner.flight.path(
            a.ball, a.velocity, a.spin, 1, bounced=a._received_bounce, horizon=0.90
        )
        if not len(path):
            return None, {
                "reason": "no native post-bounce approach",
                "virtual_legal": 0,
                "cpu_s": process_time() - started,
                "rounds": [],
            }
        path = path[
            (path[:, 0] > 0.026)
            & (path[:, 3] > 0.755)
            & (path[:, 3] < 1.37)
            & (abs(path[:, 1]) < 0.65)
        ]
        if not len(path):
            return None, {
                "reason": "no in-workspace flight",
                "virtual_legal": 0,
                "cpu_s": process_time() - started,
                "rounds": [],
            }
        stop = float(min(0.80, path[-1, 0] + 0.08))
        n = round((stop + 1.10) / DT)
        bases = [] if self.fragments_only else self.probe.basis()
        e, tags, eta = self.starts(path, bases, n)
        if len(e) > self.batch:
            ix = np.linspace(0, len(e) - 1, self.batch, dtype=int)
            e = [e[i] for i in ix]
            tags = [tags[i] for i in ix]
            eta = [eta[i] for i in ix]
        goals, origin, clock = list(e), list(tags), list(eta)
        raw = (
            2
            * qmc.Sobol(
                11, scramble=True, seed=int(self.rng.integers(2**30))
            ).random_base2(int(np.ceil(np.log2(self.batch))))
            - 1
        ) * JITTER
        for i in range(self.batch - len(e)):
            k = i % len(e)
            goals.append(jitter(e[k], eta[k], raw[i]))
            origin.append(tags[k])
            clock.append(eta[k])
        winners = []
        stats = []
        number = 0
        uniques = set()
        for stage in range(self.rounds):
            gs = np.asarray(goals)
            f = np.zeros((len(gs), 9))
            approach = np.empty((len(gs), 19))
            rates, y = self.roll.trajectories(
                gs, AIM, stop, feedback=f, approaches=approach
            )
            good, quality, loss = finite_rank(f, y)
            tracked = np.flatnonzero(good)
            counts = {
                tag: {
                    "tested": origin.count(tag),
                    "legal": int(np.sum(good & (np.asarray(origin) == tag))),
                }
                for tag in sorted(set(origin))
            }
            stats.append(
                {
                    "stage": stage,
                    "trials": len(gs),
                    "contact": int(np.sum(f[:, 1] > 0.5)),
                    "legal": len(tracked),
                    "family": counts,
                    "nearest_m": float(f[:, 0].min()),
                    "best": None if not len(tracked) else float(np.min(loss[tracked])),
                }
            )
            for idx in tracked:
                # The issued tape includes physical landing, not just contact.
                length = max(1, min(len(rates[idx]), int(np.ceil(y[idx, 2] / DT)) + 3))
                p = rates[idx, :length].copy()
                key = digest_json(np.around(p, 7).tolist())
                if key in uniques:
                    continue
                uniques.add(key)
                winners.append(
                    {
                        "score": float(loss[idx]),
                        "answerability": float(quality[idx]),
                        "rates": p,
                        "native": y[idx].tolist(),
                        "departure": f[idx, 2:5].tolist(),
                        "family": origin[idx],
                        "stage": stage,
                        "unique_rate_sha256": key,
                    }
                )
            number += len(tracked)
            if winners and min(q["answerability"] for q in winners) < self.stop_quality:
                break
            if stage + 1 >= self.rounds:
                break
            ids = unique_indices(gs, np.argsort(loss), max(8, self.batch // 10))
            if not ids:
                break
            # Use selection-independent variation for a quarter of the search and
            # exploit candidate-local signed physical return effects in the rest.
            p = []
            f2 = []
            c2 = []
            for j in ids:
                p.append(gs[j])
                f2.append(origin[j])
                c2.append(clock[j])
                app = approach[j]
                bound = stop if self.phase_geometry is not None else 0.40
                if np.isfinite(app[0]) and 0.04 < app[0] < bound:
                    stem = path[np.argmin(abs(path[:, 0] - app[0])), 4:7]
                    geometry = self.phase_geometry or self.probe.ik
                    frames = n if self.phase_geometry is not None else mpc.HORIZON
                    terms = adjustment(geometry, app, f[j], stem, frames)
                    if terms is not None:
                        a0, b, c = terms
                        gains = ((0.76, 0.64, 0.8), (1.06, 1.0, 1.15))
                        if self.phase_geometry is not None:
                            # Occupy the available time with a smooth approach
                            # rather than changing the initial motor reference.
                            # Keep one move for contact and one for departure.
                            gains = ((1.0, 0.24, 0.38), (0.56, 1.1, 1.28))
                        for w, z, m in gains:
                            correction = self.wave(w * a0 + z * b + m * c, n)
                            p.append(gs[j] + correction)
                            f2.append(origin[j] + "-centered")
                            c2.append(float(app[0]))
            raw = (
                (
                    2
                    * qmc.Sobol(
                        11, scramble=True, seed=int(self.rng.integers(2**30))
                    ).random_base2(int(np.ceil(np.log2(self.batch))))
                    - 1
                )
                * JITTER
                * (1.0 if stage < 2 else 0.68)
            )
            retain = min(self.batch // 3, len(p))
            goals, origin, clock = (
                list(p[:retain]),
                list(f2[:retain]),
                list(c2[:retain]),
            )
            for i in range(self.batch - retain):
                # Keep independent raw earned and signed geometric seeds alive too.
                if i % 5 == 0:
                    k = (i + stage) % len(e)
                    source, tag, center = e[k], tags[k], eta[k]
                else:
                    k = (i + stage) % len(p)
                    source, tag, center = p[k], f2[k], c2[k]
                goals.append(jitter(source, center, raw[i]))
                origin.append(tag)
                clock.append(center)
        if not winners:
            return None, {
                "virtual_legal": number,
                "unique_legal": 0,
                "rounds": stats,
                "windows": len(bases),
                "cpu_s": process_time() - started,
            }
        self.last_valid = winners
        chosen = min(winners, key=lambda w: w["score"])
        return chosen, {
            "virtual_legal": number,
            "unique_legal": len(winners),
            "rounds": stats,
            "windows": len(bases),
            "selection": {k: v for k, v in chosen.items() if k != "rates"},
            "cpu_s": process_time() - started,
        }

    def recover(self):
        a = self.a
        ref = a.data.ctrl[self.axes]
        return np.clip(
            (self.ready - ref) / (a._joint_increments[self.axes] * self.roll.micro),
            -0.8,
            0.8,
        )


def opportunity(events, start):
    return next(
        (
            e
            for e in reversed(events[start:])
            if e["kind"] == "separation" and e.get("collider") == "rival_racket"
        ),
        None,
    )


def run_point(
    a,
    teacher,
    shot,
    card,
    prefix,
    cap,
    interval,
    attempts,
    max_legs,
    start_event=0,
    progress=None,
    snapshot=None,
):
    actions = list(prefix)
    notes = []
    spent = {}
    current = None
    last = start_event
    incoming = None
    next_search = 0
    previous_offer_count = a.rival_returns
    actual_own = audit(a.events, opening_side=1)["learner"]
    while not (a.terminated or a.truncated):
        t = len(actions)
        if current is None:
            strike = opportunity(a.events, last)
            if strike is not None and strike["step"] >= (
                incoming["step"] if incoming is not None else -1
            ):
                incoming = strike
            open_offer = (
                incoming is not None
                and a.velocity[1] < -0.5
                and a.ball[1] > -1.72
                and sum(
                    e["kind"] == "contact"
                    and e["side"] == -1
                    and e["step"] >= incoming["step"]
                    for e in a.events[-5:]
                )
                < 2
            )
            if (
                open_offer
                and t >= next_search
                and spent.get(incoming["step"], 0) < attempts
            ):
                spent[incoming["step"]] = spent.get(incoming["step"], 0) + 1
                next_search = t + interval
                plan, note = teacher.search()
                notes.append(
                    {
                        "tick": t,
                        "scored_before": a.legal_returns,
                        "actual_own_before": actual_own,
                        "source_far_separation": incoming["time"],
                        **note,
                    }
                )
                if plan is not None:
                    current = {"tick": t, "rates": plan["rates"], "note": notes[-1]}
                    current["note"].update(
                        committed=True,
                        initialized_from_exact_current_state=True,
                        planned_actions=plan["rates"].tolist(),
                    )
            u = teacher.recover() if current is None else current["rates"][0]
        else:
            j = t - current["tick"]
            # A capped schedule can only hold its last command through an unexpected
            # remaining flight. No other stroke can interrupt the selected approach.
            u = (
                current["rates"][min(j, len(current["rates"]) - 1)]
                if j < len(current["rates"])
                else ZERO
            )
        _, _, done, capped, info = a.exchange(u, FAR_COMMAND)
        actions.append(np.asarray(u).tolist())
        offers = any(
            e["kind"] == "legal_return" and e["side"] == -1 for e in info["events"]
        )
        landed = next(
            (
                e
                for e in info["events"]
                if e["kind"] == "legal_return" and e["side"] == 1
            ),
            None,
        )
        if landed is not None:
            actual_own = audit(a.events, opening_side=1)["learner"]
            if current is not None:
                current["note"].update(
                    physical_legal=True,
                    landing=landed,
                    executed_ticks=t + 1 - current["tick"],
                )
                current = None
            incoming = None
            last = len(a.events)
            if snapshot is not None:
                snapshot(actions, notes)
            if progress and actual_own and actual_own % progress == 0:
                print(
                    "checkpoint",
                    card["id"],
                    actual_own,
                    t,
                    notes[-1].get("selection", {}).get("family"),
                    flush=True,
                )
        if done or capped:
            if current is not None:
                current["note"].update(
                    physical_legal=False,
                    executed_ticks=t + 1 - current["tick"],
                    failure=a.reason,
                )
            break
        if actual_own >= cap or (
            max_legs and offers and a.rival_returns >= previous_offer_count + max_legs
        ):
            if current is not None:
                current["note"].update(
                    capped_pending=True, executed_ticks=t + 1 - current["tick"]
                )
            break
    marks = a.point_placements()
    for item in notes:
        if item.get("physical_legal"):
            lf = item["landing"]["time"]
            item["answered"] = any(
                d["far_contact"]["time"] > lf and d["near_bounce"]
                for d in marks["delivered"]
            )
    return {
        "shot": shot,
        "opponent_seed": card["seed"],
        "actions": actions,
        "events": a.events,
        "score": audit(a.events, opening_side=1),
        "placements": marks,
        "teacher_searches": notes,
        "motor": a.motor_audit(),
        "clearance": a.clearance_result(),
        "physical_time_s": float(a.data.time),
        "launches": 1,
        "terminated": a.terminated,
        "cap_reached": actual_own >= cap,
    }


def run(args):
    source = _read(args.source)
    ids = args.ids
    launches = mpc.measured_launches(args.feeds)
    ready = source["ready"]
    if args.prepare:
        ready = np.array(ready)
        ready[[1, 2, 4]] = 0.58 * ready[[1, 2, 4]] + 0.42 * np.array(
            [-1.372, -0.923, 0.483]
        )
        ready = ready.tolist()
    if args.mode == "fork" and args.prepare:
        raise ValueError("An intact archived prefix must begin at its original posture")
    if args.mode == "fork":
        chosen = [
            (i, c, r)
            for i, (c, r) in enumerate(
                zip(source["cards"], source["rows"], strict=True)
            )
            if c["id"] in ids
        ]
        if {c["id"] for _, c, _ in chosen} != set(ids):
            raise ValueError("Fork IDs must be present in the source")
    else:
        chosen = [
            (i, c, None)
            for i, c in enumerate(_read(args.plan)["practice"][: args.limit])
        ]
    out = {
        "role": "training only: exact current-state teacher, no future opponent targets or native features exported",
        "ready": ready,
        "mode": args.mode,
        "protocol": "random",
        "source_sha256": _hash(args.source),
        "bank_sha256": {str(p): _hash(p) for p in args.banks},
        "protocol_sha256": _hash(args.plan),
        "batch": args.batch,
        "rounds": args.rounds,
        "attempts": args.attempts,
        "cap": args.cap,
        "cards": [c for _, c, _ in chosen],
        "fragment_only": args.fragments_only,
        "rows": [],
    }
    if args.phase_adjust:
        out["phase_adjust"] = True
    with mpc.centered(ready):
        a = mpc.arena(protocol="random")
        a.sample_clearance(live=False)
        try:
            teacher = Teacher(
                a,
                ready,
                args.batch,
                args.rounds,
                args.banks,
                stop_quality=args.quality,
                fragments_only=args.fragments_only,
                phase_adjust=args.phase_adjust,
            )
            out["earned_segments_available"] = teacher.bank.total
            for i, c, r in chosen:
                t0 = process_time()
                if r is not None:
                    offer = r["placements"]["delivered"][-1]
                    if offer["near_landed"] or not offer["near_bounce"]:
                        raise ValueError(
                            "The terminal fork case does not have a missed legal offer"
                        )
                    hit = offer["far_contact"]["time"]
                    event = next(
                        e
                        for e in r["events"]
                        if e["kind"] == "separation"
                        and e.get("collider") == "rival_racket"
                        and e["time"] >= hit
                    )
                    tick = int(np.ceil((event["time"] + 1e-8) / DT))
                    replay_prefix(a, r, c, tick)
                    prefix = r["actions"][:tick]
                    bundle = run_point(
                        a,
                        teacher,
                        r["shot"],
                        c,
                        prefix,
                        args.cap,
                        args.interval,
                        args.attempts,
                        2,
                    )
                    actual = bundle["placements"]["delivered"][
                        len(r["placements"]["delivered"]) - 1
                    ]
                    for k in ("far_contact", "near_bounce"):
                        if digest_json(actual[k]) != digest_json(offer[k]):
                            raise ValueError(
                                "The current arm received a changed physical ball"
                            )
                    bundle.update(
                        fork_tick=tick,
                        same_incoming=True,
                        old_legal=offer["near_landed"],
                        new_legal=actual["near_landed"],
                        id=c["id"],
                        old_score=r["score"]["learner"],
                        projected_height=projected_height(offer["receive_line"])
                        if offer["receive_line"]
                        else None,
                    )
                else:
                    a.launch(launches[c["id"]], card=c)
                    a.partner.reset(seed=c["seed"])

                    def snapshot(actions, notes, card=c):
                        out["in_progress"] = {
                            "card": card,
                            "shot": launches[card["id"]],
                            "actions": actions,
                            "events": a.events,
                            "points": a.legal_returns,
                            "teacher_searches": notes,
                            "rng": teacher.rng.bit_generator.state,
                        }
                        _write(args.out, out)

                    bundle = run_point(
                        a,
                        teacher,
                        launches[c["id"]],
                        c,
                        [],
                        args.cap,
                        args.interval,
                        args.attempts,
                        0,
                        progress=5,
                        snapshot=snapshot,
                    )
                bundle["cpu_s"] = process_time() - t0
                out["rows"].append(bundle)
                out.pop("in_progress", None)
                _write(args.out, out)
                print(
                    c["id"],
                    bundle["score"]["learner"],
                    bundle["score"]["opponent"],
                    [
                        (
                            r["tick"],
                            r["virtual_legal"],
                            r.get("physical_legal"),
                            r.get("answered"),
                        )
                        for r in bundle["teacher_searches"]
                        if r.get("committed")
                    ][-8:],
                    flush=True,
                )
        finally:
            a.close()
    out["complete"] = True
    _write(args.out, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("live", "fork"))
    p.add_argument("out", type=Path)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--ids", type=int, nargs="+", default=[4322, 8302, 14843])
    p.add_argument("--banks", nargs="+", type=Path, default=list(PRACTICE))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--interval", type=int, default=6)
    p.add_argument("--quality", type=float, default=2.7)
    p.add_argument("--cap", type=int, default=100)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--fragments-only", action="store_true")
    p.add_argument("--phase-adjust", action="store_true")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    run(a)
