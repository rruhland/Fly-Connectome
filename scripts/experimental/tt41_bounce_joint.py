"""Test a solo-measured, face-relative effect on the carried direct-joint controller."""

import argparse
from ast import literal_eval
from collections import deque
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt41_bounce_effect import forecast

ROOT = Path("artifacts")
OUT = ROOT / "tt41_bounce_joint"
SRC = ROOT / "tt39/credit-carry.json.gz"
CHECKPOINT = ROOT / "tt39/credit-carry.npz"
PRACTICE = ROOT / "tt37/camera-time-school.npz"
OPTIC = ROOT / "tt37/rebound.npz"
SOLO = ROOT / "tt41_bounce/effect.json"
NOMINAL = np.array([1.85, 0.0])
MODES = ("stock", "still", "elastic", "solo")


def receiving_height(x):
    """Camera-only screening, never a target pose or a physical ball clock."""
    if x is None or x[4] >= -0.90 or not -1.51 < x[1] < -0.51:
        return None
    delay = (-1.53 - x[1]) / (0.98 * x[4])
    return x[2] + x[5] * delay - 4.905 * delay**2 if 0.03 < delay < 0.275 else None


def relative_face(geometry, app, trace, incoming, mode, law):
    t, p, b, n = app[0], app[1:4], app[4:7], app[7:10]
    if not 0.032 < t < 0.325 or not np.isfinite(app).all():
        return None
    d = p - b
    if d @ n < 0:
        n = -n
    if n[1] < 0.46 or np.linalg.norm(d) > 0.157:
        return None
    n = n / np.linalg.norm(n)
    rim = d - n * (d @ n)
    _, signed, j, angle = geometry.pose(app[10:15], jacobian=True)
    if signed @ n < 0:
        angle = -angle
    before = np.flatnonzero(
        np.isfinite(trace[:, 6]) & (np.arange(len(trace)) * mpc.DT < t)
    )[-4:]
    if len(before) < 2:
        return None
    rel = before * mpc.DT - t
    dq = np.linalg.lstsq(
        np.c_[np.ones(len(before)), rel], trace[before, 6:11], rcond=None
    )[0][1]
    speed = j @ dq + np.cross(geometry.angular[:, geometry.dofs] @ dq, rim)
    v = forecast(incoming, speed if mode != "still" else np.zeros(3), n, law)
    return t, p, n, rim, float(d @ n), j, angle, v, speed


def destination(p, pace, y):
    # Gravity + modest drag proxy, used only to give the inverse push a target.
    t = np.clip((y - p[1]) / (0.91 * pace), 0.32, 0.79)
    net = np.clip(-p[1] / (0.93 * pace), 0.13, 0.62)
    z_landing = (0.783 - p[2]) / t + 4.905 * t
    z_net = (1.015 - p[2]) / net + 4.905 * net
    return np.array([-0.92 * p[0] / t, pace, max(z_landing, z_net)])


def commands(seq, body, incoming, mode, law):
    t, p, n, rim, d, j, angle, v, _speed = body
    a = np.r_[j, 0.14 * angle]
    inv = np.linalg.solve(a.T @ a + 0.022**2 * np.eye(5), a.T)
    slide = inv @ np.r_[0.62 * rim + n * np.clip(d - 0.026, -0.016, 0.028), np.zeros(3)]
    time = (1 + np.arange(mpc.HORIZON)) * mpc.DT
    start = np.clip(time / max(0.024, t * 0.63), 0.0, 1.0)
    ramp = start**2 * (3.0 - 2.0 * start)
    # Zero nominal drive near the intended impact; servo lag ramps it in earlier.
    slope = np.clip(time - t + 0.032, -0.060, 0.070) * ramp
    for pace, y in ((3.60, 0.59), (4.35, 0.64), (3.12, 0.46), (4.86, 0.69)):
        want = destination(p, pace, y)
        err = want - v
        impulse = want - incoming
        desired = impulse / max(0.01, np.linalg.norm(impulse))
        normal_turn = desired - n * (desired @ n)
        wrist = inv @ np.r_[np.zeros(3), 0.14 * normal_turn]
        u = np.clip((err @ n) / law[0], -1.25, 1.38)
        # Keep face angle and translational movement separate. The same requested
        # forward/upward ball change can require a different mix at another q.
        drive = (
            inv @ np.r_[u * desired, np.zeros(3)] if mode != "still" else np.zeros(5)
        )
        for w, r, s in (
            (0.52, 0.73, 0.68),
            (0.77, 1.05, 1.0),
            (0.95, 1.35, 1.25),
            (0.30, 1.64, 0.70),
        ):
            dq = np.outer(ramp, np.clip(w * slide + r * wrist, -0.48, 0.48))
            dq += np.outer(slope, np.clip(s * drive, -4.8, 4.8))
            yield seq + np.c_[dq, np.zeros(mpc.HORIZON)], want


def initial_bank(model, serial, initial, path=CHECKPOINT):
    """Exact earlier live rows from a checked archive, never same-game future rows."""
    with np.load(path) as bank:
        model.priors.online.extend(
            (state, commands, tag)
            for state, commands, label in zip(
                bank["states"], bank["goals"], bank["tags"], strict=True
            )
            if (tag := literal_eval(str(label)))[0] == "live" and tag[1] < serial
        )
    if model.priors.digest() != initial:
        raise ValueError(
            "Could not reconstruct the already-earned opening memory exactly"
        )


class Signed(mpc.MPC):
    def __init__(self, bank, raw, *, variant, budget, mode, law, extra):
        super().__init__(bank, raw, variant=variant, budget=budget)
        self.mode = mode
        self.law = np.asarray(law if mode == "solo" else NOMINAL)
        self.extra = extra
        self.collect = False

    def begin(self):
        super().begin()
        self.collect = False
        self.seen = []

    def trial(self, goals, *, long=False, traces=False):
        if not self.collect:
            return super().trial(goals, long=long, traces=traces)
        g = np.asarray(
            [mpc.tail(seqs, mpc.HORIZON + (mpc.TAIL if long else 0)) for seqs in goals]
        )
        f = np.zeros((len(g), 9))
        a = np.empty((len(g), 19))
        lines = np.full((*g.shape[:2], 17), np.nan)
        _, out = self.engine.predict.trajectories(
            g, mpc.AIM, mpc.HORIZON * mpc.DT, feedback=f, approaches=a, traces=lines
        )
        self.diag["native_candidates"] += len(g)
        loss, legal = mpc.cost(out, f)
        for seq, app, track, l, win, values in zip(
            goals, a, lines, loss, legal, out, strict=True
        ):
            if np.isfinite(app[0]) and 0.031 < app[0] < 0.31 and np.isfinite(l):
                self.seen.append((seq, app, track, float(l), bool(win), values))
        return loss, legal, lines if traces else None, out, f

    def assess(self, seqs):
        loss, nominal, _, values, _ = self.trial(seqs, long=True)
        # Prefer sound disk engagement and retain positive net clearance. All arms
        # run the same candidate budget and the same physical margin checks.
        score = np.where(
            nominal, loss + 7.0 * np.maximum(values[:, 12] - 0.045, 0.0), loss
        )
        keep = np.flatnonzero(nominal)
        chosen = keep[np.argsort(score[keep])[:6]]
        if not len(chosen):
            return None, loss, values, 0, 0
        agreement, risk = self.uncertainty([seqs[i] for i in chosen], score[chosen])
        cost = risk + 2.15 * (2 - agreement)
        k = int(np.argmin(cost))
        if agreement[k] == 0:
            return None, loss, values, len(keep), 0
        return (
            (int(chosen[k]), int(agreement[k]), float(cost[k])),
            loss,
            values,
            len(keep),
            int(np.count_nonzero(agreement)),
        )

    def search(self, tick, x, q, reference, base):
        z = receiving_height(x)
        available = (
            self.mode != "stock"
            and self.engine.bounced
            and z is not None
            and 0.972 < z < 1.061
        )
        self.seen = []
        self.collect = available
        try:
            super().search(tick, x, q, reference, base)
        finally:
            self.collect = False
        if (
            not available
            or not self.seen
            or not self.notes
            or self.notes[-1]["tick"] != tick
        ):
            return
        note = self.notes[-1]
        incumbent = (
            self.best
            if self.best is not None and self.tick == tick and note.get("chosen")
            else None
        )
        if (
            incumbent is not None
            and note.get("robust", 0) == 2
            and note.get("score", 9) < -1.12
        ):
            return
        started = process_time()
        candidates = []
        unique = []
        observations = []
        self.diag["bounce_consulted"] += 1
        for seq, app, trace, loss, legal, values in sorted(
            self.seen, key=lambda r: r[3]
        ):
            if any(np.max(np.abs(check - seq)) < 0.032 for check in unique):
                continue
            # Latest camera hypothesis is propagated to each candidate's own
            # contact, not extrapolated from a table-truth event or a fixed plane.
            p, incoming = self.engine.latest.advance((tick * mpc.DT) + app[0])
            if incoming[1] > -1.05 or np.linalg.norm(p - app[1:4]) > 0.095:
                continue
            body = relative_face(
                self.geometry, app, trace, incoming, self.mode, self.law
            )
            if body is None:
                continue
            unique.append(seq)
            t, _impact, _n, rim, _d, _j, _angle, implied, speed = body
            offers = list(commands(seq, body, incoming, self.mode, self.law))
            candidates.extend(offers)
            observations.append(
                [
                    float(loss),
                    bool(legal),
                    float(np.linalg.norm(rim)),
                    float(t),
                    *implied.tolist(),
                    *speed.tolist(),
                ]
            )
            if len(unique) >= max(2, self.extra // 16):
                break
        self.seen = []
        if not candidates:
            return
        # Identical width in each arm, including the unmodified incumbent.
        ids = np.linspace(0, len(candidates) - 1, self.extra, dtype=int)
        options = [candidates[i][0] for i in ids]
        if incumbent is not None:
            options[0] = incumbent
        hit, loss, values, nlegal, nrobust = self.assess(options)
        note["bounce_effect"] = {
            "mode": self.mode,
            "z_m": float(z),
            "n": len(options),
            "nominal_legal": nlegal,
            "robust_legal": nrobust,
            "anchors": observations,
            "extra_cpu_s": process_time() - started,
        }
        self.diag["search_cpu_s"] += process_time() - started
        if hit is None:
            return
        i, agree, risk = hit
        if incumbent is not None:
            if i == 0:
                return
            # Near-legal incumbents get displaced only by a tested margin gain.
            agree0, risk0 = self.uncertainty([incumbent], np.asarray([loss[0]]))
            if agree0[0] and risk + 0.29 >= risk0[0] + 2.15 * (2 - agree0[0]):
                return
        eta = values[i, 1]
        self.best, self.tick, self.lease = (
            options[i],
            tick,
            max(mpc.LEASE, min(16, round(eta / mpc.DT) + 2)),
        )
        self.next = tick + max(4, min(10, round(eta / mpc.DT) - 2))
        self.diag["bounce_committed"] += 1
        note["bounce_effect"].update(
            issued=True,
            robust=agree,
            merit=risk,
            target=candidates[ids[i]][1].tolist(),
            contact_delay_s=float(eta),
            score=values[i].tolist(),
        )


def fork_tick(row):
    """Evaluator schedules a fixed late-game branch. Not called by an actor."""
    e = row["placements"]["delivered"][-1]
    t = e["near_bounce"]["time"]
    return next(
        d["tick"] for d in row["searches"] if d["bounced"] and t <= d["tick"] * mpc.DT
    )


def play_fork(a, res, model, raw, card, row, serial, ready, stop):
    renderer, rig, stride, lo, hi = res
    a.launch(row["shot"], card=card)
    a.partner.reset(seed=card["seed"])
    eye = mpc.Vision(rig, a.control_dt, raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye, OPTIC)
    model.begin()
    sent = np.asarray(ready).copy()
    home = sent.copy()
    rolling = deque(maxlen=150)
    past_inbound = False
    scores = 0
    tape = []
    initial = process_time()
    runtime = None
    for tick in range(min(a.max_steps, stop + 115)):
        q = a.joints.copy()
        x = mpc.observed(eye, tick, renderer.capture() if tick % 2 == 0 else None, q)
        if x is not None and x[4] < -0.9 and x[1] < -0.90:
            past_inbound = True
        if tick < stop:
            model.engine.observe(tick, eye.eye, q, sent.copy())
            u = np.asarray(row["actions"][tick])
        else:
            if tick == stop:
                expected = [e for e in row["events"] if e["step"] <= stop]
                if digest_json(a.events) != digest_json(expected):
                    raise ValueError("Preintervention physical histories differ")
                runtime = process_time()
            desired = home
            if x is not None and (x[4] < -0.9 or (past_inbound and x[1] < 0.50)):
                target, confidence = raw.get(x, q)
                if target is not None:
                    desired = home + confidence * (target - home)
            out = model.step(tick, eye.eye, x, q, sent, desired)
            u = np.clip((np.clip(out, lo, hi) - sent) / stride, -0.8, 0.8)
        prev = sent.copy()
        sent = np.clip(prev + u * stride, lo, hi)
        _, _, done, cap, info = a.exchange(u, mpc.FAR_COMMAND)
        tape.append(u.tolist())
        rolling.append((tick, x, q, prev, sent))
        legal = any(
            e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]
        )
        if legal:
            score = mpc.audit(a.events, opening_side=1)["learner"]
            if score > scores:
                model.priors.keep(list(rolling), ("live", serial, score))
                scores = score
        if done or cap or a.rival_returns > row["score"]["opponent"]:
            break
    placement = a.point_placements()
    i = len(row["placements"]["delivered"]) - 1
    old = row["placements"]["delivered"][i]
    new = placement["delivered"][i]
    for key in ("far_contact", "near_bounce"):
        if digest_json(new[key]) != digest_json(old[key]):
            raise ValueError(
                f"The receiving branch changed the initial delivery: {key}"
            )
    j = new["near_contact"]
    depart = next(
        (
            e["velocity"]
            for e in a.events
            if j
            and e["kind"] == "separation"
            and e.get("collider") == "learner_racket"
            and j["time"] <= e["time"] < j["time"] + 0.07
        ),
        None,
    )
    return {
        "own": mpc.audit(a.events, opening_side=1)["learner"],
        "answered": a.rival_returns > row["score"]["opponent"],
        "legal": bool(new["near_landed"]),
        "depart": depart,
        "actions": tape,
        "events": a.events,
        "placement": new,
        "same_receive_line": digest_json(new["receive_line"])
        == digest_json(old["receive_line"]),
        "cpu_s": process_time() - initial,
        "branch_cpu_s": process_time() - runtime,
        "motor": a.motor_audit(),
        "diagnostics": dict(model.diag),
        "notes": model.notes,
    }


def assess(args):
    spec = _read(SRC)
    if (
        _hash(CHECKPOINT) != spec["checkpoint_sha256"]
        or _hash(mpc.BANK) != spec["source_sha256"]
    ):
        raise ValueError("Prior bundle differs from the credited point")
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    ids = args.ids or [9812, 14390, 12109]
    modes = args.modes or MODES
    out = {
        "schema": "tt41-signed-bounce-joint-1",
        "source_sha256": _hash(SRC),
        "checkpoint_sha256": _hash(CHECKPOINT),
        "effect_sha256": _hash(SOLO),
        "extras": args.extra,
        "modes": modes,
        "ids": ids,
        "rows": [],
        "design": "Archived real prefixes, reconstructed earned optical memory, rearmed independent direct MPCs on the same later incoming offer.",
    }
    law = _read(SOLO)["pooled"]
    with mpc.centered(spec["ready"]):
        a = mpc.arena()
        res = mpc.resources(a)
        try:
            for cid in ids:
                i, card = next(
                    (i, c) for i, c in enumerate(spec["cards"]) if c["id"] == cid
                )
                row = spec["rows"][i]
                stop = fork_tick(row)
                if args.shift:
                    stop += args.shift
                if (
                    not 0.99
                    <= projected_height(
                        row["placements"]["delivered"][-1]["receive_line"]
                    )
                    < 1.04
                ):
                    raise ValueError("Selected offer was not from moderate-height band")
                result = {
                    "card": card,
                    "stock_own": row["score"]["learner"],
                    "fork_tick": stop,
                    "modes": {},
                }
                for mode in modes:
                    model = Signed(
                        mpc.BANK,
                        raw,
                        variant="elastic",
                        budget=spec["budget"],
                        mode=mode,
                        law=law,
                        extra=args.extra,
                    )
                    try:
                        model.priors = Carried(
                            mpc.BANK,
                            model.engine,
                            practice=[PRACTICE],
                            held={c["id"] for c in spec["cards"]},
                        )
                        initial_bank(model, i + 1, row["memory"]["before"])
                        item = play_fork(
                            a, res, model, raw, card, row, i + 1, spec["ready"], stop
                        )
                    finally:
                        model.close()
                    result["modes"][mode] = item
                    print(
                        cid,
                        mode,
                        item["own"],
                        item["legal"],
                        item["answered"],
                        item["depart"],
                        item["diagnostics"],
                        flush=True,
                    )
                out["rows"].append(result)
                _write(args.out, out)
        finally:
            res[0].close()
            a.close()


def live(args):
    from .tt37_joint_deep import cards_for

    spec = _read(SRC)
    receipt = _read(mpc.BANK.with_suffix(".json"))
    cards = cards_for(receipt, args.plan, args.deck)[: args.limit]
    modes = args.modes or MODES
    shots = mpc.measured_launches(args.feeds)
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    law = _read(SOLO)["pooled"]
    report = {
        "schema": "tt41-signed-bounce-joint-live-1",
        "modes": modes,
        "cards": cards,
        "extras": args.extra,
        "plan_sha256": _hash(args.plan),
        "solo_sha256": _hash(SOLO),
        "rows": [],
    }
    held = {c["id"] for c in cards}
    with mpc.centered(receipt["ready"]):
        a = mpc.arena()
        res = mpc.resources(a)
        try:
            for c in cards:
                r = {"card": c, "modes": {}}
                for mode in modes:
                    model = Signed(
                        mpc.BANK,
                        raw,
                        variant="elastic",
                        budget=spec["budget"],
                        mode=mode,
                        law=law,
                        extra=args.extra,
                    )
                    try:
                        model.priors = Carried(
                            mpc.BANK, model.engine, practice=[PRACTICE], held=held
                        )
                        r["modes"][mode] = mpc.play(
                            a,
                            *res,
                            receipt["ready"],
                            c,
                            shots[c["id"]],
                            raw,
                            model,
                            OPTIC,
                            "live",
                            args.cap,
                        )
                    finally:
                        model.close()
                    print(
                        c["id"],
                        mode,
                        r["modes"][mode]["score"]["learner"],
                        r["modes"][mode]["planner"],
                        flush=True,
                    )
                report["rows"].append(r)
                _write(args.out, report)
        finally:
            res[0].close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fork", "live"))
    p.add_argument("out", type=Path)
    p.add_argument("--ids", type=int, nargs="+")
    p.add_argument("--modes", choices=MODES, nargs="+")
    p.add_argument("--extra", type=int, default=64)
    p.add_argument("--shift", type=int, default=0)
    p.add_argument("--plan", type=Path, default=ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--cap", type=int, default=200)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    opts = p.parse_args()
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    (assess if opts.mode == "fork" else live)(opts)
