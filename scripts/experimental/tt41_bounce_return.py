"""Use measured solo contact effects to propose bounded joint changes on live low shots."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from . import tt40_low_lift as core
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BladeIK
from .tt37_motor_clinic import AIM, landing_error, schedules
from .tt38_motorcoach import answerable
from .tt40_low_feasibility import replay_prefix, roster
from .tt41_bounce_effect import OUT, forecast

# Equal count and absolute size of generic proposals in the three learners.
SIZES = np.array([0.095, 0.12, 0.115, 0.15, 0.20, 0.95, 1.15, 1.25, 1.65, 1.5, 0.026])
COLD = np.array([1.85, 0.0])
MODES = ("fixed", "cold", "elastic", "solo", "online")


def plane(n, x):
    return x - (n @ x) * n


def face_velocity(kin, trace, dt, app, n):
    k = int(np.clip(round(app[0] / dt), 1, len(trace) - 2))
    good = np.flatnonzero(np.isfinite(trace[:, 6]))
    if len(good) < 3:
        return np.zeros(3)
    ix = good[np.argsort(abs(good - k))[:3]]
    point = plane(n, app[1:4] - app[4:7])
    _, _, jac, _ = kin.pose(app[10:15], jacobian=True)
    mat = kin.angular[:, kin.dofs]
    dq = np.linalg.lstsq(np.c_[np.ones(3), (ix - k) * dt], trace[ix, 6:11], rcond=None)[
        0
    ][1]
    return jac @ dq + np.cross(mat @ dq, point)


def directions(kin, app, track, optic, dt, coeff, mode):
    eta = app[0]
    if not 0.016 < eta < 0.30:
        return []
    ball, center, normal, q = app[1:4], app[4:7], app[7:10], app[10:15]
    gap = ball - center
    if gap @ normal < 0.0:
        normal = -normal
    if normal[1] < 0.48 or np.linalg.norm(gap) > 0.17:
        return []
    speed = face_velocity(kin, track, dt, app, normal)
    _, v_before = optic.advance(optic.time)
    # Approach propagation belongs to the sensor-initialized twin. Its very
    # first contact state precedes solver impulse except in grazing cases.
    incoming = app[15:18]
    if incoming[1] > -1.1 or np.linalg.norm(incoming - v_before) > 2.8:
        incoming = v_before
    active_speed = speed if mode != "cold" else np.zeros(3)
    pred = forecast(incoming, active_speed, normal, coeff)
    center, signed, jac, tilt = kin.pose(q, jacobian=True)
    if signed @ normal < 0:
        tilt = -tilt
    rows = []
    inv = np.linalg.solve(
        np.r_[jac, 0.15 * tilt].T @ np.r_[jac, 0.15 * tilt] + 0.00068 * np.eye(5),
        np.r_[jac, 0.15 * tilt].T,
    )
    slide = plane(normal, gap) * 0.68 + normal * np.clip(
        (gap @ normal) - 0.024, -0.035, 0.035
    ) * (0.52 if not app[18] else 0.0)
    for forward, landing in ((3.85, 0.61), (4.65, 0.72), (3.25, 0.48), (5.05, 0.75)):
        flight = np.clip((landing - ball[1]) / (0.92 * forward), 0.34, 0.74)
        target = np.array(
            [
                -0.91 * ball[0] / flight,
                forward,
                (0.784 - ball[2]) / flight + 4.905 * flight,
            ]
        )
        needed = target - incoming
        desired = needed / max(np.linalg.norm(needed), 0.01)
        delta_v = target - pred
        tilt_change = plane(normal, desired - normal)
        motor_delta = np.clip((delta_v @ normal) / coeff[0], -0.95, 0.95) * desired
        if mode == "cold":
            motor_delta *= 0.0
        tilt_delta = inv @ np.r_[np.zeros(3), 0.15 * tilt_change]
        displacement = inv @ np.r_[slide, np.zeros(3)]
        motor = inv @ np.r_[motor_delta, np.zeros(3)]
        rows.append((displacement, tilt_delta, motor))
    return rows


def corrupt(templates, time, shifts, anchors):
    stamp = np.clip(time[None, :] + shifts[:, 10, None], time[0], time[-1])
    where = (stamp - time[0]) * 60
    ia = where.astype(int)
    ib = np.minimum(ia + 1, len(time) - 1)
    weight = (where - ia)[..., None]
    out = (
        templates[anchors[:, None], ia] * (1 - weight)
        + templates[anchors[:, None], ib] * weight
    )
    onset = np.clip(time / 0.064, 0.0, 1.0)
    onset = onset**2 * (3.0 - 2.0 * onset)
    slope = np.clip(time - 0.106, -0.085, 0.060)
    out[:, :, :5] += onset[None, :, None] * (
        shifts[:, None, :5] + slope[None, :, None] * shifts[:, None, 5:10]
    )
    return out


class Lift:
    def __init__(self, doc, mode, batch=78, rounds=2):
        self.mode = mode
        self.law = (
            np.asarray(doc["pooled"]) if mode not in ("cold", "elastic") else COLD
        )
        self.batch = batch
        self.rounds = rounds
        self.kin = None
        self.beta = np.ones((3, 2))
        self.chosen = None
        self.feedback_history = []

    def begin(self):
        self.chosen = None

    def cancel(self):
        self.chosen = None

    def feedback(self, score, done):
        if self.mode == "online" and self.chosen is not None and (score or done):
            self.beta[self.chosen, int(not score)] += 1.0
            self.feedback_history.append({"style": self.chosen, "legal": bool(score)})
            self.cancel()

    def plan(self, engine, surf, obs, selected, score, fit, eta):
        note = {"active": False, "height": float(obs[2]), "mode": self.mode}
        result = {"tape": None, "score": None, "eta": None, "trace": [], "note": note}
        if (
            score is not None
            or self.mode == "fixed"
            or not (0.861 < obs[2] < 1.002 and 0.052 < eta < 0.200)
        ):
            return result
        if (
            engine.latest is None
            or not engine.bounced
            or not engine.twin._received_bounce
        ):
            return result
        t0 = process_time()
        dt = engine.twin.control_dt
        n = round((eta + 0.98) / dt)
        m = round((eta + 0.083) / dt)
        time = (1 + np.arange(m)) * dt
        base = schedules(*fit, eta, np.asarray(selected)[None], dt, surf.ready, m)[0]
        if self.kin is None:
            self.kin = BladeIK(engine.twin)
        notes = []
        note.update(active=True, cpu_s=0.0)

        def eval_(seq):
            goals = np.concatenate(
                [seq, np.repeat(seq[:, -1:], max(0, n - m), axis=1)], axis=1
            )
            measurements = np.zeros((len(seq), 9))
            approach = np.empty((len(seq), 19))
            trace = np.full((*goals.shape[:2], 17), np.nan)
            _, trials = engine.predict.trajectories(
                goals,
                AIM,
                eta + 0.19,
                feedback=measurements,
                approaches=approach,
                traces=trace,
            )
            wins = np.isfinite(trials[:, 0])
            loss = np.where(
                wins,
                answerable(trials) - 18.0,
                np.where(
                    measurements[:, 1] > 0.5,
                    landing_error(measurements) + 1.6 * measurements[:, 8],
                    10.0 + 29.0 * measurements[:, 0],
                ),
            )
            return wins, loss, measurements, approach, trace, trials

        stage_anchors = base[None]
        rng = np.random.default_rng(417631)
        # Each policy gets identical numbers of native nominal candidates. The
        # initial population is exactly shared. Directed additions substitute
        # for generic candidates, rather than buying extra searches.
        data = []
        styles = [0] * self.batch
        for step in range(self.rounds):
            z = qmc.Sobol(
                11, scramble=True, seed=int(rng.integers(2**30))
            ).random_base2(int(np.ceil(np.log2(self.batch))))
            perturb = (2 * z[: self.batch] - 1) * SIZES * (1.15 if step == 0 else 0.73)
            perturb[: min(len(stage_anchors), self.batch)] = 0.0
            anchor = np.arange(self.batch) % len(stage_anchors)
            seq = corrupt(stage_anchors, time, perturb, anchor)
            styles[:] = [0] * len(seq)
            if step > 0:
                proposals = []
                offers = []
                for base_i, approach, track, _, _, _ in data[:5]:
                    for disp, rot, drive in directions(
                        self.kin,
                        approach,
                        track,
                        engine.latest,
                        dt,
                        self.law,
                        self.mode,
                    ):
                        early = np.clip(time / max(0.022, approach[0] * 0.59), 0.0, 1.0)
                        ramp = early**2 * (3.0 - 2.0 * early)
                        slope = np.clip(time - approach[0] + 0.033, -0.052, 0.064)
                        for i, (s, r, d) in enumerate(
                            (
                                (0.65, 0.66, 0.65),
                                (0.75, 1.14, 1.15),
                                (0.48, 1.38, 0.90),
                                (0.88, 0.76, 1.52),
                            )
                        ):
                            if self.mode == "online":
                                q = (
                                    self.beta[1 + i // 2, 0]
                                    / self.beta[1 + i // 2].sum()
                                )
                                d *= 0.70 + 0.60 * q
                            add = ramp[:, None] * np.clip(
                                s * disp + r * rot, -0.47, 0.47
                            ) + np.outer(slope * ramp, np.clip(d * drive, -4.8, 4.8))
                            proposals.append(base_i + np.c_[add, np.zeros(m)])
                            offers.append(1 + i // 2)
                available = self.batch // 2
                if len(proposals) > available:
                    jj = np.linspace(0, len(proposals) - 1, available, dtype=int)
                else:
                    jj = np.arange(len(proposals))
                if len(jj):
                    seq[-len(jj) :] = np.asarray(proposals)[jj]
                    styles[-len(jj) :] = np.asarray(offers)[jj]
            wins, loss, meas, app, tracks, values = eval_(seq)
            ii = np.argsort(loss)
            notes.append(
                {
                    "trials": len(seq),
                    "legal": int(wins.sum()),
                    "contacts": int(np.sum(meas[:, 1] > 0.5)),
                    "closest": float(meas[:, 0].min()),
                    "best": float(loss.min()),
                }
            )
            data = sorted(
                data
                + [
                    (seq[i], app[i], tracks[i], float(loss[i]), values[i], styles[i])
                    for i in ii[:14]
                ],
                key=lambda v: v[3],
            )[:14]
            stage_anchors = np.array([pair[0] for pair in data[:10]])
        note.update(
            cpu_s=process_time() - t0,
            trials=sum(r["trials"] for r in notes),
            found=any(r["legal"] for r in notes),
        )
        result["trace"] = notes
        if not np.isfinite(data[0][4][0]):
            return result
        chosen = data[0]
        self.chosen = int(chosen[-1])
        note.update(issued=True, style=int(chosen[-1]), forecast=chosen[4].tolist())
        result.update(tape=chosen[0], score=chosen[4].tolist(), eta=eta, commit=True)
        return result


class ScoreTap:
    """Use the existing stock scorer without making it adaptive itself."""

    def __init__(self, value, lift):
        self.value = value
        self.lift = lift

    def cost(self, outcomes):
        return self.value.cost(outcomes)

    def begin_point(self):
        pass

    def issued(self, prediction):
        pass

    def feedback(self, sides, terminal):
        self.lift.feedback(1 in sides, terminal)


def witness(args):
    source = _read(core.SCHOOL)
    B = _read(args.law)
    core.joint.RECEIVE_Y = source["depth"]
    model = _read(core.PUPIL)
    surface = core.setup(source["depth"])
    out = {
        "schema": "tt41-bounce-return-witness-v1",
        "source_sha256": _hash(core.SCHOOL),
        "law_sha256": _hash(args.law),
        "selection": "development low misses chosen before testing. The branch uses no release number, aim or native contact label.",
        "rows": [],
    }
    with core.centered(source["ready"]):
        a = core.arena()
        engine = core.OpticalClinic(body=True)
        stereo, rig, stride, lo, hi = core.joint.resources(a)
        try:
            for case in roster(source):
                c, row = (source[k][case["index"]] for k in ("cards", "rows"))
                leg = case["leg"]
                offered = row["placements"]["delivered"][leg]
                decisions = [
                    p
                    for p in row["records"]
                    if offered["near_bounce"]["time"] < p["tick"] / 60
                    and (
                        offered["near_contact"] is None
                        or p["tick"] / 60 < offered["near_contact"]["time"]
                    )
                ]
                if args.first:
                    decisions = decisions[:1]
                results = []
                for decision in decisions:
                    tick = decision["tick"]
                    a.launch(row["shot"], card=c)
                    a.partner.reset(seed=c["seed"])
                    eye = core.joint.Vision(rig, a.control_dt)
                    eye.eye = core.LearnedOptics(eye.eye, source["visual"])
                    history = core.ArrivalHistory()
                    actor = core.joint.JointPolicy(
                        model | {"ready": source["ready"]}, stride, lo, hi
                    )
                    engine.previous = None
                    engine.bounced = False
                    for k in range(tick + 1):
                        q = a.joints.copy()
                        obs = history.update(
                            k * a.control_dt,
                            eye.update(k, stereo.capture() if k % 2 == 0 else None, q),
                        )
                        engine.observe(k, eye.eye, q, actor.reference.copy())
                        if k == tick:
                            break
                        u = np.asarray(row["actions"][k])
                        a.exchange(u, core.clinic.FAR)
                        actor.reference = np.clip(actor.reference + u * stride, lo, hi)
                    if digest_json(a.events) != digest_json(
                        [v for v in row["events"] if v["step"] <= tick]
                    ):
                        raise ValueError("Physical prefix disagrees")
                    if obs is None or not np.allclose(
                        np.r_[obs[1:6], q[:5]], decision["context"], atol=1e-9
                    ):
                        raise ValueError("Optical/encoder prefix disagrees")
                    if not engine.sync():
                        continue
                    results.append(
                        {
                            "tick": tick,
                            "optical": obs.tolist(),
                            "prefix": digest_json(a.events),
                            "trials": {},
                        }
                    )
                    for mode in ("cold", "solo"):
                        lift = Lift(B, mode, args.batch, args.rounds)
                        fit = core.clinic.geometry(surface, obs)
                        p = lift.plan(
                            engine,
                            surface,
                            obs,
                            np.asarray(decision["action"]),
                            decision["score"],
                            fit,
                            obs[0],
                        )
                        if p["tape"] is None:
                            played = {
                                "legal": False,
                                "contact": False,
                                "answered": False,
                            }
                        else:
                            replay_prefix(a, row, c, tick)
                            actor.reference = np.clip(
                                np.asarray(source["ready"]) + 0.0, lo, hi
                            )
                            for u in row["actions"][:tick]:
                                actor.reference = np.clip(
                                    actor.reference + np.asarray(u) * stride, lo, hi
                                )
                            actions = list(row["actions"][:tick])
                            count = 0
                            for j in range(180):
                                goal = (
                                    p["tape"][j]
                                    if j < len(p["tape"])
                                    else surface.ready
                                )
                                action = actor.drive(goal, a.joints.copy(), 0.8)
                                actions.append(action.tolist())
                                _, _, done, capped, _ = a.exchange(
                                    action, core.clinic.FAR
                                )
                                count += 1
                                if done or capped or a.rival_returns >= leg + 2:
                                    break
                            y = a.point_placements()
                            physical = y["delivered"][leg]
                            if digest_json(physical["far_contact"]) != digest_json(
                                offered["far_contact"]
                            ) or digest_json(physical["near_bounce"]) != digest_json(
                                offered["near_bounce"]
                            ):
                                raise ValueError(
                                    "Changed motion must face same physical delivery"
                                )
                            played = {
                                "legal": bool(physical["near_landed"]),
                                "contact": bool(physical["near_contact"]),
                                "answered": len(y["delivered"]) > leg + 1
                                and bool(y["delivered"][leg + 1]["near_bounce"]),
                                "actions": actions,
                                "events": a.events,
                                "attempt_ticks": count,
                                "score": core.audit(a.events, opening_side=1),
                                "physical": physical,
                                "motor_audit": a.motor_audit(),
                                "runtime_cpu_s": p["note"]["cpu_s"],
                            }
                        results[-1]["trials"][mode] = dict(
                            plan=p["note"], search=p["trace"], **played
                        )
                        print(
                            c["id"],
                            leg,
                            tick,
                            mode,
                            p["note"],
                            played["legal"],
                            played["answered"],
                            flush=True,
                        )
                out["rows"].append(
                    {
                        "card": c,
                        "case": case,
                        "stock": offered,
                        "shot": row["shot"],
                        "stock_score": row["score"],
                        "results": results,
                    }
                )
                _write(args.out, out)
        finally:
            stereo.close()
            engine.twin.close()
            a.close()


def held(args):
    B = _read(args.law)
    cards = _read(args.plan)[args.group][args.start : args.start + args.limit]
    src = _read(core.SCHOOL)
    if {c["id"] for c in cards} & {c["id"] for c in src["cards"]}:
        raise ValueError("Target development release in the assessment cohort")
    core.joint.RECEIVE_Y = src["depth"]
    surface = core.setup(src["depth"])
    model = _read(core.PUPIL)
    shots = core.measured_launches(args.feeds)
    checkpoint = _read(core.ROOT / "tt37/independent-cold-value.json.gz")
    with np.load(core.ROOT / "tt37/self-combined.npz") as bank:
        pupil = core.DeepField(bank, "local")
    params = core.clinic.generate(src["samples"])
    variants = {mode: Lift(B, mode, args.batch, args.rounds) for mode in args.modes}
    out = {
        "schema": "tt41-bounce-return-assessment-v1",
        "law_sha256": _hash(args.law),
        "plan_sha256": _hash(args.plan),
        "group": args.group,
        "modes": args.modes,
        "cards": cards,
        "cap": args.cap,
        "batch": args.batch,
        "rounds": args.rounds,
        "rows": [],
    }
    with core.centered(src["ready"]):
        a = core.arena()
        engine = core.OpticalClinic(body=True)
        value = core.MotorValue(_read(Path(src["motor_value"])))
        stereo, rig, stride, lo, hi = core.joint.resources(a)
        try:
            for c in cards:
                rows = {}
                for mode, lift in variants.items():
                    engine.motor_value = (
                        ScoreTap(value, lift) if mode == "online" else value
                    )
                    rows[mode] = core.clinic.point(
                        a,
                        stereo,
                        rig,
                        model,
                        surface,
                        stride,
                        lo,
                        hi,
                        c,
                        shots[c["id"]],
                        list(checkpoint["bank"]),
                        params,
                        "optical-fixed",
                        args.cap,
                        src["cut"],
                        src["rounds"],
                        True,
                        None,
                        None,
                        True,
                        False,
                        None,
                        engine,
                        src["visual"],
                        pupil=pupil,
                        economy=True,
                        impact=None if mode == "fixed" else lift,
                    )
                    r = rows[mode]
                    print(
                        c["id"],
                        mode,
                        r["score"]["learner"],
                        round(r["cpu_s"], 2),
                        [
                            (
                                n["tick"],
                                n["impact_margin"]["found"],
                                n["impact_margin"].get("style"),
                            )
                            for n in r["records"]
                            if n.get("impact_margin", {}).get("active")
                        ],
                        flush=True,
                    )
                out["rows"].append({"card": c, "results": rows})
                out["learned"] = {
                    mode: lift.feedback_history
                    for mode, lift in variants.items()
                    if lift.feedback_history
                }
                _write(args.out, out)
        finally:
            stereo.close()
            engine.twin.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("witness", "held"))
    p.add_argument("out", type=Path)
    p.add_argument("--law", type=Path, default=OUT / "effect.json")
    p.add_argument("--batch", type=int, default=78)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--first", action="store_true")
    p.add_argument("--plan", type=Path, default=core.ROOT / "tt39/endurance-plan.json")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--group", default="assessment")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--cap", type=int, default=200)
    p.add_argument(
        "--modes", nargs="+", choices=MODES, default=["fixed", "cold", "solo", "online"]
    )
    cfg = p.parse_args()
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    (witness if cfg.task == "witness" else held)(cfg)
