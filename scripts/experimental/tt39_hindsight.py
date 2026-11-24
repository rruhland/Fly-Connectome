"""Learn a direct action prior from physically verified repairs of scoreless practice."""

import argparse
from collections import Counter
from pathlib import Path
from time import process_time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt29_vision import Flight
from .tt31_curriculum import card, opening_candidates
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_joint_capture import observed
from .tt37_joint_deep import DeepField
from .tt37_joint_report import paired, read_complete, replay_case
from .tt37_opponent import arena
from .tt37_visual_inertia import LearnedOptics
from .tt38_flight_fit import smooth
from .tt38_joint_mpc_report import metrics


OUT = Path("artifacts/tt39")
OPTICS = Path("artifacts/tt37/rebound.npz")
ROOT_PLAN = Path("artifacts/tt37/opponent-plan.json")
PRACTICE_ITEMS = (0, 1, 3, 5, 8, 11, 12, 14, 15, 16)
DT = mpc.DT


def paths(folder, stage):
    return folder / f"hindsight-{stage}.json.gz"


def freeze_plan(folder, feeds, optic):
    sources = [ROOT_PLAN, Path("artifacts/tt37/school-plan.json"),
               Path("artifacts/tt39/endurance-plan.json"), Path("artifacts/tt37/clinic-learning-plan.json"),
               Path("artifacts/tt37/memory-course.json"), mpc.BANK.with_suffix(".json")]
    serial = _read(ROOT_PLAN)
    exclude = set()
    def cards(value):
        if isinstance(value, dict):
            if isinstance(value.get("id"), int) and "seed" in value:
                exclude.add(value["id"])
            for v in value.values():
                cards(v)
        if isinstance(value, list):
            for v in value:
                cards(v)
    for p in sources:
        s = _read(p)
        cards(s)
        exclude.update(i for i in s.get("exclusions", []) if isinstance(i, int))
    rng = np.random.default_rng(39104120)
    def new(sid, j):
        return dict(card(int(sid), "full", j, int(rng.integers(1, 2**30))), profile="reach",
                    opening="random", reply_seed=int(rng.integers(1, 2**30)))
    practice = [new(serial["practice"][i]["id"], j) for j, i in enumerate(PRACTICE_ITEMS)]
    available = [s["id"] for s in opening_candidates(measured_launches(feeds)) if s["id"] not in exclude]
    if len(available) < 8:
        raise ValueError("Not enough completely disjoint measured releases")
    held = [new(i, j) for j, i in enumerate(rng.choice(available, size=8, replace=False))]
    _write(folder / "hindsight-plan.json", dict(
        practice=practice, fresh=held, feed_sha256=_hash(feeds), optics_sha256=_hash(optic),
        bank_sha256=_hash(mpc.BANK), prior_exclusions=len(exclude),
        selection="Ten predeclared random practice offerings. At most six zero-score, low-first optical arcs "
                  "get at most two candidate checkpoints each. Eight unscreened, disjoint assessment releases.",
    ))


def start_eye(rig, dt, optic):
    eye = Vision(rig, dt, raw=True)
    eye.eye = LearnedOptics(eye.eye, optic)
    return eye


def packet(eye, renderer, tick, q, ref):
    x = observed(eye, tick, renderer.capture() if tick % 2 == 0 else None, q)
    f = eye.eye.latest
    tail = eye.eye.filter.track
    xyz = None
    if tail and abs(tail[-1][0] - tick * DT) < 1e-9:
        xyz = tail[-1][1].tolist()
    return dict(q=q.tolist(), ref=ref.tolist(), x=None if x is None else x.tolist(), xyz=xyz,
                damp=None if f is None else float(f.damping), bounced=bool(f is not None and f.bounced))


def row(a, shot, c, actions):
    return dict(shot=shot, opponent_seed=c["seed"], score=audit(a.events, opening_side=1), events=a.events,
                placements=a.point_placements(), actions=actions, motor_audit=a.motor_audit())


def capture(folder, feeds, optic):
    plan_path = folder / "hindsight-plan.json"
    d = _read(plan_path)
    if (_hash(feeds), _hash(optic), _hash(mpc.BANK)) != (
            d["feed_sha256"], d["optics_sha256"], d["bank_sha256"]):
        raise ValueError("Practice sources changed after schedule freeze")
    ready = _read(mpc.BANK.with_suffix(".json"))["ready"]
    with np.load(mpc.BANK) as p:
        raw = DeepField(p, "local")
    cases = measured_launches(feeds)
    out = dict(cards=d["practice"], ready=ready, rows=[], plan_sha256=_hash(plan_path))
    with centered(ready):
        a = arena()
        camera, rig, stride, low, high = resources(a)
        try:
            for c in out["cards"]:
                shot = cases[c["id"]]
                a.launch(shot, card=c)
                a.partner.reset(seed=c["seed"])
                eye = start_eye(rig, a.control_dt, optic)
                home = np.asarray(ready)
                ref = home.copy()
                actions, sensory = [], []
                incoming = False
                for tick in range(min(500, a.max_steps)):
                    q = a.joints
                    stamp = packet(eye, camera, tick, q, ref)
                    sensory.append(stamp)
                    x = stamp["x"]
                    goal = home
                    if x is not None and x[4] < -.9 and x[1] < -.90:
                        incoming = True
                    if x is not None and (x[4] < -.9 or (incoming and x[1] < .50)):
                        u, confidence = raw.get(x, q, reference=ref)
                        if u is not None:
                            goal = home + confidence * (u - home)
                    action = np.clip((np.clip(goal, low, high) - ref) / stride, -.8, .8)
                    actions.append(action.tolist())
                    ref = np.clip(ref + action * stride, low, high)
                    _, _, done, capped, _ = a.exchange(action, FAR_COMMAND)
                    if done or capped or a.legal_returns > 1:
                        break
                r = row(a, shot, c, actions)
                r.update(sensory=sensory, ended=bool(done), camera_height_m=camera_height(sensory))
                out["rows"].append(r)
                _write(paths(folder, "practice"), out)
                print("practice", c["id"], r["score"]["learner"], r["camera_height_m"], flush=True)
        finally:
            camera.close()
            a.close()


def camera_height(sensory):
    """A triage measurement, not an interception target or actor input."""
    raw = [p["xyz"] for p in sensory if p["xyz"] is not None]
    for a, b in zip(raw, raw[1:]):
        if -1.0 > a[1] >= -1.10 > b[1] > -1.27:
            blend = (-1.10 - a[1]) / (b[1] - a[1])
            return float(a[2] + (b[2] - a[2]) * blend)
    return None


def future_state(sensory, tick):
    """Recover an earlier inbound state from a later smooth post-bounce camera arc."""
    causal = sensory[tick]["x"]
    if causal is None or causal[4] > -.9 or not -.99 < causal[1] < -.63:
        return None
    views = [(k, p["xyz"]) for k, p in enumerate(sensory[tick + 2:tick + 17], tick + 2)
             if p["xyz"] is not None and -1.79 < p["xyz"][1] < -.68]
    # Prefer the longest reliable visible segment, never fit through a contact or bounce.
    for stop in range(len(views), 3, -1):
        stamp = np.asarray([k for k, _ in views[:stop]], dtype=float)
        if max(np.diff(stamp)) > 3 or stamp[-1] - stamp[0] < 6:
            continue
        p = np.asarray([v for _, v in views[:stop]])
        if np.any(np.diff(p[:, 1]) > -.018):
            continue
        times = (stamp - tick) * DT
        end, speed, accel, error, _ = smooth(times, p)
        before = np.r_[end - times[-1] * speed + .5 * times[-1]**2 * accel, speed - times[-1] * accel]
        if (error < .0048 and -5.5 < before[4] < -1.1 and before[5] > .12
                and .767 < before[2] < 1.38
                and np.linalg.norm(before[:3] - causal[:3]) < .053
                and np.linalg.norm(before[3:] - causal[3:]) < 1.85):
            return dict(tick=tick, reconstructed=before.tolist(), future_ticks=stamp.astype(int).tolist(),
                        fit_max_m=float(error), change=(before - causal).tolist())
    return None


def anchors(sensory):
    good = [fit for i in range(0, len(sensory)) if (fit := future_state(sensory, i)) is not None]
    if not good:
        return []
    out = [good[0]]
    later = next((v for v in good[1:] if v["tick"] >= out[0]["tick"] + 4), None)
    if later:
        out.append(later)
    return out


def flight(sample, tick):
    if sample["x"] is None:
        return None
    x = np.asarray(sample["x"])
    return Flight(x[:3], x[3:], tick * DT, sample["damp"] or .3, .4, .01, 5, sample["bounced"])


def propose(actor, sensory, attempt, refit):
    tick = attempt["tick"]
    actor.begin()
    for i, sample in enumerate(sensory[:tick + 1]):
        q, ref = np.array(sample["q"]), np.array(sample["ref"])
        f = flight(sample, i)
        actor.engine.observe(i, SimpleNamespace(latest=f), q, ref)
    state = np.array(sensory[tick]["x"])
    if refit:
        state = np.array(attempt["reconstructed"])
        actor.engine.latest = Flight(state[:3], state[3:], tick * DT, f.damping, .4, .01, 5, True)
        actor.engine.bounced = True
    goal, conf = actor.raw.get(np.asarray(sensory[tick]["x"]), q, reference=ref)
    base = actor.priors.home if goal is None else actor.priors.home + conf * (goal - actor.priors.home)
    actor.search(tick, state, q, ref, base)
    note = actor.notes[-1] if actor.notes else {}
    valid = note.get("chosen") is not None and note.get("robust", 0) >= 1
    return (actor.best.copy() if valid else None), dict(note, trials=actor.diag["native_candidates"])


def verify(a, stride, low, high, ready, c, stock, tick, goals):
    """All physics comes from a fresh measured launch and replayed actual commands."""
    a.launch(stock["shot"], card=c)
    a.partner.reset(seed=c["seed"])
    rem = np.asarray(ready).copy()
    actions = []
    for i in range(min(500, a.max_steps)):
        if i < tick:
            action = np.array(stock["actions"][i])
        elif i < tick + len(goals):
            if i == tick:
                ref = stock["sensory"][tick]
                prev = [e for e in stock["events"] if e["time"] <= tick * DT + 1e-9]
                if (digest_json(a.events) != digest_json(prev)
                        or not np.allclose(a.joints, ref["q"], atol=1e-9, rtol=0.)):
                    raise ValueError("The live replay did not reach its archived practice prefix")
                if not np.array_equal(rem, ref["ref"]):
                    raise ValueError("Replayed actuator references differ")
            action = np.clip((np.clip(goals[i - tick], low, high) - rem) / stride, -.8, .8)
        else:
            action = np.zeros(6)
        actions.append(action.tolist())
        rem = np.clip(rem + action * stride, low, high)
        _, _, done, capped, _ = a.exchange(action, FAR_COMMAND)
        if done or capped or a.legal_returns > 1:
            break
    if len(actions) <= tick:
        raise ValueError("The physical referee terminated before the branch point")
    result = row(a, stock["shot"], c, actions)
    result.update(fork_tick=tick, exact_prefix=True)
    return result


def labels(a, camera, rig, stride, low, high, ready, c, physical, optic):
    a.launch(physical["shot"], card=c)
    a.partner.reset(seed=c["seed"])
    eye = start_eye(rig, a.control_dt, optic)
    ref = np.asarray(ready)
    history, issued = [], []
    for tick, u in enumerate(physical["actions"]):
        history.append(packet(eye, camera, tick, a.joints, ref))
        ref = np.clip(ref + np.asarray(u) * stride, low, high)
        issued.append(ref.copy())
        a.exchange(u, FAR_COMMAND)
    if (digest_json(a.events) != digest_json(physical["events"])
            or digest_json(a.point_placements()) != digest_json(physical["placements"])):
        raise ValueError("Independent success tape failed exact scoring")
    hit = next(s["hit"] for s in audit(a.events, opening_side=1)["shots"] if s["side"] == 1)
    samples = []
    for k in range(max(0, physical["fork_tick"] - 7), len(history), 2):
        r = history[k]
        if r["x"] is None or r["x"][4] > -.9 or not -.42 < k * DT - hit < -.02:
            continue
        samples.append(dict(key=mpc.query(r["x"], np.array(r["q"]), np.array(r["ref"])).tolist(),
                            actions=mpc.tail(np.asarray(issued[k:k + mpc.HORIZON]), mpc.HORIZON).tolist(),
                            source_tick=k))
    return samples


def teach(folder, optic):
    src = paths(folder, "practice")
    data = read_complete(src)
    plan = folder / "hindsight-plan.json"
    d = _read(plan)
    if (data["plan_sha256"] != _hash(plan) or data["cards"] != d["practice"]
            or d["optics_sha256"] != _hash(optic) or d["bank_sha256"] != _hash(mpc.BANK)):
        raise ValueError("The optical practice lineage changed before converting misses")
    options = []
    for c, r in zip(data["cards"], data["rows"], strict=True):
        if r["ended"] and not r["score"]["learner"] and r["placements"]["delivered"]:
            available = anchors(r["sensory"])
            if available:
                options.append((c, r, available))
    options.sort(key=lambda v: v[1]["camera_height_m"] if v[1]["camera_height_m"] is not None else 10)
    options = options[:6]
    ready = data["ready"]
    out = dict(cards=[c for c, _, _ in options], rows=[], practice_sha256=_hash(src),
               attempted_zero_balls=len(options), bank_sha256=_hash(mpc.BANK), optics_sha256=_hash(optic))
    learned = []
    with np.load(mpc.BANK) as doc:
        raw = DeepField(doc, "local")
    with centered(ready):
        a = arena()
        camera, rig, stride, low, high = resources(a)
        actor = mpc.MPC(mpc.BANK, raw, variant="elastic", budget=34)
        try:
            for c, r, fits in options:
                log = dict(id=c["id"], camera_height_m=r["camera_height_m"], results=[])
                successful = []
                for fit in fits:
                    item = dict(fit=fit)
                    for label, mode in (("causal", False), ("hindsight", True)):
                        goal, note = propose(actor, r["sensory"], fit, mode)
                        actual = None if goal is None else verify(a, stride, low, high, ready, c, r, fit["tick"], goal)
                        item[label] = dict(search=note, physical=actual)
                        if label == "hindsight" and actual is not None and actual["score"]["learner"]:
                            successful.append((actual, fit))
                    log["results"].append(item)
                if successful:
                    actual, fit = successful[0]
                    correction = dict(state=r["sensory"][fit["tick"]]["x"], delta=fit["change"])
                    learned.append(dict(id=c["id"], correction=correction,
                                        commands=labels(a, camera, rig, stride, low, high, ready, c, actual, optic),
                                        events_sha256=digest_json(actual["events"]), exact_prefix=True))
                log["hindsight_converted"] = bool(successful)
                log["causal_converted"] = any(p["causal"]["physical"] and p["causal"]["physical"]["score"]["learner"]
                                              for p in log["results"])
                out["rows"].append(log)
                _write(paths(folder, "verification"), out)
                print("verify", c["id"], log["causal_converted"], log["hindsight_converted"], flush=True)
        finally:
            actor.close()
            camera.close()
            a.close()
    _write(paths(folder, "verification"), out)
    _write(paths(folder, "bank"), dict(complete=True, rows=learned, count=len(learned),
                                     verified_sha256=_hash(paths(folder, "verification")),
                                     bank_sha256=_hash(mpc.BANK), optics_sha256=_hash(optic)))


class HindsightPrior(mpc.MPC):
    """Reuse only real-scoring fragment labels. The retained inputs are causal."""

    def __init__(self, *args, learned, **kwargs):
        super().__init__(*args, **kwargs)
        self.learned = [(np.asarray(s["key"]), np.asarray(s["actions"]), r)
                        for r in learned for s in r["commands"]]

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        note = self.notes[-1]
        if note.get("chosen") is not None and note.get("robust", 0) == 2:
            return
        t = mpc.query(x, q, reference)
        nearby = sorted(((np.linalg.norm(key - t), g, r) for key, g, r in self.learned),
                        key=lambda a: a[0])
        possible = []
        associated = []
        used = set()
        for d, g, r in nearby:
            if d > 1.45 or len(possible) == 3:
                break
            if r["id"] not in used:
                used.add(r["id"])
                possible.append(g)
                associated.append(r["correction"])
        if not possible or not self.engine.sync():
            return
        started = process_time()
        self.diag["hindsight_queries"] += 1
        _, _, trace, _, _ = self.trial(possible, traces=True)
        for g, t in zip(list(possible), trace, strict=True):
            shift = self.transport(t)
            if shift is not None:
                possible.append(g + .55 * shift)
        best = self.risk_screen(possible, need=1)
        kind = "ordinary_optics"
        if best is None:
            # An earlier fit is reusable only near that measured incoming region.
            scale = np.array([.14, .10, .12, .70, .80, .75])
            errors = [np.linalg.norm((x - c["state"]) / scale) for c in associated]
            k = int(np.argmin(errors))
            if errors[k] < 1.20:
                change = np.clip(np.asarray(associated[k]["delta"]), -.42, .42)
                change[:3] = np.clip(change[:3], -.022, .022)
                change *= .68 * np.exp(-.5 * errors[k]**2)
                t = self.engine.twin
                p, v = t.ball.copy(), t.velocity.copy()
                t.ball[:] = p + change[:3]
                t.velocity[:] = v + change[3:]
                try:
                    self.diag["hindsight_residual_queries"] += 1
                    best = self.risk_screen(possible, need=2)
                    kind = "bounded_learned_optical_residual"
                finally:
                    t.ball[:] = p
                    t.velocity[:] = v
        self.diag["hindsight_cpu_s"] += process_time() - started
        if best is None:
            return
        index, agreement, quality, delay = best
        if note.get("chosen") is not None and quality > note["score"] - .40:
            return
        self.best, self.tick, self.lease = possible[index], tick, mpc.LEASE
        note.update(chosen="verified_hindsight", robust=agreement, score=quality,
                    contact_delay_s=delay, prior_screen=kind)
        self.diag["hindsight_issued"] += 1

    def risk_screen(self, options, *, need):
        loss, legal, _, y, _ = self.trial(options, long=True)
        indices = np.flatnonzero(legal)
        if not len(indices):
            return None
        support, quality = self.uncertainty([options[i] for i in indices], loss[indices])
        allowed = np.flatnonzero(support >= need)
        if not len(allowed):
            return None
        winner = allowed[np.argmin(quality[allowed])]
        index = int(indices[winner])
        return index, int(support[winner]), float(quality[winner]), float(y[index, 1])


def assess(folder, mode, feeds, optic):
    p = paths(folder, "bank")
    learned = _read(p)
    if not learned["count"] or not any(r["commands"] for r in learned["rows"]):
        raise ValueError("No real-practice conversions supplied a joint memory; stop without holding out")
    if (learned["verified_sha256"] != _hash(paths(folder, "verification"))
            or learned["bank_sha256"] != _hash(mpc.BANK) or learned["optics_sha256"] != _hash(optic)):
        raise ValueError("The frozen conversion evidence changed")
    plan = folder / "hindsight-plan.json"
    course = _read(plan)
    if course["feed_sha256"] != _hash(feeds) or course["optics_sha256"] != _hash(optic):
        raise ValueError("Assessment camera or measured launches changed")
    if {r["id"] for r in course["fresh"]} & {r["id"] for r in course["practice"]}:
        raise ValueError("A physical practice recording leaked into assessment")
    dest = paths(folder, mode)
    factory = mpc.MPC if mode == "fixed" else lambda *a, **k: HindsightPrior(*a, learned=learned["rows"], **k)
    with patch.object(mpc, "MPC", factory):
        mpc.run(feeds, mpc.BANK, plan, "fresh", "frozen", dest, optic=optic, variant="elastic", cap=24)
    out = read_complete(dest)
    out["hindsight_sha256"] = _hash(p) if mode != "fixed" else None
    _write(dest, out)


def report(folder):
    a = read_complete(paths(folder, "verification"))
    bank = _read(paths(folder, "bank"))
    v = dict(practice_cases=len(_read(paths(folder, "practice"))["rows"]), qualifying=len(a["rows"]),
             causal=int(sum(bool(r["causal_converted"]) for r in a["rows"])),
             hindsight=int(sum(bool(r["hindsight_converted"]) for r in a["rows"])),
             only_hindsight=sum(r["hindsight_converted"] and not r["causal_converted"] for r in a["rows"]),
             action_samples=sum(len(r["commands"]) for r in bank["rows"]))
    if paths(folder, "fixed").exists() and paths(folder, "learned").exists():
        fixed, live = (read_complete(paths(folder, mode)) for mode in ("fixed", "learned"))
        if (live["hindsight_sha256"] != _hash(paths(folder, "bank"))
                or live["source_sha256"] != fixed["source_sha256"] or live["optical_sha256"] != fixed["optical_sha256"]):
            raise ValueError("Frozen policy comparison is not matched")
        v["fresh"] = dict(fixed=metrics(fixed), learned=metrics(live), comparison=paired(fixed, live))
    _write(folder / "hindsight-results.json", v)


def audit_estimate(folder):
    """Use solver state only after learning, to measure errors and verify outcomes."""
    verification_path = paths(folder, "verification")
    verification = read_complete(verification_path)
    practice = read_complete(paths(folder, "practice"))
    data = {c["id"]: r for c, r in zip(practice["cards"], practice["rows"], strict=True)}
    readings = []
    recovered = None
    with centered(practice["ready"]):
        a = arena()
        try:
            for c, doc in zip(verification["cards"], verification["rows"], strict=True):
                old = data[c["id"]]
                at = {i["fit"]["tick"]: i["fit"] for i in doc["results"]}
                a.launch(old["shot"], card=c)
                a.partner.reset(seed=c["seed"])
                for tick, u in enumerate(old["actions"]):
                    if tick in at:
                        true = np.r_[a.ball.copy(), a.velocity.copy()]
                        causal = np.array(old["sensory"][tick]["x"])
                        exact = np.array(at[tick]["reconstructed"])
                        readings.append(dict(
                            id=c["id"], tick=tick, causal_error=(causal - true).tolist(),
                            hindsight_error=(exact - true).tolist(),
                        ))
                    a.exchange(u, FAR_COMMAND)
                if digest_json(a.events) != digest_json(old["events"]):
                    raise ValueError("Audit of the original missed point diverged")
                if recovered is None:
                    for stage in doc["results"]:
                        r = stage["hindsight"]["physical"]
                        if r is not None and r["score"]["learner"]:
                            recovered = c, r
                            break
        finally:
            a.close()
    errors = ({k: np.mean(np.abs([r[k] for r in readings]), axis=0).tolist()
               for k in ("causal_error", "hindsight_error")} if readings else {})
    proof = None if recovered is None else replay_case(practice["ready"], "random", *recovered)
    _write(folder / "hindsight-audit.json", dict(
        note="This audit uses simulator truth only to evaluate estimates after the bank was already frozen.",
        verification_sha256=_hash(verification_path), snapshots=readings, component_mae=errors,
        independently_replayed_recovered_point=proof,
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "capture", "teach", "fixed", "learned", "report", "audit"))
    p.add_argument("--folder", type=Path, default=OUT)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--optic", type=Path, default=OPTICS)
    args = p.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        freeze_plan(args.folder, args.feeds, args.optic)
    elif args.stage == "capture":
        capture(args.folder, args.feeds, args.optic)
    elif args.stage == "teach":
        teach(args.folder, args.optic)
    elif args.stage == "report":
        report(args.folder)
    elif args.stage == "audit":
        audit_estimate(args.folder)
    else:
        assess(args.folder, args.stage, args.feeds, args.optic)
