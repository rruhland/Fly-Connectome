"""Receding, camera-initialized joint reference search with no contact plane or stroke chart."""

import argparse
import hashlib
from collections import Counter, deque
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, measured_launches
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_joint_capture import observed
from .tt37_joint_deep import DeepField, cards_for
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt37_visual_inertia import LearnedOptics


ROOT = Path("artifacts")
OUT = ROOT / "tt38"
BANK = ROOT / "tt37" / "long-rgb.npz"
PLAN = ROOT / "tt37" / "opponent-plan.json"
AIM = np.array([0., 1.30, 1.07, 3.55])
DT = 1. / 60.
HORIZON = 21
TAIL = 59
LEASE = 5
OFFSET = np.array([.12, .12, .14, .18, .20])
OPEN_NORM = np.array([.16, .15, .13, .85, 1.1, 1.3, 1.0, .9, 1.05, .86, .95, .32, .32, .36, .38, .4])
DEVELOPMENT_INDICES = (0, 1, 3, 5, 8, 12)
ASSESSMENT_LIMITS = {"held_openings": 12, "held_replies": 8}
GUARDED_INDICES = (12, 13, 14, 15, 17, 19, 21, 23)


def query(x, q, reference):
    return np.r_[x, q[:5], reference[:5] - q[:5]] / OPEN_NORM


def tail(goals, frames):
    if len(goals) >= frames:
        return goals[:frames].copy()
    return np.vstack((goals, np.broadcast_to(goals[-1], (frames - len(goals), goals.shape[1]))))


def protocol(bank, plan, feeds, optic, output):
    course = _read(plan)
    development = [dict(course["practice"][i], reply_seed=3831100 + 13 * j)
                   for j, i in enumerate(DEVELOPMENT_INDICES)]
    _write(output, dict(
        held={k: course[k][:n] for k, n in ASSESSMENT_LIMITS.items()}, development=development,
        plan_sha256=_hash(plan), feeds_sha256=_hash(feeds), prior_sha256=_hash(bank),
        optics_sha256=_hash(optic) if optic else None,
        keep_frames=HORIZON, across_point_capacity=192, plan_lease_frames=LEASE,
        selection="Variants compared only on six development cases. Held comparison uses same RGB "
                  "preprocessing, packet of training examples, and identical card/seed pairs; live "
                  "starts empty on each deck and remembers only reward-credited observed joint commands.",
    ))


def guarded_protocol(bank, plan, feeds, optic, output):
    course = _read(plan)
    cards = [course["held_openings"][i] for i in GUARDED_INDICES]
    receipt = _read(bank.with_suffix(".json"))
    used = {c["id"] for c in course["held_openings"][:ASSESSMENT_LIMITS["held_openings"]]}
    used.update(row["card"]["id"] for row in receipt["practice"])
    used.update(c["id"] for c in course["held_replies"][:ASSESSMENT_LIMITS["held_replies"]])
    if used & {card["id"] for card in cards}:
        raise ValueError("The new guarded course overlaps established tests or training")
    _write(output, dict(
        held={"guarded_challenge": cards}, plan_sha256=_hash(plan), feeds_sha256=_hash(feeds),
        prior_sha256=_hash(bank), optics_sha256=_hash(optic) if optic else None,
        provenance="Eight disjoint measured opening recordings from the reserved second half of TT37 novel. "
                   "Neither score nor any actor rollout informs which recordings to select.",
    ))


class Fragments:
    """A bank of short control priors. Native trials, not retrieval, authorize an action."""

    def __init__(self, path, engine):
        self.home = np.asarray(_read(path.with_suffix(".json"))["ready"])
        self.rows = []
        self.online = deque(maxlen=192)
        receipt = _read(path.with_suffix(".json"))
        if receipt["bank_sha256"] != _hash(path):
            raise ValueError("Training bank integrity mismatch")
        a = engine.twin
        axes = engine.actuators
        stride = a._joint_increments[axes] * engine.predict.micro
        low, high = a._joint_low[axes], a._joint_high[axes]
        with np.load(path) as bank:
            states, marks, targets = (bank[k] for k in ("states", "marks", "targets_now"))
        hashes, tapes = receipt["sources"], {}
        start = 0
        for i, row in enumerate(receipt["practice"]):
            group = row["source"]
            if group not in tapes:
                source = path.with_name(group)
                if _hash(source) != hashes[group]:
                    raise ValueError(f"Archived legal command tape changed: {group}")
                tapes[group] = _read(source)
            tape = tapes[group]["rows"][row["source_row"]]
            if tapes[group]["cards"][row["source_row"]] != row["card"]:
                raise ValueError("Archived card and reward provenance mismatch")
            issued = np.zeros((len(tape["actions"]) + 1, 6))
            issued[0] = self.home
            for tick, u in enumerate(tape["actions"]):
                issued[tick + 1] = np.clip(issued[tick] + stride * u, low, high)
            stop = start + row["frames"]
            idx = marks[start:stop, 0].astype(np.int64)
            if not np.allclose(issued[idx + 1], targets[start:stop], atol=1e-5):
                raise ValueError("The observed labels and reconstructed issued controls differ")
            for k in range(start, stop):
                x, q = states[k, :6], states[k, 6:11]
                tick = int(marks[k, 0])
                age = float(marks[k, 1])
                if -.48 < age < -.012 and x[4] < -.90:
                    seq = tail(issued[tick + 1:tick + 1 + HORIZON], HORIZON)
                    # Group by physical legal hit for diverse proposals; not the current episode ID.
                    leg = i, round((tick * DT - age) * 480)
                    self.rows.append((query(x, q, issued[tick]), seq, leg))
            start = stop
        self.tree = cKDTree(np.asarray([r[0] for r in self.rows]))

    def priors(self, x, q, reference, limit):
        key = query(x, q, reference)
        d, ids = self.tree.query(key, min(len(self.rows), 160))
        found = [(dist, self.rows[i][1], self.rows[i][2], False) for dist, i in zip(d, ids, strict=True)]
        found.extend((max(0., np.linalg.norm(state - key) - .10), goals, label, True)
                     for state, goals, label in self.online)
        selected, used = [], set()
        for dist, goals, label, online in sorted(found, key=lambda r: r[0]):
            if label in used:
                continue
            if dist > 3.1 and len(selected) >= 2:
                break
            selected.append((goals.copy(), online))
            used.add(label)
            if len(selected) >= limit:
                break
        return selected

    def keep(self, archive, leg):
        near = [i for i, (_, x, _, _, _) in enumerate(archive)
                if x is not None and x[4] < -.9 and x[1] < -.68]
        if not near:
            return 0
        last = near[-1]
        snapshots = 0
        for i in range(max(0, last - 16), last - 1, 3):
            _, x, q, prev, _ = archive[i]
            if x is None or x[4] > -.9 or x[1] > -.62:
                continue
            controls = np.asarray([r[-1] for r in archive[i:i + HORIZON]])
            self.online.append((query(x, q, prev), tail(controls, HORIZON), leg))
            snapshots += 1
        return snapshots

    def digest(self):
        digest = hashlib.sha256()
        for state, goals, tag in self.online:
            digest.update(np.asarray(state, dtype="<f8").tobytes())
            digest.update(np.asarray(goals, dtype="<f8").tobytes())
            digest.update(repr(tag).encode())
        return dict(rows=len(self.online), state_sha256=digest.hexdigest())


def cost(scores, response):
    """Proxies use nominal forward ball consequences, never live opponent state."""
    legal = np.isfinite(scores[:, 0])
    separate = response[:, 1] > 0
    impact, v = response[:, 5:8], response[:, 2:5]
    time = (v[:, 2] + np.sqrt(np.maximum(0., v[:, 2]**2 + 19.62 * (impact[:, 2] - .779)))) / 9.81
    time = .94 * np.maximum(time, .01)
    p = impact[:, :2] + v[:, :2] * time[:, None]
    tnet = -impact[:, 1] / np.maximum(.6, .94 * v[:, 1])
    h = impact[:, 2] + v[:, 2] * tnet - 4.905 * tnet**2
    weak = (3.8 * np.maximum(0., np.abs(p[:, 0]) - .42) + 1.75 * np.abs(p[:, 1] - .74)
            + 4.8 * np.maximum(0., .978 - h) + .6 * np.maximum(0., 2.05 - v[:, 1])
            + 3.8 * response[:, 8])
    close = np.where(response[:, 0] >= 1.9, 1., response[:, 0])
    miss = np.where(separate, 5.0 + weak, 9. + 22. * close)
    playable = (3.7 * np.abs(scores[:, 4] - .70) + 4.1 * np.abs(scores[:, 3])
                + 3.8 * np.maximum(np.abs(scores[:, 6]) - .27, 0.)
                + 2.6 * np.maximum(1.09 - scores[:, 7], 0.) + 1.9 * np.abs(scores[:, 8] - 1.08)
                + .65 * np.abs(scores[:, 9] - 3.46) + 9. * np.maximum(.052 - scores[:, 5], 0.)
                + 16. * np.maximum(scores[:, 12] - .035, 0.))
    return np.where(legal, playable - 5.0, miss), legal


def structured(goals, noise, scale=1.):
    progress = (1 + np.arange(HORIZON)) / HORIZON
    ramp = np.minimum(1., 3.5 * progress)
    slope = 2. * progress - 1.
    return (goals + np.c_[noise[:5] * ramp[:, None] * OFFSET,
                         np.zeros(HORIZON)] * scale
            + np.c_[noise[5:] * (ramp * slope)[:, None] * OFFSET, np.zeros(HORIZON)] * (.8 * scale))


class MPC:
    def __init__(self, bank, raw, *, variant, budget=28):
        self.engine = OpticalClinic(body=True)
        # Pure nominal forward kinematics and local differential. No inverse fit or pose table.
        self.geometry = BladeIK(self.engine.twin)
        self.priors = Fragments(bank, self.engine)
        self.raw = raw
        self.budget = budget
        self.variant = variant
        self.serial = 0

    def begin(self):
        self.engine.previous = None
        self.engine.bounced = False
        self.best = None
        self.tick = -99
        self.lease = LEASE
        self.next = 0
        self.rng = np.random.default_rng(382785)
        self.diag = Counter()
        self.notes = []
        self.serial += 1

    def close(self):
        self.engine.twin.close()

    def trial(self, goals, *, long=False, traces=False):
        g = np.asarray([tail(x, HORIZON + (TAIL if long else 0)) for x in goals])
        feedback = np.zeros((len(g), 9))
        track = np.full((*g.shape[:2], 17), np.nan) if traces else None
        _, y = self.engine.trajectories(g, AIM, HORIZON * DT, feedback=feedback, traces=track)
        self.diag["native_candidates"] += len(g)
        loss, legal = cost(y, feedback)
        return loss, legal, track, y, feedback

    def transport(self, trace, *, face=False):
        use = np.flatnonzero(np.isfinite(trace[:, 0]) & (trace[:, 1] < -.78))
        if not len(use):
            return None
        p = trace[use, :3]
        q = trace[use, 6:11]
        center = np.asarray([self.geometry.pose(row)[0] for row in q])
        ix = int(np.argmin(np.linalg.norm(p - center, axis=1)))
        b, normal, jac, angle = self.geometry.pose(q[ix], jacobian=True)
        delta = p[ix] - b
        if np.linalg.norm(delta) > .34:
            return None
        # Move toward the sampled flight at the solution's own closest moment.
        # A small ball radius offset is relative to the paddle, not a table-plane intercept.
        delta += normal * (.021 if self.variant == "fragment" else -.020)
        axes = np.array([1., .32, .86, .8, .60])
        if face:
            elapsed = np.clip((.71 - p[ix, 1]) / 4.10, .39, .67)
            wanted = (np.array([0., .71, .783]) - p[ix]) / elapsed
            wanted[2] += 4.905 * elapsed
            impulse = wanted - trace[use[ix], 3:6]
            n = impulse / max(.01, np.linalg.norm(impulse))
            delta = np.r_[delta, .093 * np.clip(n - normal, -.55, .55)]
            jac = np.vstack((jac, .093 * angle))
        j = jac * axes
        step = axes * (j.T @ np.linalg.solve(j @ j.T + .030**2 * np.eye(len(delta)), delta))
        ramp = np.minimum((1 + np.arange(HORIZON)) / max(3., use[ix] * .65), 1.)
        bound = .27 if self.variant == "fragment" else .37
        return np.c_[ramp[:, None] * np.clip(step, -bound, bound), np.zeros(HORIZON)]

    def pupil_prior(self, trace, base):
        last = base
        output = [last]
        for row in trace[1:]:
            if np.isfinite(row).all():
                command, conf = self.raw.get(row[:6], row[6:11])
                if command is not None:
                    last = self.priors.home + conf * (command - self.priors.home)
            output.append(last)
        return np.array(output)

    def uncertainty(self, goals, nominal):
        a = self.engine.twin
        ball, velocity = a.ball.copy(), a.velocity.copy()
        spin = a.data.qvel[a._ball_dof + 3:a._ball_dof + 6]
        old_spin = spin.copy()
        trials = []
        try:
            for side in (-1., 1.):
                a.ball[:] = ball + side * np.array([.0018, -.0023, .0028])
                a.velocity[:] = velocity + side * np.array([.023, .043, .073])
                spin[:] = old_spin + [16. + 17. * side, 0., 0.]
                loss, legal, *_ = self.trial(goals, long=True)
                trials.append((loss, legal))
        finally:
            a.ball[:] = ball
            a.velocity[:] = velocity
            spin[:] = old_spin
        agreement = np.sum([t[1] for t in trials], axis=0)
        margins = np.mean([np.where(t[1], t[0], 2.75 + .12 * np.minimum(t[0], 10.))
                           for t in trials], axis=0)
        return agreement, .43 * nominal + .57 * margins

    def robust(self, goals, rank, nominal):
        agreement, val = self.uncertainty(goals, nominal)
        if self.variant == "guarded":
            val = np.where(agreement == 2, val, val + 15.)
        winner = int(np.argmin(val))
        return int(rank[winner]), int(agreement[winner]), float(val[winner])

    def preserve(self, tick, base):
        """Check the currently issued branch before looking for a replacement."""
        if self.best is not None and 0 <= tick - self.tick < HORIZON - 2:
            warm = tail(self.best[tick - self.tick:], HORIZON)
            loss, valid, _, values, _ = self.trial([warm], long=True)
            if valid[0] and loss[0] < -1.25:
                agree, risk = self.uncertainty([warm], loss)
                if agree[0]:
                    self.best, self.tick, self.lease = warm, tick, LEASE + 3
                    self.diag["incumbent_kept"] += 1
                    return warm, ("warm_incumbent", int(agree[0]), float(risk[0]), values[0])
        # Roll the actual raw feedback law forward twice on nominal p/v and q.
        # This is not a constant motor command billed as the raw controller.
        raw = np.broadcast_to(base, (HORIZON, 6)).copy()
        for _ in range(2):
            _, _, trace, _, _ = self.trial([raw], traces=True)
            raw = self.pupil_prior(trace[0], base)
        loss, valid, _, values, _ = self.trial([raw], long=True)
        if valid[0]:
            agree, risk = self.uncertainty([raw], loss)
            if agree[0]:
                self.best = None
                self.diag["raw_preserved"] += 1
                return raw, ("raw_incumbent", int(agree[0]), float(risk[0]), values[0])
        return raw, None

    def search(self, tick, x, q, reference, base):
        start = process_time()
        if not self.engine.sync():
            return
        observed_bounce = self.engine.bounced
        guarded = self.variant == "guarded"
        elastic = self.variant != "fragment"
        raw_prior = None
        if guarded:
            raw_prior, winner = self.preserve(tick, base)
            if winner:
                name, robust, quality, values = winner
                self.diag["searches"] += 1
                self.diag["search_cpu_s"] += process_time() - start
                self.notes.append(dict(tick=tick, bounced=bool(observed_bounce), p=x[:3].tolist(),
                                       legal_trials=1, chosen=name, robust=robust, score=quality,
                                       contact_delay_s=float(values[1]), target_errors=values[3:10].tolist()))
                return
        # Three arms of the search: raw one-step feedback, warm replan, and local demonstrated actions.
        goals = [np.broadcast_to(base, (HORIZON, 6)).copy()]
        source = ["field"]
        if self.best is not None and tick - self.tick < HORIZON - 1:
            goals.append(tail(self.best[tick - self.tick:], HORIZON))
            source.append("warm")
        for seq, online in self.priors.priors(x, q, reference, 4 if guarded else 7):
            goals.append(seq)
            source.append("credited" if online else "demo")
        prior_loss, _, tr, _, _ = self.trial(goals, traces=True)
        seeds = [g.copy() for g in goals]
        anchors = list(np.argsort(prior_loss)[:min(3 if guarded else 6, len(goals))])
        if elastic:
            goals.append(raw_prior if raw_prior is not None else self.pupil_prior(tr[0], base))
            source.append("pupil_rollin")
        for i in anchors:
            adjust = self.transport(tr[i])
            if adjust is not None:
                second = self.transport(tr[i], face=True) if elastic else adjust
                scale = .64 if self.variant == "fragment" else .72
                goals.extend([seeds[i] + scale * adjust, seeds[i] + second])
                source.extend([source[i] + "_transport"] * 2)
        while len(goals) < self.budget:
            i = anchors[(len(goals) + tick) % len(anchors)]
            anchor = seeds[i]
            if len(goals) % 3 == 1 and len(goals) > len(seeds):
                anchor = goals[len(seeds) + ((len(goals) + tick) % (len(goals) - len(seeds)))]
            stretch = self.rng.uniform(.76, 1.24)
            ix = np.clip(np.arange(HORIZON) * stretch + self.rng.uniform(-2.2, 2.2), 0., HORIZON - 1.)
            sample = np.array([np.interp(ix, np.arange(HORIZON), anchor[:, k]) for k in range(6)]).T
            noise = np.clip(self.rng.normal(size=10), -2.12, 2.12)
            goals.append(structured(sample, noise, .72))
            source.append(source[i] + "_sample")
        near, _, _, _, f = self.trial(goals)
        rank = np.unique(np.r_[0, np.argsort(near)[:6 if guarded else 8], 1 if source[1:2] == ["warm"] else 0])
        loss, legal, _, y, _ = self.trial([goals[j] for j in rank], long=True)
        valid = np.flatnonzero(legal)
        if not guarded and not len(valid) and np.any(f[:, 1] > 0):
            # A second light, local sweep only when a contact had plausible geometry.
            elite = np.argsort(near)[:4]
            extra = [structured(goals[i], self.rng.normal(size=10), .42)
                     for i in elite for _ in range(3)]
            trial_loss, _, _, _, _ = self.trial(extra)
            order = np.argsort(trial_loss)[:5]
            loss2, legal2, _, y2, _ = self.trial([extra[j] for j in order], long=True)
            rank = np.r_[rank, len(goals) + order]
            goals.extend(extra)
            source.extend(["refine"] * len(extra))
            valid = np.flatnonzero(np.r_[legal, legal2])
            loss, legal, y = np.r_[loss, loss2], np.r_[legal, legal2], np.vstack((y, y2))
        self.diag["searches"] += 1
        self.diag["contact_trials"] += int(np.sum(f[:, 1] > 0))
        self.diag["legal_trials"] += int(np.sum(legal))
        note = dict(tick=tick, bounced=bool(observed_bounce), p=x[:3].tolist(),
                    contact_trials=int(np.sum(f[:, 1] > 0)), legal_trials=int(np.sum(legal)),
                    best_separation_m=float(np.min(f[:, 0])), chosen=None)
        if len(valid):
            best = valid[np.argsort(loss[valid])[:4]]
            choice, robust, score = self.robust([goals[i] for i in rank[best]], rank[best], loss[best])
            if not guarded or robust == 2:
                ix = int(np.flatnonzero(rank == choice)[0])
                self.best = goals[choice]
                self.tick = tick
                self.lease = LEASE + 3 if guarded else LEASE
                note.update(chosen=source[choice], robust=robust, score=score,
                            contact_delay_s=float(y[ix, 1]), target_errors=y[ix, 3:10].tolist())
                self.diag["accepted"] += 1
                self.diag["robust_accepted"] += bool(robust)
            else:
                self.diag["declined_weak_replacement"] += 1
        elif self.variant == "elastic" and x[1] > -1.30:
            sharp = np.array([k.startswith(("demo", "credited", "warm", "pupil")) for k in source[:len(f)]])
            possible = np.where(sharp & (f[:, 0] < .145) & (f[:, 0] + .018 < f[0, 0]))[0]
            if len(possible):
                chosen = possible[np.argmin(near[possible])]
                self.best = base + .52 * np.clip(goals[chosen] - base, -.40, .40)
                self.tick, self.lease = tick, 3
                note["prepare"] = source[chosen]
                self.diag["prepared"] += 1
        self.diag["search_cpu_s"] += process_time() - start
        self.notes.append(note)

    def step(self, tick, eye, x, q, reference, base):
        self.engine.observe(tick, eye, q, reference.copy())
        if x is not None and x[4] > .9:
            self.best = None
            return base
        early = self.variant != "fragment"
        if (x is not None and x[4] < -.9 and (-.49 if early else -.62) > x[1] > -1.62
                and x[2] > .77 and (self.engine.bounced or early) and tick >= self.next):
            self.next = tick + 5
            self.search(tick, x, q, reference, base)
        if self.best is not None and 0 <= tick - self.tick < self.lease:
            self.diag["screened_ticks"] += 1
            return self.best[tick - self.tick]
        return base


def play(a, renderer, rig, stride, lower, upper, ready, card, shot, raw, model, optical, mode, cap, progress=0):
    a.launch(shot, card=card)
    a.partner.reset(seed=card["seed"])
    eye = Vision(rig, a.control_dt, raw=True)
    if optical:
        eye.eye = LearnedOptics(eye.eye, optical)
    if model is not None:
        model.begin()
    before = model.priors.digest() if model is not None else None
    reference = np.asarray(ready).copy()
    home = reference.copy()
    sensed_inbound = False
    commands, memory = [], deque(maxlen=150)
    previous_reward = 0
    started = process_time()
    for tick in range(a.max_steps):
        q = a.joints
        x = observed(eye, tick, renderer.capture() if tick % 2 == 0 else None, q)
        if x is not None and x[4] < -.9 and x[1] < -.90:
            sensed_inbound = True
        desired = home
        if x is not None and (x[4] < -.9 or (sensed_inbound and x[1] < .50)):
            target, confidence = raw.get(x, q)
            if target is not None:
                desired = home + confidence * (target - home)
        if model is not None:
            desired = model.step(tick, eye.eye, x, q, reference, desired)
        u = np.clip((np.clip(desired, lower, upper) - reference) / stride, -.8, .8)
        sent = np.clip(reference + u * stride, lower, upper)
        _, _, done, capped, info = a.exchange(u, FAR_COMMAND)
        commands.append(u.tolist())
        if mode == "live":
            memory.append((tick, x, q, reference, sent))
            if any(e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]):
                score = audit(a.events, opening_side=1)["learner"]
                if score > previous_reward:
                    learned = model.priors.keep(list(memory), ("live", model.serial, score))
                    model.diag["credited_fragments"] += learned
                    model.diag["updates"] += bool(learned)
                    previous_reward = score
                    if progress and score % progress == 0:
                        print("in_progress", card["id"], score,
                              a.legal_returns + a.rival_returns - 1, tick, flush=True)
        reference = sent
        if done or capped or a.legal_returns + a.rival_returns - 1 >= cap:
            break
    return dict(shot=shot, opponent_seed=card["seed"], actions=commands, events=a.events,
                score=audit(a.events, opening_side=1), placements=a.point_placements(),
                motor_audit=a.motor_audit(), runtime_cpu_s=process_time() - started,
                field_ticks=int(model.diag["screened_ticks"]) if model else 0,
                planner=dict(model.diag) if model else {}, searches=model.notes if model else [],
                memory=dict(before=before, after=model.priors.digest()) if model is not None else None)


def run(feeds, bank, plan, deck, mode, output, *, optic=None, cap=200, limit=None, budget=28,
        variant="fragment", protocol_path=None, progress=0):
    receipt = _read(bank.with_suffix(".json"))
    if receipt["bank_sha256"] != _hash(bank):
        raise ValueError("Joint instruction archive changed")
    if deck == "development":
        base = _read(plan)["practice"]
        cards = [dict(base[i], reply_seed=3831100 + 13 * j) for j, i in enumerate(DEVELOPMENT_INDICES)]
    elif deck == "guarded_challenge":
        cards = [_read(plan)["held_openings"][i] for i in GUARDED_INDICES]
    else:
        cards = cards_for(receipt, plan, deck)
    if limit is not None:
        cards = cards[:limit]
    if protocol_path is not None:
        expected = _read(protocol_path)
        expected_cards = expected["development"] if deck == "development" else expected["held"][deck]
        if (expected_cards != cards or expected["plan_sha256"] != _hash(plan)
                or expected["feeds_sha256"] != _hash(feeds) or expected["prior_sha256"] != _hash(bank)
                or expected["optics_sha256"] != (_hash(optic) if optic else None)):
            raise ValueError("Held card roster, preprocessing or demonstrations changed from the frozen protocol")
    if deck == "held_openings" and {c["id"] for c in cards} & {r["card"]["id"] for r in receipt["practice"]}:
        raise ValueError("Held openings overlap with the physical instruction archive")
    ready = receipt["ready"]
    with np.load(bank) as b:
        raw = DeepField(b, "local")
    shots = measured_launches(feeds)
    results = dict(mode=mode, deck=deck, ready=ready, cards=cards, cap=cap, optical_sha256=_hash(optic) if optic else None,
                   source_sha256=_hash(bank), plan_sha256=_hash(plan), variant=variant, budget=budget, rows=[])
    results["protocol_sha256"] = _hash(protocol_path) if protocol_path else None
    results["model_source_sha256"] = {
        name: _hash(Path(__file__).with_name(name)) for name in (
            "tt38_joint_mpc.py", "tt37_optical_clinic.py", "tt29_rally_rollout.cpp")
    }
    results["online_boundary"] = (
        "empty per independent run; legal near-return observations can carry to later points on this deck only"
        if mode == "live" else "no observed action labels added")
    output.parent.mkdir(parents=True, exist_ok=True)
    with centered(ready):
        a = arena()
        model = None
        renderer, rig, stride, lower, upper = resources(a)
        try:
            model = MPC(bank, raw, variant=variant, budget=budget) if mode != "raw" else None
            for c in cards:
                r = play(a, renderer, rig, stride, lower, upper, ready, c, shots[c["id"]],
                         raw, model, optic, mode, cap, progress)
                results["rows"].append(r)
                _write(output, results)
                print(mode, deck, c["id"], r["score"]["learner"], r["score"]["returns"],
                      r["planner"], flush=True)
        finally:
            renderer.close()
            if model is not None:
                model.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("protocol", "guarded-protocol", "raw", "frozen", "live"))
    p.add_argument("deck", choices=("development", "held_openings", "held_replies", "fresh_replies", "guarded_challenge"))
    p.add_argument("output", type=Path)
    p.add_argument("--bank", type=Path, default=BANK)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--optic", type=Path)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--cap", type=int, default=200)
    p.add_argument("--limit", type=int)
    p.add_argument("--budget", type=int, default=28)
    p.add_argument("--variant", choices=("fragment", "elastic", "guarded"), default="fragment")
    p.add_argument("--progress-every", type=int, default=0)
    cli = p.parse_args()
    if cli.mode == "protocol":
        protocol(cli.bank, cli.plan, cli.feeds, cli.optic, cli.output)
    elif cli.mode == "guarded-protocol":
        guarded_protocol(cli.bank, cli.plan, cli.feeds, cli.optic, cli.output)
    else:
        run(cli.feeds, cli.bank, cli.plan, cli.deck, cli.mode, cli.output, optic=cli.optic,
            cap=cli.cap, limit=cli.limit, budget=cli.budget, variant=cli.variant, protocol_path=cli.protocol,
            progress=cli.progress_every)
