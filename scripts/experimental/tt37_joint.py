"""Use a wide framewise joint-position field alongside present-time visual reach."""

import argparse
from collections import Counter, deque
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, measured_launches
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, activation, features, resources
from .tt36_reach import BODY_BANK, Body, Reach, Relation
from .tt37_joint_capture import ARTIFACTS, OUTPUT, PUPIL, observed
from .tt37_opponent import DIAGNOSTIC_CAP, arena


WIDTHS = np.array([.18, .15, .12, .95, 1.35, 1.8, 1.6, 1.6, 1.6, 1.3, 1.6, 6., 6., 6., 6., 6.])
BODY_WIDTHS = np.r_[WIDTHS, [.23, .29, .18, .72, .70, .58, .90, .90, .90]]
REACH_DIR = ARTIFACTS / "tt36"


def smooth(value):
    u = float(np.clip(value, 0., 1.))
    return u * u * (3. - 2. * u)


def motor_features(state, blade):
    q, dq = state[6:11], state[11:16]
    center, normal, velocity, _ = blade.pose(q, jacobian=True)
    return np.r_[state, state[:3] - center, normal, velocity @ dq]


class Field:
    """Map the current observation to one bounded joint position vector."""

    def __init__(self, bank, mode, blade):
        states = bank["states"]
        self.blade = blade
        self.mode = mode
        self.body = mode in ("body", "recovery", "gated")
        scale = BODY_WIDTHS if self.body else WIDTHS
        source = np.array([motor_features(s, blade) for s in states]) if self.body else states
        self.scale = scale
        density = Counter(bank["releases"].tolist())
        balances = np.array([1. / np.sqrt(density[r]) for r in bank["releases"]])
        self.banks = {}
        self.prior = {}
        self.memory_states = np.empty((0, 16))
        self.memory_targets = np.empty((0, 5))
        for direction in (-1, 1):
            use = np.sign(states[:, 4]) == direction
            x = source[use] / scale
            q = bank["targets_lead"][use, :5]
            if len(x) < 32:
                raise ValueError("The direct field needs samples in both flight directions")
            self.prior[direction] = x, q, balances[use]
            self.banks[direction] = cKDTree(x), *self.prior[direction]

    def retain(self, states, targets):
        if not len(states):
            return
        self.memory_states = np.concatenate((self.memory_states, states))[-1152:]
        self.memory_targets = np.concatenate((self.memory_targets, targets))[-1152:]
        samples = self.memory_states
        source = np.array([motor_features(s, self.blade) for s in samples]) if self.body else samples
        for direction in (-1, 1):
            use = np.sign(samples[:, 4]) == direction
            x, q, weight = self.prior[direction]
            points = np.concatenate((x, source[use] / self.scale), axis=0)
            goals = np.concatenate((q, self.memory_targets[use]), axis=0)
            weights = np.r_[weight, np.full(use.sum(), .09)]
            self.banks[direction] = cKDTree(points), points, goals, weights

    def get(self, state):
        direction = -1 if state[4] < 0. else 1
        tree, source, target, balance = self.banks[direction]
        value = (motor_features(state, self.blade) if self.body else state) / self.scale
        dist, ix = tree.query(value, 28)
        confidence = smooth((1.85 - dist[6]) / 1.12)
        if not confidence:
            return None, 0., float(dist[6])
        local = source[ix] - value
        weight = balance[ix] * np.exp(-.5 * (dist / max(.22, dist[9])) ** 2)
        weight /= weight.sum()
        center = weight @ local
        mean = weight @ target[ix]
        deviations = local - center
        coef = np.linalg.solve(deviations.T @ (weight[:, None] * deviations) + .07 * np.eye(len(value)),
                               deviations.T @ (weight[:, None] * (target[ix] - mean)))
        return mean - np.clip(center @ coef, -.23, .23), confidence, float(dist[6])


class Direct:
    """Use visual field support to blend its immediate target with the frozen reach head."""

    def __init__(self, raw, stride, lower, upper, rig, body, relation, authority, field=None):
        self.raw = JointPolicy(raw, stride, lower, upper)
        self.reach = Reach(body, relation, self.raw, rig, authority, 1)
        self.field = field
        self.inbound_seen = False
        self.field_ticks = 0
        self.recovery_ticks = 0
        self.clipped_model_ticks = 0

    def command(self, tick, optical, q, dq, frames, *, correction=None):
        inbound = optical if optical is not None and optical[4] < -.9 else None
        if inbound is not None and inbound[1] < -.88:
            self.inbound_seen = True
        offset = self.reach.bias(tick, inbound, q, frames)
        if self.field is None and correction is None:
            return self.raw.command(inbound, q, offset=offset)

        goal = self.raw.home.copy()
        limit = self.raw.options["recovery"]
        if inbound is not None:
            x = features(inbound, self.raw.options["features"], q, self.raw.home)
            weight = activation(x, self.raw.centers, self.raw.options["width"])
            goal += weight @ self.raw.weights
            if offset is not None:
                goal[:5] += offset
            limit = self.raw.options["limit"]
        recovery = optical is not None and inbound is None and self.inbound_seen and optical[1] < .50
        wanted = self.field is not None and (recovery or (inbound is not None and self.field.mode != "recovery"))
        if wanted:
            target, amount, _ = self.field.get(np.r_[optical, q[:5], dq[:5]])
            if inbound is not None and self.field.mode == "gated":
                distance, _ = self.reach.support.query(x)
                amount *= smooth((distance - .90) / .80)
            if target is not None and (self.field.mode in ("plain", "body") or amount > .02):
                # Limit its difference from the tested fallback while retaining a large movement range.
                change = target - goal[:5]
                goal[:5] += amount * np.clip(change, -.62, .62)
                self.field_ticks += 1
                self.recovery_ticks += inbound is None
                self.clipped_model_ticks += int(np.max(abs(change)) > .62)
        if correction is not None and inbound is not None:
            goal[:5] += correction
        return self.raw.drive(goal, q, limit)



class RewardedFrames:
    """Reinforce the recent measured movement after a real legal return."""

    def __init__(self):
        self.frames = deque(maxlen=108)
        self.last_rewarded_inbound = -1
        self.updates = []

    def record(self, tick, optical, q, dq, target):
        state = None if optical is None else np.r_[optical, q[:5], dq[:5]]
        self.frames.append((tick, state, target[:5].copy()))

    def keep(self, tick, field):
        seen = [i for i, state, _ in self.frames
                if state is not None and state[4] < -.9 and state[1] < -.85]
        if not seen:
            return
        inbound = seen[-1]
        if inbound <= self.last_rewarded_inbound or tick - inbound > 95:
            return
        saved = list(self.frames)
        recent, targets = [], []
        for i, (step, state, _) in enumerate(saved):
            if state is None or step < inbound - 34 or step % 2:
                continue
            if state[4] < -.9 and step <= inbound or state[4] > .9 and step > inbound and state[1] < .50:
                recent.append(state)
                targets.append(saved[min(i + 2, len(saved) - 1)][2])
        if len(recent) < 5:
            return
        field.retain(np.asarray(recent), np.asarray(targets))
        self.last_rewarded_inbound = inbound
        self.updates.append(dict(reward_tick=tick, last_optical_inbound_tick=inbound, frames=len(recent)))

def play(environment, renderer, rig, model, stride, lower, upper, entry, shot, *,
         body, relation, authority, field, cap, learn=False):
    environment.launch(shot, card=entry)
    environment.partner.reset(seed=entry["seed"])
    eye = Vision(rig, environment.control_dt, raw=True)
    pupil = Direct(model, stride, lower, upper, rig, body, relation, authority, field)
    previous = environment.joints
    actions = []
    experience = RewardedFrames() if learn else None
    started = process_time()
    for tick in range(environment.max_steps):
        q = environment.joints
        dq = (q - previous) / environment.control_dt
        frames = renderer.capture() if tick % 2 == 0 else None
        optical = observed(eye, tick, frames, q)
        command = pupil.command(tick, optical, q, dq, frames)
        _, _, done, capped, referee = environment.exchange(command, FAR_COMMAND)
        actions.append(command.tolist())
        if experience is not None:
            experience.record(tick, optical, q, dq, pupil.raw.reference)
            if any(event["kind"] == "legal_return" and event["side"] == 1 for event in referee["events"]):
                experience.keep(tick, field)
        previous = q
        if done or capped or environment.legal_returns + environment.rival_returns - 1 >= cap:
            break
    return dict(
        shot=shot, opponent_seed=entry["seed"], actions=actions, events=environment.events,
        score=audit(environment.events, opening_side=1), placements=environment.point_placements(),
        field_ticks=pupil.field_ticks, recovery_ticks=pupil.recovery_ticks,
        reach_ticks=len(pupil.reach.active), clipped_model_ticks=pupil.clipped_model_ticks,
        motor_audit=environment.motor_audit(), runtime_cpu_s=process_time() - started,
        updates=experience.updates if experience is not None else [],
    )


def evaluate(feeds, plan, deck, bank_path, output, mode, *, cap=DIAGNOSTIC_CAP, limit=None,
             learn=False, carry=None, save=None):
    if mode == "reach" and (learn or carry is not None or save is not None):
        raise ValueError("The unchanged reach control does not have a mutable frame memory")
    raw = _read(PUPIL)
    poses = _read(bank_path.with_suffix(".json"))
    if poses["sha256"] != _hash(bank_path) or poses["pupil_sha256"] != _hash(PUPIL):
        raise ValueError("Observation source and raw pupil do not match")
    roster = _read(plan)
    cards = roster if isinstance(roster, list) else roster[deck]
    if isinstance(cards, dict):
        cards = cards["cards"]
    if limit is not None:
        cards = cards[:limit]
    if deck == "held_openings" and {c["id"] for c in cards}.intersection(poses["releases"]):
        raise ValueError("New-opening assessment overlaps demonstration releases")
    if deck.startswith("held") and {c["reply_seed"] for c in cards}.intersection(
            item["reply"] for item in poses["records"] if item["reply"] is not None):
        raise ValueError("Held continuation overlaps a demonstration opponent seed")
    shots = measured_launches(feeds)
    with np.load(BODY_BANK) as archive:
        body = Body(archive)
    with np.load(REACH_DIR / "reach-observations.npz") as archive:
        relation = Relation(archive)
    authority = _read(REACH_DIR / "reach-checkpoint.json")["authority_rad"]
    report = dict(
        ready=raw["ready"], opening_side=1, cap=cap, mode=mode, deck=deck, cards=cards, rows=[],
        plan_sha256=_hash(plan), bank_sha256=_hash(bank_path),
        pupil_sha256=_hash(PUPIL), feed_sha256=_hash(feeds), protocol="random",
        learn=learn, carried_sha256=_hash(carry) if carry else None,
    )
    with centered(raw["ready"]):
        environment = arena(protocol="random")
        try:
            renderer, rig, stride, lower, upper = resources(environment)
            try:
                with np.load(bank_path) as archive:
                    field = Field(archive, mode, BladeIK(environment)) if mode != "reach" else None
                if carry is not None:
                    receipt = _read(carry.with_suffix(".json"))
                    if receipt["features"] != mode or receipt["bank_sha256"] != _hash(bank_path):
                        raise ValueError("Live reinforcement requires the same frozen field")
                    if deck == "held_openings" and {c["id"] for c in cards}.intersection(receipt["seen_releases"]):
                        raise ValueError("New-opening assessment overlaps live practice")
                    if {c["reply_seed"] for c in cards}.intersection(receipt["seen_reply_seeds"]):
                        raise ValueError("Held continuation sees a practiced opponent seed")
                    with np.load(carry) as updates:
                        field.retain(updates["states"], updates["targets"])
                for entry in cards:
                    row = play(environment, renderer, rig, raw, stride, lower, upper, entry, shots[entry["id"]],
                               body=body, relation=relation, authority=authority, field=field, cap=cap, learn=learn)
                    report["rows"].append(row)
                    _write(output, report)
                    print(mode, deck, entry["id"], row["score"]["learner"], row["score"]["returns"],
                          row["field_ticks"], row["recovery_ticks"], len(row["updates"]), flush=True)
                if save is not None:
                    np.savez_compressed(save, states=field.memory_states, targets=field.memory_targets)
                    _write(save.with_suffix(".json"), dict(
                        bank_sha256=_hash(bank_path), source_sha256=_hash(output), features=mode,
                        seen_releases=sorted({c["id"] for c in cards}),
                        seen_reply_seeds=sorted({c["reply_seed"] for c in cards}),
                        frames=len(field.memory_states), updates=sum(len(r["updates"]) for r in report["rows"]),
                    ))
            finally:
                renderer.close()
        finally:
            environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("deck")
    parser.add_argument("mode", choices=("reach", "plain", "body", "recovery", "gated"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--bank", type=Path, default=OUTPUT / "joint-observations.npz")
    parser.add_argument("--cap", type=int, default=DIAGNOSTIC_CAP)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--carry", type=Path)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    evaluate(args.feeds, args.plan, args.deck, args.bank, args.output, args.mode, cap=args.cap, limit=args.limit,
             learn=args.learn, carry=args.carry, save=args.save)


if __name__ == "__main__":
    main()
