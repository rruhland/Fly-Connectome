"""Learn how far to preview bounded direct joint feedback from real returns."""
import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt37_opponent import draw
from .tt39_carry_joints import Carried

OUT = Path("artifacts/tt44_preview")
FEEDS = Path("/tmp/tt25-rallies.json")
PLAN = OUT / "course.json"
CARRY = course.ROOT / "learning-24.json.gz"
DELAYS = (0, 2, 4, 7)
EXPLORE = .55
OFFERS_SEED = 4421903
FEATURES = 6


def eligible(x):
    return x is not None and -.53 < x[1] < .98 and x[4] < -1.15 and .79 < x[2] < 1.69


def preview(x, ticks):
    """A short displacement of the sensed p/v state, not a requested hit."""
    h = min(ticks * mpc.DT, (x[1] + .61) / max(-x[4], 1.15))
    z = x[2] + h * x[5] - 4.905 * h**2
    if z < .785:
        h *= .42
    y = x.copy()
    y[:3] += h * x[3:6]
    y[2] -= 4.905 * h**2
    y[5] -= 9.81 * h
    return y


def features(track, q, ready):
    x = np.mean(track, axis=0)
    return np.r_[x[0] / .32, x[3] / .85, (x[4] + 4.6) / 1.35, x[5] / 1.65,
                 x[1] / .85, np.linalg.norm((q - ready)[:5]) / 1.7]


class Bandit:
    """Closed-form kernel-smoothed Beta reward readout on actual executed delays."""

    def __init__(self):
        self.context = []
        self.action = []
        self.score = []

    def update(self, context, action, earned):
        if action not in range(len(DELAYS)) or earned not in (0, 1):
            raise ValueError("Invalid observed reward or delay")
        self.context.append(np.array(context, dtype=float))
        self.action.append(int(action))
        self.score.append(int(earned))

    def estimates(self, context):
        rates = np.full(len(DELAYS), .50)
        support = np.zeros(len(DELAYS))
        if not self.context:
            return rates, support
        distance = np.sum((np.asarray(self.context) - context)**2, axis=1)
        label, action = np.asarray(self.score), np.asarray(self.action)
        local = np.exp(-.5 * distance)
        for i in range(len(DELAYS)):
            use = action == i
            w = local[use]
            support[i] = w.sum()
            # Small, shared uninformative prior. The weaker across-context
            # term keeps a single unusually close trial from steering.
            w = .17 + .83 * w
            rates[i] = (1.05 + w @ label[use]) / (2.1 + w.sum())
        return rates, support

    def greedy(self, context):
        rates, support = self.estimates(context)
        order = np.argsort(-rates, kind="stable")
        for i in order:
            if i == 0:
                break
            if support[i] >= 1.15 and rates[i] - rates[0] >= .045 and support[0] >= .9:
                return int(i)
        return 0

    def save(self, path):
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, context=np.asarray(self.context).reshape(-1, FEATURES),
                               action=np.asarray(self.action, dtype=np.int8),
                               score=np.asarray(self.score, dtype=np.int8), delays=DELAYS)
        tmp.replace(path)
        return _hash(path)

    @classmethod
    def load(cls, path):
        out = cls()
        with np.load(path) as f:
            if tuple(f["delays"]) != DELAYS:
                raise ValueError("Checkpoint action menu changed")
            for c, u, y in zip(f["context"], f["action"], f["score"], strict=True):
                out.update(c, u, y)
        return out


def exploration(serial, approach):
    """The matched calendar needs neither release ID nor the opponent's seed."""
    rng = np.random.default_rng([OFFERS_SEED, serial, approach])
    return bool(rng.random() < EXPLORE), int(rng.integers(len(DELAYS)))


class Passage:
    def __init__(self, policy, memory, *, serial):
        self.policy = policy
        self.memory = memory
        self.serial = serial
        self.current = None
        self.track = []
        self.trace = []
        self.opportunity = False
        self.counter = 0
        self.last_tick = None

    def get(self, tick, x, q, home, branch, desired):
        if not eligible(x):
            self.track.clear()
            return desired
        if self.current is None:
            if (self.last_tick is None or self.last_tick != tick - 1
                    or self.track and not -.22 < x[1] - self.track[-1][1] < -.009):
                self.track.clear()
            self.last_tick = tick
            self.track.append(x.copy())
            if len(self.track) < 3:
                return desired
            z = features(self.track[-3:], q, home)
            explore, forced = exploration(self.serial, self.counter)
            is_training = self.policy in ("adapt", "no-update")
            explore = bool(explore and is_training)
            if explore:
                action = forced
            elif self.policy in ("adapt", "learned"):
                action = self.memory.greedy(z)
            else:
                action = 0
            self.current = {
                "depth": float(x[1]), "tick": tick, "context": z.tolist(),
                "arm": action, "delay_ticks": DELAYS[action], "forced_exploration": explore,
                "offer": self.counter, "effect_ticks": 0, "max_changed_rad": 0.,
                "data_at_choice": len(self.memory.score),
            }
            self.counter += 1
        arm = self.current["arm"]
        if arm == 0:
            return desired
        look = preview(x, DELAYS[arm])
        future, certainty = branch.get(look, q)
        if future is None:
            return desired
        goal = home + certainty * (future - home)
        adjust = np.clip(goal - desired, [-.42, -.32, -.33, -.36, -.43, 0.],
                         [.42, .32, .33, .36, .43, 0.])
        max_adjust = float(np.max(abs(adjust)))
        if max_adjust > .00001:
            self.current["effect_ticks"] += 1
            self.current["max_changed_rad"] = max(self.current["max_changed_rad"], max_adjust)
        # This modifies only the early reference. The ordinary joint rate
        # and hard joint limits remain authoritative for the issued command.
        return desired + adjust

    def heard(self, score, tick):
        if self.current is not None:
            self.current.update(earned=score, outcome_tick=tick,
                                legally_delivered=bool(self.opportunity))
            if score is not None and self.policy == "adapt":
                self.memory.update(self.current["context"], self.current["arm"], score)
            self.trace.append(self.current)
        self.current = None
        self.track.clear()
        self.last_tick = None
        self.opportunity = False


def physical(a, image, rig, stride, low, high, ready, card, shot, raw, cap, passage, late=None):
    a.launch(shot, card=card)
    a.partner.reset(seed=card["seed"])
    eye = mpc.Vision(rig, a.control_dt, raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye, course.OPTIC)
    if late:
        late.begin()
    ref = np.asarray(ready).copy()
    sensed_inbound = False
    n = 0
    controls = []
    start = process_time()
    for tick in range(a.max_steps):
        q = a.joints
        x = mpc.observed(eye, tick, image.capture() if tick % 2 == 0 else None, q)
        if x is not None and x[4] < -.9 and x[1] < -.90:
            sensed_inbound = True
        desired = ready
        if x is not None and (x[4] < -.9 or (sensed_inbound and x[1] < .50)):
            target, confidence = raw.get(x, q)
            if target is not None:
                desired = ready + confidence * (target - ready)
        desired = passage.get(tick, x, q, ready, raw, desired)
        if late:
            desired = late.step(tick, eye.eye, x, q, ref, desired)
        u = np.clip((np.clip(desired, low, high) - ref) / stride, -.8, .8)
        ref = np.clip(ref + u * stride, low, high)
        _, _, done, capped, events = a.exchange(u, mpc.FAR_COMMAND)
        controls.append(u.tolist())
        for event in events["events"]:
            if event["kind"] != "legal_return":
                continue
            if event["side"] == -1:
                passage.opportunity = True
            else:
                now = audit(a.events, opening_side=1)["learner"]
                if now > n:
                    passage.heard(1 if passage.opportunity else None, tick)
                    n = now
        if done or capped or a.legal_returns + a.rival_returns - 1 >= cap:
            passage.heard(0 if done and passage.opportunity else None, tick)
            break
    requests = a.point_placements()
    for leg, item in enumerate(requests["requested"]):
        if item["target"] != list(draw(card, leg, "random", None)[0]):
            raise ValueError("Physical opponent did not apply the pre-specified draw")
    return {
        "shot": shot, "opponent_seed": card["seed"], "actions": controls, "events": a.events,
        "score": audit(a.events, opening_side=1), "placements": requests,
        "motor_audit": a.motor_audit(), "runtime_cpu_s": process_time() - start,
        "episodes": passage.trace,
    }


def freeze(a):
    original = _read(course.PLAN)
    train, held = original["practice"][:a.n_train], original["assessment"][12:12+a.n_held]
    if not train or not held or any(c["profile"] != "reach" or c["opening"] != "random" for c in train + held):
        raise ValueError("Use the predeclared broad random course only")
    seen = {c["id"] for c in train}
    if len(seen) != len(train) or seen & {c["id"] for c in held}:
        raise ValueError("Practice and frozen releases overlap")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    _write(a.out, {
        "train": train, "held": held, "train_cap": a.train_cap, "held_cap": a.held_cap,
        "delays": DELAYS, "exploration_probability": EXPLORE, "exploration_seed": OFFERS_SEED,
        "source_sha256": _hash(Path(__file__)), "bank_sha256": _hash(mpc.BANK),
        "optics_sha256": _hash(course.OPTIC), "feeds_sha256": _hash(a.feeds),
        "plan_sha256": _hash(course.PLAN),
        "boundary": "Practice learns online after physical, legally delivered offers only. Common exploration uses ordinal and does not look at opponent state. Held releases were not in preview practice, and held choices are deterministic without updates.",
    })


def run(a):
    plan = _read(a.plan)
    if (plan["source_sha256"] != _hash(Path(__file__)) or plan["bank_sha256"] != _hash(mpc.BANK)
            or plan["optics_sha256"] != _hash(course.OPTIC) or plan["feeds_sha256"] != _hash(a.feeds)):
        raise ValueError("Frozen inputs differ")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    check = a.out.with_suffix("").with_suffix(".npz")
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    feeds = mpc.measured_launches(a.feeds)
    ready = np.asarray(_read(mpc.BANK.with_suffix(".json"))["ready"])
    hist = Bandit.load(a.checkpoint) if a.checkpoint else Bandit()
    transfer = a.kind == "transfer"
    if transfer and not a.checkpoint:
        raise ValueError("Late MPC transfer needs a preview checkpoint")
    phases = [("train", a.kind), ("held", "learned" if a.kind == "adapt" else "stock")] if not transfer else [
        ("held", a.policy)]
    doc = {
        "kind": a.kind, "phases": dict(phases), "plan_sha256": _hash(a.plan),
        "ready": ready.tolist(), "checkpoint_start": _hash(a.checkpoint) if a.checkpoint else None,
        "late_carry_sha256": _hash(CARRY) if transfer else None,
        "records": {"train": [], "held": []}, "checkpoint_sha256": None, "complete": False,
    }
    with mpc.centered(ready):
        ar = mpc.arena()
        image, rig, stride, lo, hi = mpc.resources(ar)
        late = None
        if transfer:
            # Same late policy and previously earned physical motion base for
            # both arms. No preview training outcomes inform late selection.
            with patch.object(mpc, "Fragments", lambda b, e: Carried(
                    b, e, practice=[course.PRACTICE], held={c["id"] for c in plan["held"]})):
                late = mpc.MPC(mpc.BANK, raw, variant="elastic")
            course.resume(late, CARRY.with_suffix("").with_suffix(".npz"), _read(CARRY))
        try:
            for phase, policy in phases:
                cards = plan[phase]
                if transfer:
                    cards = cards[:a.transfer_games]
                for i, card in enumerate(cards):
                    initial_launches = ar.launches
                    init = len(hist.score)
                    state = Passage(policy, hist, serial=i)
                    row = physical(ar, image, rig, stride, lo, hi, ready, card,
                                   feeds[card["id"]], raw, plan[phase + "_cap"], state, late)
                    if ar.launches - initial_launches != 1 or (phase == "held" and len(hist.score) != init):
                        raise ValueError("Incomplete launch or frozen memory changed")
                    row["card"] = card
                    row["learner_updates"] = len(hist.score) - init
                    doc["records"][phase].append(row)
                    if phase == "train" or doc["checkpoint_sha256"] is None:
                        doc["checkpoint_sha256"] = hist.save(check)
                    doc["complete"] = (phase == phases[-1][0] and i + 1 == len(cards))
                    _write(a.out, doc)
                    print(a.kind, phase, i + 1, card["id"], row["score"]["learner"],
                          len(state.trace), sum(e["effect_ticks"] for e in state.trace),
                          len(hist.score), round(row["runtime_cpu_s"], 2), flush=True)
        finally:
            if late:
                late.close()
            image.close()
            ar.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("plan", "adapt", "no-update", "transfer"))
    p.add_argument("out", type=Path)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--feeds", type=Path, default=FEEDS)
    p.add_argument("--n-train", type=int, default=18)
    p.add_argument("--n-held", type=int, default=12)
    p.add_argument("--train-cap", type=int, default=8)
    p.add_argument("--held-cap", type=int, default=16)
    p.add_argument("--transfer-games", type=int, default=4)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--policy", choices=("learned", "stock"), default="learned")
    args = p.parse_args()
    freeze(args) if args.kind == "plan" else run(args)
