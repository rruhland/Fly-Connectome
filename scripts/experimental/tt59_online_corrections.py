"""One public danger-triggered repair per received offer, credited only on real returns."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt59_carry_fewshot as carry
from . import tt59_fewshot_joint_games as game
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit

base, school, mpc, course, season = game.repair, game.school, game.mpc, game.course, game.season
PLAN = carry.ROOT / "online.protocol.json"
OUT = carry.ROOT / "online.json.gz"
MODES = ("stock", "frozen23", "adaptive23")
N = 8


class Memory(base.Corrections):
    def __init__(self, model):
        super().__init__(model)
        self.original_size = len(self.ids)

    def credit(self, x, feature, fail, corrected):
        # This is a scored movement minus a public predicted baseline, not an
        # observed counterfactual failure. Do not feed native trial state here.
        eta = base.transfer.eta(x) / base.CANONICAL
        f = school.warp(fail, eta, 0.)
        d = (school.warp(corrected, eta, 0.) - f).reshape(-1)
        self.ids = np.r_[self.ids, -(len(self.ids)-self.original_size+1)]
        self.features = np.vstack((self.features, feature * base.transfer.SCALE))
        self.failures = np.concatenate((self.failures, f[None]), axis=0)
        self.weights = np.vstack((self.weights, d @ self.basis.T))


def low_envelope(x):
    t = max(0., (x[1]+1.10) / max(1.6, -x[4]))
    z, v = x[2], x[5]
    root = (v + np.sqrt(max(0., v*v + 19.62*(z-.780)))) / 9.81
    if 0 < root < t:
        v = .82 * (9.81*root-v)
        t, z = t-root, .780
    return z + t*v - 4.905*t*t


class Online(game.Player):
    def begin(self):
        super().begin()
        self.credit_context = None
        self.learned_here = 0

    def search(self, tick, x, q, ref, base_goal):
        # Keep the exact stock search and avoid shifting its RNG stream.
        season.Player.search(self, tick, x, q, ref, base_goal)
        feature = school.context(x, q, ref, self.engine.rate, self.engine.bounced)
        if self.examined or not base.visible(x, feature) or not self.notes or self.notes[-1]["tick"] != tick:
            return
        self.examined = True
        height = low_envelope(x)
        if self.notes[-1].get("robust", 0) or not .765 <= height <= 1.07 or not self.engine.sync():
            return
        anchors = base.anchor_paths(self.engine, self.raw, self.priors.home, x, q, ref, base_goal)
        paths, details = self.readout.proposals(x, feature, anchors, "paired")
        options, diag = base.rank_paths(self.engine, paths, ref, details)
        # Real paired successes include model false negatives. Allow a
        # narrow speculative lane on predicted-danger shots, once per offer.
        # Never explore if the best modeled contact is too weak or near-no-hit.
        options = [(u, d) for u, d in options if d["legal"] or d["robust"]
                   or d["separated"] and d["nominal"] < 8.40]
        stats = dict(tick=tick, public_height=float(height), issued=bool(options), **diag)
        if options:
            # Fixed contextual rank 1 is the same decision as the first
            # active retry arm. A rare adjacent candidate in uncertain cases
            # comes from the same seeded public stream in both conditions.
            order = sorted(options, key=lambda r: r[1]["score"])
            index = (1 if len(order) > 1 and not order[0][1]["robust"] and self.explore.random() < .23 else 0)
            rates, pick = order[index]
            self.pending = school.atlas.reference_path(ref, rates, *self.limits)[1:]
            self.pending_tick = tick
            self.best, self.tick, self.lease = self.pending, tick, base.H
            self.credit_context = (x.copy(), feature.copy(), anchors[pick["anchor"]], ref.copy(), [])
            stats["selected"] = pick
        self.repairs.append(stats)

    def sensory_credit(self, tick, sent_reference, legal_near):
        """Only commands actually sent and a one-bit legal near reward arrive here."""
        if self.credit_context is None:
            return
        x, feature, fail, ref, sent = self.credit_context
        elapsed = tick-self.pending_tick
        if elapsed > 110:
            self.credit_context = None
            return
        if elapsed < base.H:
            sent.append(sent_reference.copy()-ref)
        if legal_near:
            if self.mode == "adaptive23" and len(sent) >= 6:
                self.readout.credit(x, feature, fail, mpc.tail(np.array(sent), base.H))
                self.learned_here += 1
            self.credit_context = None


def play(a, cam, rig, limits, ready, card, shot, raw, actor, seed):
    stride, lower, upper = limits
    a.launch(shot, card=card)
    a.partner.reset(seed=card["seed"])
    eye = mpc.Vision(rig, a.control_dt, raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye, course.OPTIC)
    actor.begin()
    if isinstance(actor, Online):
        actor.explore = np.random.default_rng(seed)
    reference = ready.copy()
    once = False
    moves = []
    for tick in range(a.max_steps):
        q = a.joints
        x = mpc.observed(eye, tick, cam.capture() if tick % 2 == 0 else None, q)
        if x is not None and x[4] < -.9 and x[1] < -.90:
            once = True
        desired = ready
        if x is not None and (x[4] < -.9 or once and x[1] < .50):
            target, confidence = raw.get(x, q)
            if target is not None:
                desired = ready + confidence * (target - ready)
        desired = actor.step(tick, eye.eye, x, q, reference, desired)
        u = np.clip((np.clip(desired, lower, upper) - reference) / stride, -.8, .8)
        reference = np.clip(reference + u * stride, lower, upper)
        _, _, done, capped, info = a.exchange(u, mpc.FAR_COMMAND)
        moves.append(u.tolist())
        if isinstance(actor, Online):
            near = any(e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"])
            actor.sensory_credit(tick, reference, near)
        if done or capped:
            break
    placements = a.point_placements()
    for leg, req in enumerate(placements["requested"]):
        if req["target"] != list(game.draw(card, leg, "random", None)[0]):
            raise ValueError("Opponent's independently drawn goal changed")
    return dict(actions=moves, shot=shot, score=audit(a.events, opening_side=1), events=a.events,
                placements=placements, repairs=getattr(actor, "repairs", []),
                complete_point=a.events[-1]["kind"] == "point", motor_lessons=getattr(actor, "learned_here", 0))


def new_memory():
    with patch.object(base, "PLAN", carry.PLAN):
        return Memory(carry.MODEL)


def freeze(feeds):
    p, old, broad = _read(base.PLAN), _read(carry.PLAN), _read(game.PLAN)
    banned = set(p["excluded"]) | {x["id"] for z in (p, old) for x in z["cases"]}
    banned |= {r["card"]["id"] for r in broad["cases"]}
    stock = _read(season.CARRY)
    banned |= {c["id"] for c in stock["cards"][:len(stock["rows"])]}
    for path in (mpc.BANK, course.PRACTICE):
        banned |= {r["card"]["id"] for r in _read(path.with_suffix(".json"))["practice"]}
    standard = _read(course.SOURCE)
    decks = (*standard["practice"], *standard["assessment"], *_read(season.fresh.PLAN)["assessment"])
    fresh = list({x["id"]: x for x in decks
                  if x["id"] not in banned and x["opening"] == "random" and x["profile"] == "reach"}.values())
    playable = set()
    for path in (base.SOURCES[0], base.SOURCES[3], carry.SOURCES[0], carry.SOURCES[2]):
        past = _read(path)
        playable.update(c["id"] for c,r in zip(past["cards"], past["rows"]) if r["score"]["learner"] >= 1)
    if len(fresh) < N:
        raise ValueError("Insufficient new release IDs")
    rng = np.random.default_rng(590767)
    observed = mpc.measured_launches(feeds)
    rows = []
    order = sorted(rng.permutation(len(fresh)), key=lambda j: fresh[j]["id"] not in playable)
    for j in order[:N]:
        c = dict(fresh[j], seed=int(rng.integers(1, 2**30)), reply_seed=int(rng.integers(1, 2**30)))
        rows.append(dict(card=c, shot=observed[c["id"]], exploratory_seed=int(rng.integers(1, 2**30))))
    fixed = (carry.PLAN, carry.MODEL, game.PLAN, course.OPTIC, mpc.BANK, course.PRACTICE, season.CARRY,
             season.CARRY.with_suffix("").with_suffix(".npz"), season.fresh.RISK, course.SOURCE,
             season.fresh.PLAN, base.SOURCES[0], base.SOURCES[3], carry.SOURCES[0], carry.SOURCES[2])
    _write(PLAN, dict(cases=rows, modes=MODES, files={str(x): _hash(x) for x in fixed}, feeds=_hash(feeds),
                      ready=_read(mpc.BANK.with_suffix(".json"))["ready"],
                      role="Eight previously unused correction-release IDs, 24 preassigned ordinary one-launch continuous points. Prioritize measured launches that an ordinary unrelated archive previously returned and use new opponent draws. Frozen and adaptive repair start identically. Adaptive carries only legal-return feedback from its own games across this stream. Fixed broad opponent law. A public low-flight danger gate allows modest exploration when no robust stock choice exists. No extra practice games, retries, or intervened opponent delivery. The two physical fork courses remain separate."))
    print("online roster", [v["card"]["id"] for v in rows], flush=True)


def run():
    p = _read(PLAN)
    if any(_hash(Path(k)) != h for k,h in p["files"].items()):
        raise ValueError("Changed online curriculum")
    if OUT.exists():
        raise ValueError("Online point stream starts once from its frozen shared origin")
    with np.load(mpc.BANK) as f:
        raw = mpc.DeepField(f, "local")
    ready = np.asarray(p["ready"])
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    cp = _read(season.CARRY)
    result = dict(plan=_hash(PLAN), operator=_hash(Path(__file__)), rows=[], complete=False)
    ids = {r["card"]["id"] for r in p["cases"]}
    with mpc.centered(ready), school.enabled():
        a = mpc.arena()
        cam, rig, stride, lo, hi = mpc.resources(a)
        limits = (stride, lo, hi)
        actors = {}
        try:
            for tag in MODES:
                kw = dict(coverage=rank, variant="elastic", budget=28)
                with patch.object(mpc, "Fragments", lambda f,e: game.Carried(f,e,practice=[course.PRACTICE],held=ids)):
                    actors[tag] = (season.Player(mpc.BANK, raw, **kw) if tag == "stock" else
                                   Online(mpc.BANK, raw, readout=new_memory(), mode=tag, limits=limits, **kw))
                course.resume(actors[tag], season.CARRY.with_suffix("").with_suffix(".npz"), cp)
            for r in p["cases"]:
                card = r["card"]
                results = {}
                for tag, actor in actors.items():
                    row = play(a, cam, rig, limits, ready, card, r["shot"], raw, actor, r["exploratory_seed"])
                    if tag != "stock":
                        row["same_offer_changes"] = game.compare(results["stock"], row)
                    results[tag] = row
                    print(card["id"], tag, row["score"]["learner"], len(row["actions"]),
                          sum(x["issued"] for x in row["repairs"]), row["motor_lessons"], flush=True)
                result["rows"].append(dict(r, arms=results))
                result["complete"] = len(result["rows"]) == N
                _write(OUT, result)
        finally:
            for actor in actors.values():
                actor.close()
            cam.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "run"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    freeze(a.feeds) if a.mode == "plan" else run()
