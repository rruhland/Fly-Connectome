"""Learn mixtures of direct stroke and visual reach from completed physical exchanges."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt34_results import paired
from .tt35_error import Observations, passage
from .tt36_online import outcome
from .tt36_reach import Body, Reach, Relation, smooth
from .tt37_opponent import arena as make_arena

ROOT = Path(__file__).resolve().parents[2] / "artifacts"
OUT = ROOT / "tt37"
PUPIL = ROOT / "tt33" / "joint-raw.json.gz"
MODEL = ROOT / "tt34" / "lag-bank.npz"
RELATION = ROOT / "tt36" / "reach-observations.npz"
SEARCH = ROOT / "tt35" / "search-checkpoint.json"
# (continuous body reach, inherited five-joint residual, graded gap adjustment)
MIXES = np.array([[1., 0., 0.], [0., 0., 0.], [0., 1., 0.],
                  [0., -.75, 0.], [.60, .75, 0.], [.90, 0., 1.]])
SCALE = np.array([.22, .20, .24, .181, .73, 1.8, 3.0, 1.18, 1.3, 1.3, 1.3, 1.35])
PRIOR = np.array([.32, .28, .29, .26, .28, .26])
MAX_BANK = 144


class ExchangeMemory:
    """Keep local action consequences and a small sensor-derived inverse-reach hint."""

    def __init__(self, records=()):
        self.records = list(records)

    def keep(self, record):
        if record["reward"] is not None:
            self.records.append(record)
            self.records = self.records[-MAX_BANK:]

    def choose(self, context, rng, exploration):
        estimates = PRIOR.copy()
        gap = np.zeros(5)
        if self.records:
            entries = self.records
            score_bearing = [r for r in entries if r["reward"] is not None]
            delta = (np.array([r["context"] for r in score_bearing]) - context) / SCALE
            weights = np.exp(-.8 * np.sum(delta * delta, axis=1))
            arms = np.array([r["choice"] for r in score_bearing])
            labels = np.array([r["reward"] for r in score_bearing])
            counts = np.bincount(arms, weights=weights, minlength=len(PRIOR))
            total = np.bincount(arms, weights=labels * weights, minlength=len(PRIOR))
            estimates = (.80 * PRIOR + total) / (.80 + counts)
            estimates[1:][counts[1:] < .30] = -1.0
            w = np.array([k if r["miss_dq_rad"] is not None else 0. for k, r in zip(weights, score_bearing)])
            hints = np.array([r["miss_dq_rad"] if r["miss_dq_rad"] is not None else np.zeros(5)
                              for r in score_bearing])
            gap = w @ hints / max(.5, w.sum())
        chosen = int(np.argmax(estimates))
        explore = rng.random() < exploration
        selected = int(rng.integers(len(MIXES))) if explore else chosen
        return selected, gap, chosen, explore


def sensory_context(optical, q, body):
    position = body.local(q[:5])[0]
    # All twelve features describe the arm and ball at the current observed instant.
    return np.r_[optical[0], optical[:3] - position, optical[3:6], q[:5]]


class Stroke:
    def __init__(self, body, memory, offsets, rng, exploration, update):
        self.body, self.memory, self.offsets, self.rng = body, memory, offsets, rng
        self.exploration, self.update = exploration, update
        self.current, self.previous_y = None, None
        self.outcomes, self.outgoing = [], []
        self.joints = {}

    def start(self, tick, optical, q, frames):
        if self.current is not None or optical is None or not -.99 < optical[1] <= -.20:
            return
        x = sensory_context(optical, q, self.body)
        index, hint, exploit, explore = self.memory.choose(x, self.rng, self.exploration)
        self.current = dict(tick=tick, context=x.tolist(), choice=index, exploit=exploit, exploratory=explore,
                            hint_rad=hint.tolist(), prior_outcomes=len(self.memory.records))
        self.outgoing, self.joints = [], {}

    def action(self, tick, optical, q, frames, reach):
        if self.current is None or optical is None:
            return reach.bias(tick, optical, q, frames)
        self.joints[tick] = q[:5].copy()
        a, b, c = MIXES[self.current["choice"]]
        correction = reach.bias(tick, optical, q, frames) if a else None
        # Smooth each potential adjustment after the visual onset. Actual motor
        # speed/drive limits remain those of the unmodified direct pupil.
        onset = smooth((-.20 - optical[1]) / .37)
        late = smooth((-.69 - optical[1]) / .46)
        return ((a * correction if correction is not None else np.zeros(5))
                + onset * b * self.offsets + late * c * np.asarray(self.current["hint_rad"]))

    def finish(self, tick, stereo, legal):
        if self.current is None:
            return
        i = self.current["tick"]
        cross = passage([(j, p, q) for j, p, q in stereo if i <= j <= tick], i)
        gap = None if cross is None else float(np.linalg.norm(cross["gap_xz_m"]))
        speed = max(self.outgoing, default=None)
        grade = outcome(legal, speed, gap)
        hint = None
        if not legal and speed is None and cross is not None and self.joints and .035 < gap < .25:
            t = min(self.joints, key=lambda s: abs(s - cross["tick"]))
            _, jac, support = self.body.local(self.joints[t])
            if abs(t - cross["tick"]) < 8 and support < .85:
                j = jac[[0, 2]]
                desired = np.clip(cross["gap_xz_m"], -.12, .12)
                dq = j.T @ np.linalg.solve(j @ j.T + .011 * np.eye(2), desired)
                hint = np.clip(dq * .7, -.18, .18).tolist()
        record = dict(self.current, end_tick=tick, legal=legal,
                      reward=None if grade is None else float(grade), visual_gap_m=gap, visual_outgoing_mps=speed,
                      miss_dq_rad=hint)
        self.outcomes.append(record)
        if self.update:
            self.memory.keep(record)
        self.current = None

    def observe(self, tick, optical, q, frames, vision, view, dt):
        f = vision.eye.latest
        if f is not None and f.time == tick * dt and f.velocity[1] > .9 and f.rms < 2.4 and f.depth_sigma < .16:
            if self.current is not None and -1.58 < f.position[1] < -.47:
                self.outgoing.append(float(f.velocity[1]))
        if optical is not None:
            # A new inbound run observed near midcourt certifies the old run
            # has ended even if no rebound was visible.
            y = optical[1]
            if self.current is not None and self.previous_y is not None and self.previous_y < -.54 and y > .26:
                self.finish(tick, view.rows, False)
            self.previous_y = y
            self.start(tick, optical, q, frames)


def point(arena, view, rig, model, stride, lo, hi, body, relation, offsets, shot, card, bank, rng,
          exploration, update, cap):
    arena.launch(shot, card=card)
    arena.partner.reset(seed=card["seed"])
    view.begin()
    wing = joint.JointPolicy(model, stride, lo, hi)
    reach = Reach(body, relation, wing, rig, .38, 1.)
    eye = joint.Vision(rig, arena.control_dt, raw=True)
    stroke = Stroke(body, bank, offsets, rng, exploration, update)
    commands, reward_ticks = [], []
    start = process_time()
    for tick in range(arena.max_steps):
        q = arena.joints
        frames = view.capture() if tick % 2 == 0 else None
        optical = eye.update(tick, frames, q)
        stroke.observe(tick, optical, q, frames, eye, view, arena.control_dt)
        offset = stroke.action(tick, optical, q, frames, reach)
        command = wing.command(optical, q, offset=offset)
        _, _, done, capped, referee = arena.exchange(command, joint.FAR_COMMAND)
        commands.append(command.tolist())
        if any(e["kind"] == "legal_return" and e["side"] == 1 for e in referee["events"]):
            if stroke.current is not None:
                stroke.finish(tick, view.rows, True)
                reward_ticks.append(tick)
        if done or capped or arena.legal_returns + arena.rival_returns - 1 >= cap:
            break
    stroke.finish(tick, view.rows, False)
    return dict(shot=shot, opponent_seed=card["seed"], actions=commands, events=arena.events.copy(),
                score=audit(arena.events, opening_side=1), placements=arena.point_placements(),
                feedback=stroke.outcomes, reward_ticks=reward_ticks, runtime_cpu_s=process_time()-start)


def run(args):
    plan = _read(args.course)
    if plan["feed_sha256"] != _hash(args.feeds):
        raise ValueError("The supplied feeds differ from the locked opponent course")
    cards = plan[args.phase]
    if args.limit:
        cards = cards[:args.limit]
    model = _read(PUPIL)
    held = args.mode == "carry"
    source = args.checkpoint or OUT / "online-practice-learn-checkpoint.json"
    if held:
        state = _read(source)
        if args.phase == "held_openings" and set(state["release_ids"]) & {c["id"] for c in cards}:
            raise ValueError("Held release seen in practice")
        if set(state["reply_seeds"]) & {c["reply_seed"] for c in cards}:
            raise ValueError("Held reply seed seen in practice")
        bank = ExchangeMemory(state["records"])
    else:
        bank = ExchangeMemory()
    with np.load(MODEL) as raw:
        body = Body(raw)
    with np.load(RELATION) as archive:
        relation = Relation(archive)
    offsets = np.array(_read(SEARCH)["bias_rad"])
    shots = measured_launches(args.feeds)
    cap = args.cap
    receipt = dict(ready=model["ready"], cards=cards, phase=args.phase, mode=args.mode, rows=[],
                   opening_side=1, cap=cap, protocol=args.protocol, plan_sha256=_hash(args.course),
                   start_sha256=_hash(source) if held else None, feed_sha256=_hash(args.feeds),
                   dependencies={p.name: _hash(p) for p in (PUPIL, MODEL, RELATION, SEARCH)})
    output = args.output or OUT/f"online-{args.phase}-{args.mode}.json.gz"
    with centered(model["ready"]):
        arena = make_arena(protocol=args.protocol)
        stereo, rig, stride, lo, hi = joint.resources(arena)
        view = Observations(stereo, rig)
        try:
            for i, card in enumerate(cards):
                epsilon = max(.23, .65 - .12 * (i // 8)) if args.mode in ("learn", "frozen") else 0.
                active = len(bank.records)
                row = point(arena, view, rig, model, stride, lo, hi, body, relation, offsets,
                            shots[card["id"]], card, bank, np.random.default_rng(371851+i),
                            epsilon, args.mode == "learn", cap)
                row["memory_before"], row["memory_after"] = active, len(bank.records)
                receipt["rows"].append(row)
                _write(output, receipt)
                if args.mode == "learn":
                    _write(source, dict(records=bank.records, practice_id=i+1, source_sha256=_hash(output),
                                        release_ids=sorted({c["id"] for c in cards[:i+1]}),
                                        reply_seeds=sorted({c["reply_seed"] for c in cards[:i+1]})))
                print(args.phase, args.mode, i+1, card["id"], row["score"]["learner"],
                      row["score"]["returns"], active, len(bank.records), flush=True)
        finally:
            stereo.close()
            arena.close()


def stats(rows):
    own = [r["score"]["learner"] for r in rows]
    actions = [f for r in rows for f in r["feedback"]]
    return dict(points=len(rows), own=sum(own), first=sum(x>0 for x in own), longest=max(own, default=0),
                later=sum(max(0, x-1) for x in own), alternating=sum(r["score"]["returns"] for r in rows),
                choice_distribution=dict(Counter(f["choice"] for f in actions)), cpu_s=sum(r["runtime_cpu_s"] for r in rows),
                labels=sum(e["reward"] is not None for e in actions), missed_bearing_updates=sum(
                    e["miss_dq_rad"] is not None for e in actions),
                within_point_reads=sum(f["prior_outcomes"] > r["memory_before"]
                    for r in rows for f in r["feedback"]))


def summarize():
    sections = {}
    for phase, variants in {"practice": ("frozen", "learn"),
                            "held_openings": ("body", "carry"),
                            "held_replies": ("body", "carry")}.items():
        paths = {v: OUT/f"online-{phase}-{v}.json.gz" for v in variants}
        base, candidate = (_read(paths[v]) for v in variants)
        if any(len(d["cards"]) != len(d["rows"]) for d in (base, candidate)):
            raise ValueError("Incomplete play")
        if base["cards"] != candidate["cards"] or base["cap"] != candidate["cap"]:
            raise ValueError("Learner and control have different schedules")
        for doc in (base, candidate):
            expected = doc["rows"][0]["memory_before"]
            for row in doc["rows"]:
                if row["memory_before"] != expected:
                    raise ValueError("Causal memory count was not retained")
                if doc["mode"] == "learn":
                    expected = min(MAX_BANK, expected + sum(f["reward"] is not None for f in row["feedback"]))
                if row["memory_after"] != expected:
                    raise ValueError("Incomplete training updates or unplanned held updates")
                if row["score"]["learner"] != sum(f["legal"] for f in row["feedback"]):
                    raise ValueError("Observed stroke and referee awards disagree")
        active = [i for i, row in enumerate(base["rows"]) if row["placements"]["delivered"]]
        treated = [i for i, row in enumerate(candidate["rows"]) if row["placements"]["delivered"]]
        if active != treated:
            raise ValueError("A learner changed the number of initial legal opponent deliveries")
        selected = [dict(d, cards=[d["cards"][i] for i in active], rows=[d["rows"][i] for i in active])
                    for d in (base, candidate)]
        sections[phase] = dict(compare=paired(*selected), without_initial_offer=len(base["cards"])-len(active), variants={
            v: dict(stats=stats(d["rows"]), blocks=[stats(d["rows"][i:i+8]) for i in range(0,len(d["rows"]),8)],
                    source_sha256=_hash(paths[v])) for v,d in zip(variants, (base, candidate), strict=True)})
    _write(OUT/"online-results.json", sections)
    print(sections, flush=True)


def verify_changes():
    """Replay every changed held point in the independent full-CAD court."""
    checked = {}
    with TemporaryDirectory(prefix="tt37-online-verification-") as temporary:
        subset, audit_path = (Path(temporary)/f for f in ("games.json.gz", "check.json"))
        for phase in ("held_openings", "held_replies"):
            control = _read(OUT/f"online-{phase}-body.json.gz")
            source = OUT/f"online-{phase}-carry.json.gz"
            trained = _read(source)
            differences = [(i, b["score"]["learner"]-a["score"]["learner"])
                           for i, (a, b) in enumerate(zip(control["rows"], trained["rows"], strict=True))
                           if a["score"]["learner"] != b["score"]["learner"]]
            selected = dict(trained, cards=[trained["cards"][i] for i, _ in differences],
                            rows=[trained["rows"][i] for i, _ in differences])
            _write(subset, selected)
            with patch.object(joint, "replay_arena", lambda cards, **kw:
                              make_arena(cards, protocol=trained["protocol"], **kw)):
                joint.verify(subset, audit_path, cad=True)
            proof = _read(audit_path)
            if any(r["clearance"]["penetration_samples"] for r in proof["points"]):
                raise ValueError("A measured changed game penetrated the CAD envelope")
            checked[phase] = dict(carry_sha256=_hash(source), changes=differences, replay=proof["points"])
    _write(OUT/"online-audit.json", checked)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("practice", "held_openings", "held_replies", "summary", "audit"))
    p.add_argument("mode", choices=("learn", "frozen", "body", "carry"), nargs="?")
    p.add_argument("--course", type=Path, default=OUT/"opponent-plan.json")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--protocol", choices=("random","tt36","fixed"), default="random")
    p.add_argument("--limit", type=int)
    p.add_argument("--cap", type=int, default=24)
    args=p.parse_args()
    inspect = {"summary": summarize, "audit": verify_changes}.get(args.phase)
    inspect() if inspect else run(args)
