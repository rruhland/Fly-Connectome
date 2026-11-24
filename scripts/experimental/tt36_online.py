"""Learn a local categorical joint residual once per real rally from visual and score evidence."""

import argparse
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_curriculum_summary import compare_delivery
from .tt31_ready import centered
from .tt34_results import paired
from .tt35_error import Observations, passage
from .tt35_replies import varied_opponent

ROOT = Path(__file__).resolve().parents[2] / "artifacts"
OUT = ROOT / "tt36"
BASE = ROOT / "tt33" / "joint-raw.json.gz"
SEARCH = ROOT / "tt35" / "search-checkpoint.json"
PRACTICE = ROOT / "tt35" / "confirmation-plan.json.gz"
MAX_POINTS_IN_MEMORY = 48
CAP = 16
SEED = 362041
SCALE = np.array([.17, .37, .19, .75, 1.9, 3.1, 1.75, 1.75, 1.75, 1.75, 1.75])
PRIOR = np.array([.28, .23, .23, .23])
REGULARIZATION = .60
SUPPORT = .30
EXPLORE = .50


def templates():
    prior = np.array(_read(SEARCH)["bias_rad"])
    return np.array([np.zeros(5), prior, -.75 * prior, [.07, -.04, .03, -.035, 0.]])


class Memory:
    """Kernel local means with a weak fallback and bounded measured evidence."""

    def __init__(self, bank=()):
        self.bank = list(bank)

    def greedy(self, context):
        if not self.bank:
            return 0
        x = np.array([r["context"] for r in self.bank])
        arms = np.array([r["arm"] for r in self.bank])
        reward = np.array([r["reward"] for r in self.bank])
        k = np.exp(-.5 * np.sum(np.square((x - context) / SCALE), axis=1))
        mass = np.bincount(arms, weights=k, minlength=len(PRIOR))
        value = (np.bincount(arms, weights=k * reward, minlength=len(PRIOR))
                 + REGULARIZATION * PRIOR) / (mass + REGULARIZATION)
        value[1:][mass[1:] < SUPPORT] = -np.inf
        return int(np.argmax(value))

    def learn(self, records):
        self.bank.extend(record for record in records if record["reward"] is not None)
        self.bank = self.bank[-MAX_POINTS_IN_MEMORY:]


class GlobalPrior(Memory):
    """Keep the inherited reward offset at every approach, with no local learning."""

    def greedy(self, context):
        return 1


class Actor(joint.JointPolicy):
    """Read only current optical ball state, current encoders, and previous learned evidence."""

    def __init__(self, model, stride, low, high, memory, actions, rng, explore):
        super().__init__(model, stride, low, high)
        self.memory, self.rng, self.explore = memory, rng, explore
        self.variants = [self.weights, *(self.weights + np.r_[u, 0.] for u in actions[1:])]
        self.choice = None
        self.choices = []
        self.previous_depth = None

    def command(self, optical, q, tick, outgoing):
        if outgoing:
            self.choice = None
            self.previous_depth = None

        if optical is not None:
            depth = optical[1]
            # A fresh approach after a previous near-arm flight still re-arms when
            # motion reversal itself was occluded.
            if self.previous_depth is not None and self.previous_depth < -.38 and depth > .46:
                self.choice = None
            self.previous_depth = depth

            if self.choice is None and -1.18 < depth <= .42:
                context = np.r_[optical, q[:5]]
                exploit = self.memory.greedy(context)
                exploratory = self.rng.random() < self.explore
                arm = int(self.rng.integers(len(self.variants))) if exploratory else exploit
                self.choice = dict(tick=tick, context=context.tolist(), arm=arm, exploit=exploit,
                                   exploration=exploratory)
                self.choices.append(self.choice)

        # The offset in each coefficient preserves baseline attenuation outside
        # the readout's demonstrated support, as in the TT35 frozen adjustment.
        self.weights = self.variants[self.choice["arm"] if self.choice else 0]
        return super().command(optical, q)


def outcome(landed, visible_outgoing, gap):
    if landed:
        return 1.
    if visible_outgoing is not None:
        return .36 + .25 * np.clip((visible_outgoing - .9) / 3.2, 0., 1.)
    if gap is not None:
        return .32 * np.exp(-gap / .12)
    return None


def labels(choices, optical_rows, outgoing, events, end, dt):
    """Attach observed outcomes after play, using only RGB and the earned-return pulse."""
    legal = [e["time"] / dt for e in events if e["kind"] == "legal_return" and e["side"] == 1]
    records = []
    for choice, following in zip(choices, choices[1:] + [{"tick": end + 1}], strict=True):
        i, j = choice["tick"], following["tick"]
        landed = any(i <= tick < j for tick in legal)
        window = [(tick, ball, blade) for tick, ball, blade in optical_rows if i <= tick < j]
        crossing = passage(window, i)
        gap = None if crossing is None else float(np.linalg.norm(crossing["gap_xz_m"]))
        # Strength from an observed ball already moving away, no projected landing.
        speed = max((v for tick, v in outgoing if i <= tick < j), default=None)
        reward = outcome(landed, speed, gap)
        records.append(dict(choice, reward=reward, landed=landed, optical_outgoing_mps=speed,
                            optical_gap_m=gap, optical_passage=crossing))
    return records


def point(arena, view, rig, model, stride, low, high, shot, card, seed, memory, actions, explore):
    arena.launch(shot, card=card)
    arena.partner.reset(seed=card["seed"])
    view.begin()
    actor = Actor(model, stride, low, high, memory, actions, np.random.default_rng(seed), explore)
    eye = joint.Vision(rig, arena.control_dt, raw=True)
    tape, outgoing = [], []
    started = process_time()
    for tick in range(3600):
        q = arena.joints
        optical = eye.update(tick, view.capture() if tick % 2 == 0 else None, q)
        f = eye.eye.latest
        observed_out = (f is not None and f.velocity[1] > .9 and f.rms < 2.4
                        and f.depth_sigma < .16 and 0 <= tick * arena.control_dt - f.time < .07)
        if (observed_out and tick % 2 == 0 and f.time == tick * arena.control_dt
                and -1.58 < f.position[1] < -.47 and actor.choices):
            outgoing.append((tick, float(f.velocity[1])))
        command = actor.command(optical, q, tick, observed_out)
        _, _, done, capped, _ = arena.exchange(command, joint.FAR_COMMAND)
        tape.append(command.tolist())
        if done or capped or arena.legal_returns + arena.rival_returns - 1 >= CAP:
            break
    records = labels(actor.choices, view.rows, outgoing, arena.events, len(tape), arena.control_dt)
    return dict(shot=shot, opponent_seed=card["seed"], actions=tape, score=audit(arena.events, opening_side=1),
                events=arena.events.copy(), placements=arena.point_placements(), feedback=records,
                runtime_cpu_s=process_time() - started, motor_audit=arena.motor_audit())


def roster(section, path=None):
    if section == "practice":
        varied_path = OUT / "course-replies.json.gz"
        prior_cards = _read(varied_path)["cards"]
        familiar = list({c["id"]: c for c in prior_cards}.values())
        pool = [c for c in _read(PRACTICE)["cards"] if c["id"] not in {f["id"] for f in familiar}]
        rng = np.random.default_rng(SEED)
        shuffled = [pool[i] for i in rng.permutation(len(pool))[:16]]
        cards, replies = [], []
        for block in range(4):
            cards.extend(familiar)
            replies.extend(range(362100 + 4 * block, 362104 + 4 * block))
            cards.extend(shuffled[4 * block:4 * block + 4])
            replies.extend([None] * 4)
        return list(zip(cards, replies, strict=True))
    if path is None:
        path = OUT / f"course-{section}.json.gz"
    source = _read(path)
    return list(zip(source["cards"], source.get("reply_seeds", [None] * len(source["cards"])), strict=True))


def run(args):
    stage, mode = args.stage, args.mode
    training = stage == "practice"
    if mode not in (("acquire", "frozen") if training else ("carry", "reset", "prior")):
        raise ValueError("Use acquire/frozen during practice or carry/reset/prior on held courses")
    cards, reply_seeds = map(list, zip(*roster(stage, args.course), strict=True))
    if args.limit is not None:
        cards, reply_seeds = cards[:args.limit], reply_seeds[:args.limit]
    base = _read(BASE)
    actions = templates()
    if mode == "carry":
        state_path = args.memory or OUT / "online-memory.json"
        state = _read(state_path)
        if state["base_sha256"] != _hash(BASE) or state["actions_rad"] != actions.tolist():
            raise ValueError("Memory was learned using different joint targets")
        if stage == "new" and {card["id"] for card in cards} & set(state["practice_ids"]):
            raise ValueError("A final release was present in learning")
        if stage == "replies" and set(reply_seeds) & set(state["practice_reply_seeds"]):
            raise ValueError("A final reply seed was present in learning")
        memory = Memory(state["bank"])
    else:
        state_path = None
        memory = GlobalPrior() if mode == "prior" else Memory()
    start_count = len(memory.bank)
    shots = measured_launches(args.feeds)
    path = args.output or OUT / f"online-{stage}-{mode}.json.gz"
    doc = dict(schema="tt36-online-1", stage=stage, mode=mode, cards=cards, reply_seeds=reply_seeds,
               start_memory=start_count, rows=[], ready=base["ready"], opening_side=1, cap=CAP,
               base_sha256=_hash(BASE), feed_sha256=_hash(args.feeds), actions_rad=actions.tolist(),
               memory_sha256=_hash(state_path) if state_path else None)
    with centered(base["ready"]):
        arena = replay_arena([])
        stereo, rig, stride, low, high = joint.resources(arena)
        view = Observations(stereo, rig)
        try:
            for i, (card, reply_seed) in enumerate(zip(cards, reply_seeds, strict=True)):
                branch = varied_opponent(card["index"], reply_seed) if reply_seed is not None else nullcontext()
                with branch:
                    row = point(arena, view, rig, base, stride, low, high, shots[card["id"]], card,
                                SEED + i, memory, actions, EXPLORE if training else 0.)
                row["start_memory"] = len(memory.bank)
                if mode == "acquire":
                    memory.learn(row["feedback"])
                row["end_memory"] = len(memory.bank)
                doc["rows"].append(row)
                _write(path, doc)
                print(stage, mode, i + 1, card["id"], row["score"]["learner"], len(memory.bank),
                      [(a["arm"], None if a["reward"] is None else round(a["reward"], 3))
                       for a in row["feedback"]], flush=True)
        finally:
            stereo.close()
            arena.close()
    if mode == "acquire" and args.limit is None:
        _write(OUT / "online-memory.json", dict(base_sha256=_hash(BASE), actions_rad=actions.tolist(),
                source_sha256=_hash(path), bank=memory.bank, practice_ids=sorted({c["id"] for c in cards}),
                practice_reply_seeds=[s for s in reply_seeds if s is not None]))


def figures(rows):
    counts = [r["score"]["learner"] for r in rows]
    replies = [x for r in rows for x in r["placements"]["delivered"]]
    feedback = [entry for r in rows for entry in r["feedback"]]
    return dict(points=len(rows), returns=sum(counts), openings=sum(c > 0 for c in counts),
                later=sum(max(c - 1, 0) for c in counts),
                alternating=sum(r["score"]["returns"] for r in rows),
                contacts=sum(x["near_contact"] is not None for x in replies),
                opportunities=len(replies), choices=dict(Counter(x["arm"] for x in feedback)),
                interventions=sum(bool(x["arm"]) for x in feedback),
                optical_misses=sum(x["optical_gap_m"] is not None and x["optical_outgoing_mps"] is None
                                  and not x["landed"] for x in feedback),
                optical_nonscoring_outgoing=sum(x["optical_outgoing_mps"] is not None and not x["landed"]
                                                for x in feedback),
                rewarded_returns=sum(x["landed"] for x in feedback),
                process_cpu_s=sum(r["runtime_cpu_s"] for r in rows))


def openings(doc):
    return {str(i): next(iter(row["placements"]["delivered"]), None)
            for i, row in enumerate(doc["rows"])}


def summarize():
    result = {}
    prefix_bank = Memory()
    snapshots = {}
    for stage, modes in {"practice": ("acquire", "frozen"), "new": ("carry", "reset"),
                         "replies": ("carry", "reset")}.items():
        loaded = [_read(OUT / f"online-{stage}-{m}.json.gz") for m in modes]
        a, b = loaded
        if any((d["cards"], d["reply_seeds"]) != (a["cards"], a["reply_seeds"]) for d in loaded):
            raise ValueError("The deck differs across controls")
        if any(len(d["rows"]) != len(d["cards"]) for d in loaded):
            raise ValueError("An incomplete tape is not an assessment")
        paired(b, a)
        for doc in loaded:
            expected = doc["start_memory"]
            for row in doc["rows"]:
                if row["start_memory"] != expected:
                    raise ValueError("An outcome entered memory before the point was complete")
                if doc["mode"] == "acquire":
                    expected = min(MAX_POINTS_IN_MEMORY, expected + sum(
                        x["reward"] is not None for x in row["feedback"]))
                if row["end_memory"] != expected:
                    raise ValueError("Wrong outcome count or unexpected assessment update")
                if sum(x["landed"] for x in row["feedback"]) != row["score"]["learner"]:
                    raise ValueError("A return pulse did not attach to exactly one optical approach")
        changes = [x["score"]["learner"] - y["score"]["learner"]
                   for x, y in zip(a["rows"], b["rows"], strict=True)]
        pass_gaps = [(r1["feedback"][0]["optical_gap_m"], r2["feedback"][0]["optical_gap_m"])
                     for r1, r2 in zip(a["rows"], b["rows"], strict=True)
                     if r1["feedback"] and r2["feedback"] and
                     all(r["placements"]["delivered"] and r["placements"]["delivered"][0]["near_contact"] is None
                         for r in (r1, r2))]
        common_gaps = [x for x in pass_gaps if None not in x]
        result[stage] = dict(
            paired=dict(wins=sum(c > 0 for c in changes), losses=sum(c < 0 for c in changes),
                        ties=changes.count(0), matches=compare_delivery(openings(a), openings(b)),
                        match_actions=sum(digest_json(r1["actions"]) == digest_json(r2["actions"])
                                          for r1, r2 in zip(a["rows"], b["rows"], strict=True)),
                        common_opening_optical_misses=len(common_gaps),
                        mean_gap_m={m: float(np.mean([p[j] for p in common_gaps])) if common_gaps else None
                                    for j, m in enumerate(modes)},
                        per_point=list(zip([c["id"] for c in a["cards"]], a["reply_seeds"], changes, strict=True))),
            stages={m: figures(d["rows"]) for m, d in zip(modes, loaded, strict=True)},
            successive_eights={m: [figures(d["rows"][i:i + 8]) for i in range(0, len(d["rows"]), 8)]
                               for m, d in zip(modes, loaded, strict=True)},
            sha256={m: _hash(OUT / f"online-{stage}-{m}.json.gz") for m in modes})
        if stage == "practice":
            snapshots[0] = Memory()
            for n, row in enumerate(a["rows"], 1):
                prefix_bank.learn(row["feedback"])
                if n % 8 == 0:
                    snapshots[n] = Memory(prefix_bank.bank)
            state_path = OUT / "online-memory.json"
            state = _read(state_path)
            if state["bank"] != prefix_bank.bank or state["source_sha256"] != result[stage]["sha256"]["acquire"]:
                raise ValueError("The checkpoint is not the endpoint of the recorded online run")
            result[stage]["memory_sha256"] = _hash(state_path)
        else:
            if a["memory_sha256"] != result["practice"]["memory_sha256"]:
                raise ValueError("The carry run was not started from the recorded learned checkpoint")
            contexts = [np.array(row["feedback"][0]["context"]) for row in b["rows"] if row["feedback"]]
            result[stage]["opening_choice_by_practice_dose"] = {
                n: dict(Counter(snapshot.greedy(context) for context in contexts))
                for n, snapshot in snapshots.items()
            }
            hits = [record for record in prefix_bank.bank
                    if record["landed"] or record["optical_outgoing_mps"] is not None]
            misses = [record for record in prefix_bank.bank if record not in hits]
            result[stage]["opening_choice_ablation_not_replayed"] = {
                name: dict(examples=len(records), differs_from_full=sum(
                    prefix_bank.greedy(context) != partial.greedy(context) for context in contexts))
                for name, records in (("hits_only", hits), ("misses_only", misses))
                for partial in (Memory(records),)
            }
            if stage == "replies":
                prior_path = OUT / "online-replies-prior.json.gz"
                if prior_path.is_file():
                    prior = _read(prior_path)
                    if (len(prior["rows"]) != len(a["rows"])
                            or prior["reply_seeds"] != a["reply_seeds"]
                            or any(row["start_memory"] or row["end_memory"] for row in prior["rows"])
                            or any(x["arm"] != 1 for row in prior["rows"] for x in row["feedback"])):
                        raise ValueError("Incomplete inherited global-offset control")
                    result[stage]["inherited_global_offset"] = dict(
                        sha256=_hash(prior_path), stats=figures(prior["rows"]),
                        compare_to_carry=paired(prior, a))
    _write(OUT / "online-results.json", result)
    for stage, entry in result.items():
        print(stage, entry["stages"], entry["paired"], flush=True)


def audit_results():
    """Compare the reset directly to the raw nominal and verify changed games with CAD."""
    result = {}
    for stage in ("new", "replies"):
        carry_path = OUT / f"online-{stage}-carry.json.gz"
        carry = _read(carry_path)
        reset = _read(OUT / f"online-{stage}-reset.json.gz")
        fixed = _read(OUT / f"paired-{stage}-fixed.json.gz")
        if (reset["cards"] != fixed["cards"]
                or any(digest_json(a["actions"]) != digest_json(b["actions"])
                       or digest_json(a["events"]) != digest_json(b["events"])
                       for a, b in zip(reset["rows"], fixed["rows"], strict=True))):
            raise ValueError("Reset does not reproduce the raw pupil")
        audits = []
        differences = [i for i, (a, b) in enumerate(zip(carry["rows"], reset["rows"], strict=True))
                       if a["score"]["learner"] != b["score"]["learner"]]
        if not differences:
            # If the controls tie, audit one learned intervention anyway.
            differences = [i for i, row in enumerate(carry["rows"])
                           if any(f["arm"] for f in row["feedback"])][:1]
        with TemporaryDirectory(prefix="tt36-online-") as temporary:
            source = Path(temporary) / "game.json.gz"
            receipt = Path(temporary) / "cad.json"
            for i in differences:
                card, row, seed = carry["cards"][i], carry["rows"][i], carry["reply_seeds"][i]
                _write(source, dict(carry, cards=[card], rows=[row]))
                branch = varied_opponent(card["index"], seed) if seed is not None else nullcontext()
                with branch:
                    joint.verify(source, receipt, cad=True)
                checked = _read(receipt)["points"][0]
                if checked["clearance"]["penetration_samples"]:
                    raise ValueError("An audited game penetrated the robot envelope")
                audits.append(dict(checked, reply_seed=seed, position=i))
        result[stage] = dict(source_sha256=_hash(carry_path), exact_nominal_reproductions=len(reset["rows"]),
                             audited=audits)
    _write(OUT / "online-audits.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("practice", "new", "replies", "summary", "audit"))
    parser.add_argument("mode", nargs="?")
    parser.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    parser.add_argument("--course", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--memory", type=Path, help="Use a frozen acquired or ablated checkpoint in carry mode")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    entry = {"summary": summarize, "audit": audit_results}.get(arguments.stage)
    entry() if entry else run(arguments)
