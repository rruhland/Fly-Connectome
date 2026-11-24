"""Learn from physical return scores whether to trust native late steering."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_capture import OUTPUT
from .tt37_joint_deep import run
from .tt37_joint_deep_report import totals
from .tt37_joint_report import paired, read_complete, replay_case
from .tt37_joint_augment import CandidateField, MixedField


SCALE = np.array([.16, .26, .15, 1.20, 1.25, 1.50])
SEEDS = {"practice": 1029379000, "challenge": 1041153000}
TREATMENTS = ("rgb", "gate", "active", "carry")


def make_plan(folder):
    archive = folder / "joint-deep-observations.json"
    before = _read(archive)
    one = {row["card"]["id"]: row["card"] for row in before["practice"]}
    prior = {row["card"]["reply_seed"] for row in before["practice"]}
    for f in (folder / "joint-deep-fresh_replies-local.json.gz",
              folder / "joint-deep-held_openings-local.json.gz",
              *(folder.glob("clinic-labels-*.json.gz"))):
        prior.update(card["reply_seed"] for card in _read(f)["cards"])
    decks = {}
    for name, seed in SEEDS.items():
        cards = [dict(card, reply_seed=seed + 29 * i + j)
                 for i, card in enumerate(list(one.values())[:12]) for j in range(2)]
        new = {c["reply_seed"] for c in cards}
        if len(new) != 24 or new & prior:
            raise ValueError("The fixed trust-learning seeds are not independent of existing assessment")
        prior.update(new)
        decks[name] = cards
    _write(folder / "joint-trust-plan.json", dict(
        **decks, receipt_sha256=_hash(archive), existing_reply_seeds_excluded=True,
        note="Twelve familiar measured openings, two separate random continuation streams each. "
             "No school-plan cases and no prior reply labels.",
    ))


class TrustField:
    """Choose a direct steering mix using only observable state and physical reward."""

    def __init__(self, rgb, native, mode, records=()):
        self.rgb = rgb
        self.native = native
        self.mixed = MixedField(rgb, native)
        self.mode = mode
        self.records = list(records)
        self.decisions = 0
        self.begin_point()

    def begin_point(self):
        self.mixed.begin_point()
        self.diagnostics = dict(rgb_choices=0, coherent_choices=0, updates=[], candidate_ticks=0)
        self.score = 0
        self.choice = None
        self.arrived = False

    def choose(self, state):
        count = self.decisions
        self.decisions += 1
        if self.mode == "active" and (count < 8 or count % 5 == 0):
            return (count % 2) if count < 8 else ((count // 5) % 2)
        if not self.records:
            return 1
        old = np.asarray([record["state"] for record in self.records])
        same = np.exp(-.5 * np.linalg.norm((old - state) / SCALE, axis=1)**2)
        arm = np.asarray([record["coherent"] for record in self.records])
        reward = np.asarray([record["reward"] for record in self.records])
        estimate = []
        for i in (0, 1):
            w = same * (arm == i)
            estimate.append((1. + .08 * i + np.sum(w * reward)) / (2. + np.sum(w)))
        return int(estimate[0] <= estimate[1] + .025)

    def get(self, optical, q, *, reference=None):
        if not self.score or optical[4] > -.9:
            return self.rgb.get(optical, q, reference=reference)
        if self.mode == "gate":
            value = self.mixed.get(optical, q, reference=reference)
            self.diagnostics["candidate_ticks"] = self.mixed.diagnostics["candidate_ticks"]
            return value
        if self.choice is None:
            _, support = self.native.get(optical, q, reference=reference)
            if support:
                arm = self.choose(optical)
                self.choice = dict(state=optical.tolist(), coherent=arm)
                self.diagnostics["coherent_choices" if arm else "rgb_choices"] += 1
        if self.choice is not None:
            if optical[1] < -.86:
                self.arrived = True
            if self.choice["coherent"]:
                value = self.mixed.get(optical, q, reference=reference)
                self.diagnostics["candidate_ticks"] = self.mixed.diagnostics["candidate_ticks"]
                return value
        return self.rgb.get(optical, q, reference=reference)

    def record(self, reward):
        if self.choice is not None and self.mode == "active":
            update = dict(self.choice, reward=int(reward))
            self.records.append(update)
            self.records = self.records[-128:]
            self.diagnostics["updates"].append(update)
        self.choice = None
        self.arrived = False

    def score_feedback(self, score):
        if score > self.score:
            if self.score:
                self.record(True)
            self.score = score

    def end_point(self, *, terminal):
        if terminal and self.arrived:
            self.record(False)


def assess(folder, feeds, deck, arm, output, *, cap, carry=None):
    roster = folder / "joint-trust-plan.json"
    doc = _read(roster)
    if doc["receipt_sha256"] != _hash(folder / "joint-deep-observations.json"):
        raise ValueError("The frozen practice roster was not built for this RGB learner")
    chooser = _read(folder / "joint-native-choice.json")
    mode = _read(folder / "joint-deep-choice.json")["mode"]
    native_bank = folder / "joint-native-candidate.npz"
    if chooser["bank_sha256"] != _hash(native_bank):
        raise ValueError("Native calibration changed")
    with np.load(native_bank) as bank:
        native = CandidateField(bank, mode, chooser["variant"])
    if (arm == "carry") != (carry is not None):
        raise ValueError("Only the carried policy accepts a memory from practice")
    prior = []
    if carry is not None:
        checkpoint = _read(carry)
        practice = folder / "joint-trust-practice-active.json.gz"
        if (checkpoint["native_sha256"] != _hash(native_bank)
                or checkpoint["practice_sha256"] != _hash(roster)
                or checkpoint["outcome_sha256"] != _hash(practice)):
            raise ValueError("Carried memory is not bound to the completed physical practice ledger")
        prior = checkpoint["records"]
    if arm == "active" and deck != "practice":
        raise ValueError("No adaptation is allowed on the independent challenge")
    instances = []

    def wrap(rgb):
        if arm == "rgb":
            return rgb
        policy = TrustField(rgb, native, arm, prior)
        instances.append(policy)
        return policy

    run(feeds, folder / "joint-deep-observations.npz", roster, deck, mode, output,
        cap=cap, field_hook=wrap, cards_override=doc[deck],
        details=dict(arm=arm, native_sha256=_hash(native_bank), carry_sha256=_hash(carry) if carry else None))
    if arm == "active":
        _write(folder / "joint-trust-carry.json", dict(
            records=instances[0].records, outcome_sha256=_hash(output), practice_sha256=_hash(roster),
            native_sha256=_hash(native_bank), actor="frozen continuation trust only",
        ))


def report(folder):
    scores = {}
    for deck, arms in (("practice", ("gate", "active")), ("challenge", ("rgb", "gate", "carry"))):
        docs = {arm: read_complete(folder / f"joint-trust-{deck}-{arm}.json.gz") for arm in arms}
        details = {}
        for arm, doc in docs.items():
            if doc["candidate_source"]["arm"] != arm:
                raise ValueError("Unexpected treatment in the saved physical point tapes")
            stats = totals(doc)
            diag = [r.get("controller_diagnostics", {}) for r in doc["rows"]]
            stats.update(**{k: sum(row.get(k, 0) for row in diag)
                            for k in ("rgb_choices", "coherent_choices", "candidate_ticks")})
            stats["updates"] = sum(len(row.get("updates", [])) for row in diag)
            stats["outcome_memory"] = {
                name: dict(Counter("scored" if update["reward"] else "no_score"
                                   for row in diag for update in row.get("updates", [])
                                   if update["coherent"] == index))
                for index, name in enumerate(("rgb", "coherent"))
            }
            details[arm] = stats
        control = docs["gate"]
        def opening(doc):
            return [next((s for s in r["score"]["shots"] if s["side"] == 1), None) for r in doc["rows"]]
        if any(opening(doc) != opening(control) for doc in docs.values()):
            raise ValueError("Continuation-only steering changed opening results")
        scores[deck] = dict(stats=details, exact_first_near_shots=True,
                            vs_gate={k: paired(control, v) for k, v in docs.items() if k != "gate"})
    _write(folder / "joint-trust-results.json", scores)


def audit(folder):
    path = folder / "joint-trust-challenge-carry.json.gz"
    carried = read_complete(path)
    gate = read_complete(folder / "joint-trust-challenge-gate.json.gz")
    if carried["cards"] != gate["cards"]:
        raise ValueError("Challenge changed")
    i = max(range(len(carried["rows"])), key=lambda i: carried["rows"][i]["score"]["learner"])
    item = replay_case(carried["ready"], "random", carried["cards"][i], carried["rows"][i])
    _write(folder / "joint-trust-audit.json", dict(
        **item, source_sha256=_hash(path), gate_own=gate["rows"][i]["score"]["learner"],
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--folder", type=Path, default=OUTPUT)
    assess_cli = commands.add_parser("assess")
    assess_cli.add_argument("feeds", type=Path)
    assess_cli.add_argument("deck", choices=tuple(SEEDS))
    assess_cli.add_argument("arm", choices=TREATMENTS)
    assess_cli.add_argument("output", type=Path)
    assess_cli.add_argument("--carry", type=Path)
    assess_cli.add_argument("--cap", type=int, default=200)
    assess_cli.add_argument("--folder", type=Path, default=OUTPUT)
    summary = commands.add_parser("report")
    summary.add_argument("--folder", type=Path, default=OUTPUT)
    replay = commands.add_parser("audit")
    replay.add_argument("--folder", type=Path, default=OUTPUT)
    a = p.parse_args()
    if a.command == "plan":
        make_plan(a.folder)
    elif a.command == "report":
        report(a.folder)
    elif a.command == "audit":
        audit(a.folder)
    else:
        assess(a.folder, a.feeds, a.deck, a.arm, a.output, cap=a.cap, carry=a.carry)
