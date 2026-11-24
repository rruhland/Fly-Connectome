"""Re-score fixed online experience with separate feedback channels, then play held points."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt34_results import paired
from .tt35_replies import varied_opponent
from .tt36_online import OUT, Memory, figures, outcome, run

COMPONENTS = ("legal", "optical")


def checkpoint(component):
    return OUT / f"online-component-{component}-memory.json"


def tape(stage, component):
    if component in ("reset", "carry", "prior"):
        return OUT / f"online-{stage}-{component}.json.gz"
    return OUT / f"online-component-{stage}-{component}.json.gz"


def transform(records, component):
    """Replace only feedback values, or abstain where a visual label is absent."""
    result = []
    for row in records:
        if component == "legal":
            value = float(row["landed"])
        elif component == "optical":
            value = outcome(False, row["optical_outgoing_mps"], row["optical_gap_m"])
        else:
            raise ValueError(component)
        if value is not None:
            result.append(dict(context=row["context"], arm=row["arm"], reward=float(value)))
    return result


def prepare():
    baseline = OUT / "online-memory.json"
    practice = OUT / "online-practice-acquire.json.gz"
    state, archive = _read(baseline), _read(practice)
    if state["source_sha256"] != _hash(practice) or len(archive["rows"]) != 32:
        raise ValueError("Expected the unchanged 32-point online practice")
    original = Memory()
    for row in archive["rows"]:
        original.learn(row["feedback"])
    if original.bank != state["bank"]:
        raise ValueError("Ablations must be derived from exactly the retained mixed experience")

    for component in COMPONENTS:
        bank = transform(original.bank, component)
        _write(checkpoint(component), dict(
            state, bank=bank, reward_mode=component, full_memory_sha256=_hash(baseline),
            observations_in_full=len(original.bank), omitted_without_visible_outcome=len(original.bank) - len(bank),
            outcomes=dict(
                legal_pulses_in_full=sum(row["landed"] for row in original.bank),
                visible_outgoing_in_full=sum(row["optical_outgoing_mps"] is not None for row in original.bank),
                visible_gap_in_full=sum(row["optical_gap_m"] is not None for row in original.bank),
                nonzero_labels=sum(row["reward"] > 0. for row in bank),
            ),
        ))
        print(component, len(bank), "same acquired exposure", flush=True)


def play(stage, component, feeds):
    args = SimpleNamespace(stage=stage, mode="carry" if component in COMPONENTS else "prior",
                           course=None, limit=None, feeds=feeds, output=tape(stage, component),
                           memory=checkpoint(component) if component in COMPONENTS else None)
    run(args)


def summarize():
    practice = _read(OUT / "online-practice-acquire.json.gz")
    post = dict(checkpoints={}, recorded_approaches=sum(len(r["feedback"]) for r in practice["rows"]), held={})
    archived = _read(OUT / "online-memory.json")
    for c in COMPONENTS:
        state = _read(checkpoint(c))
        if state["bank"] != transform(archived["bank"], c):
            raise ValueError(f"{c} is not an ablation of the retained online checkpoint")
        post["checkpoints"][c] = dict(
            sha256=_hash(checkpoint(c)), outcomes=state["outcomes"], available=len(state["bank"]),
            omitted=state["omitted_without_visible_outcome"])
    for stage in ("replies", "new"):
        expected = 16 if stage == "replies" else 32
        modes = ("reset", "carry", *COMPONENTS, "prior")
        tapes = {mode: _read(tape(stage, mode)) for mode in modes}
        frozen, full = (tapes[v] for v in ("reset", "carry"))
        report = {}
        for mode, doc in tapes.items():
            if doc["reply_seeds"] != frozen["reply_seeds"] or len(doc["rows"]) != expected:
                raise ValueError(f"Different or incomplete held course for {stage} and {mode}")
            if any(doc[k] != frozen[k] for k in ("base_sha256", "feed_sha256", "actions_rad", "cap")):
                raise ValueError("A component unexpectedly changed the motor or the delivery")
            if doc["start_memory"] != (len(archived["bank"]) if mode == "carry" else
                                       len(_read(checkpoint(mode))["bank"]) if mode in COMPONENTS else 0):
                raise ValueError("Wrong retained experience in held play")
            if mode in COMPONENTS and doc["memory_sha256"] != _hash(checkpoint(mode)):
                raise ValueError(f"{mode} played a different checkpoint")
            if any(row["start_memory"] != row["end_memory"] or row["start_memory"] != doc["start_memory"]
                   for row in doc["rows"]):
                raise ValueError("Held play updated the learner")
            report[mode] = dict(stats=figures(doc["rows"]), reset_to_mode=paired(frozen, doc),
                                full_to_mode=paired(full, doc), source_sha256=_hash(tape(stage, mode)),
                                identical_full_action_tapes=sum(
                                    digest_json(v["actions"]) == digest_json(f["actions"])
                                    for v, f in zip(doc["rows"], full["rows"], strict=True)))
            print(stage, mode, report[mode]["stats"]["returns"], report[mode]["stats"]["later"], flush=True)
        post["held"][stage] = dict(cards=frozen["cards"], reply_seeds=frozen["reply_seeds"], modes=report)
    _write(OUT / "online-component-results.json", post)


def audit():
    """Replay one large score change or the longest tied point per component and course."""
    verified = {}
    with TemporaryDirectory(prefix="tt36-components-") as temporary:
        subset = Path(temporary) / "tape.json.gz"
        receipt = Path(temporary) / "audit.json"
        for stage in ("replies", "new"):
            full = _read(tape(stage, "carry"))
            for c in COMPONENTS:
                source = tape(stage, c)
                variant = _read(source)
                differences = [a["score"]["learner"] - b["score"]["learner"]
                               for a, b in zip(variant["rows"], full["rows"], strict=True)]
                position = max(range(len(differences)),
                               key=lambda i: (abs(differences[i]), variant["rows"][i]["score"]["learner"]))
                card, seed, row = (variant[key][position] for key in ("cards", "reply_seeds", "rows"))
                _write(subset, dict(variant, cards=[card], rows=[row]))
                branch = varied_opponent(card["index"], seed) if seed is not None else nullcontext()
                with branch:
                    joint.verify(subset, receipt, cad=True)
                verified[f"{stage}-{c}"] = dict(
                    _read(receipt)["points"][0], position=position, reply_seed=seed,
                    delta_vs_full=differences[position], source_sha256=_hash(source))
    if any(r["clearance"]["penetration_samples"] for r in verified.values()):
        raise ValueError("A replayed comparison game penetrated the robot envelope")
    _write(OUT / "online-component-audits.json", verified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "play", "summary", "audit"))
    parser.add_argument("--stage", choices=("new", "replies"))
    parser.add_argument("--component", choices=(*COMPONENTS, "prior"))
    parser.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    args = parser.parse_args()
    phase = {"prepare": prepare, "summary": summarize, "audit": audit}.get(args.phase)
    if phase:
        phase()
    elif args.stage is None or args.component is None:
        parser.error("play requires --stage and --component")
    else:
        play(args.stage, args.component, args.feeds)
