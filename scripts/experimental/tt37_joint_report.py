"""Select a field on practice and compare paired random-opponent games."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_arm_clearance import ArmClearance
from .tt29_endurance import audit
from .tt29_rally_physics import mujoco
from .tt31_contacts import record
from .tt31_ready import centered
from .tt37_joint_capture import OUTPUT
from .tt37_opponent import arena


DECKS = ("held_openings", "held_replies")
ARM = ("reach", "field", "carry")
VARIANTS = ("recovery", "gated", "plain", "body")


def totals(doc):
    rows = doc["rows"]
    own = [r["score"]["learner"] for r in rows]
    return dict(
        points=len(rows), offered=sum(len(r["placements"]["delivered"]) for r in rows),
        contacts=sum(p["near_contact"] is not None for r in rows for p in r["placements"]["delivered"]),
        own=sum(own), later=sum(max(0, n - 1) for n in own), longest_own=max(own, default=0),
        alternating=sum(r["score"]["returns"] for r in rows),
        longest_alternating=max((r["score"]["returns"] for r in rows), default=0),
        episodes_with_later=sum(n > 1 for n in own), field_ticks=sum(r["field_ticks"] for r in rows),
        recovery_ticks=sum(r["recovery_ticks"] for r in rows),
        updates=sum(len(r.get("updates", [])) for r in rows),
        update_frames=sum(u["frames"] for r in rows for u in r.get("updates", [])),
        runtime_cpu_s=sum(r["runtime_cpu_s"] for r in rows),
    )


def read_complete(path):
    doc = _read(path)
    if len(doc["rows"]) != len(doc["cards"]):
        raise ValueError(f"Incomplete run {path.name}")
    return doc


def select(folder):
    paths = {mode: folder / f"joint-practice-{mode}.json.gz" for mode in ("reach", *VARIANTS)}
    docs = {mode: read_complete(path) for mode, path in paths.items()}
    if any(doc["cards"] != docs["reach"]["cards"] for doc in docs.values()):
        raise ValueError("Practice cards changed")
    stats = {mode: totals(doc) for mode, doc in docs.items()}
    selected = max(VARIANTS, key=lambda mode: (
        stats[mode]["own"] + stats[mode]["later"], stats[mode]["own"], -VARIANTS.index(mode)))
    _write(folder / "joint-field-choice.json", dict(
        chosen=selected, stats=stats, sources={mode: _hash(path) for mode, path in paths.items()},
        choice="maximize legal returns plus later legal returns, then prefer limited strike intervention",
    ))
    print(selected, stats, flush=True)


def paired(left, right):
    if left["cards"] != right["cards"]:
        raise ValueError("Paired random card roster changed")
    rows = []
    both_missing = first_offer_exact = matched = serving_equal = 0
    for card, a, b in zip(left["cards"], left["rows"], right["rows"], strict=True):
        before, after = (next(iter(doc["placements"]["delivered"]), None) for doc in (a, b))
        both_missing += before is None and after is None
        if before is None and after is None:
            serving_equal += digest_json(a["events"]) == digest_json(b["events"])
        elif before is not None and after is not None:
            matched += before["target"] == after["target"]
            first_offer_exact += all(digest_json(before[key]) == digest_json(after[key])
                                     for key in ("far_contact", "near_bounce", "receive_line"))
        rows.append(dict(id=card["id"], seed=card["seed"], reply_seed=card["reply_seed"],
                         near=[a["score"]["learner"], b["score"]["learner"]],
                         alternating=[a["score"]["returns"], b["score"]["returns"]]))
    return dict(both_missing=both_missing, same_missing_referee_stream=serving_equal,
                first_target_matches=matched, first_offer_exact=first_offer_exact,
                wins=sum(r["near"][1] > r["near"][0] for r in rows),
                losses=sum(r["near"][1] < r["near"][0] for r in rows), rows=rows)


def summarize(folder):
    calibration = _read(folder / "joint-field-choice.json")
    active = read_complete(folder / "joint-practice-active.json.gz")
    fit = read_complete(folder / f"joint-practice-{calibration['chosen']}.json.gz")
    result = dict(practice=calibration, live_stats=totals(active), live_vs_frozen=paired(fit, active), held={})
    for deck in DECKS:
        files = {mode: folder / f"joint-{deck}-{mode}.json.gz" for mode in ARM}
        docs = {mode: read_complete(path) for mode, path in files.items()}
        result["held"][deck] = dict(
            scores={mode: totals(doc) for mode, doc in docs.items()},
            versus_reach={mode: paired(docs["reach"], doc) for mode, doc in docs.items() if mode != "reach"},
            live_vs_frozen=paired(docs["field"], docs["carry"]),
            sources={mode: _hash(path) for mode, path in files.items()},
        )
        print(deck, result["held"][deck]["scores"], flush=True)
    _write(folder / "joint-results.json", result)


def replay_selected(folder):
    """Replay the strongest complete rally in each arm with no actor or cameras."""
    files = [(deck, mode, folder / f"joint-{deck}-{mode}.json.gz") for deck in DECKS for mode in ARM]
    rows = []
    for deck, mode, path in files:
        doc = read_complete(path)
        i = max(range(len(doc["cards"])), key=lambda i: doc["rows"][i]["score"]["learner"])
        card, row = doc["cards"][i], doc["rows"][i]
        rows.append(dict(deck=deck, mode=mode, source_sha256=_hash(path),
                         **replay_case(doc["ready"], doc["protocol"], card, row)))
    _write(folder / "joint-audit.json", dict(rows=rows))


def replay_case(ready, protocol, card, row):
    with centered(ready):
        environment = arena(protocol=protocol)
        cad = arena(protocol=protocol, detail="cad")
        try:
            environment.launch(row["shot"], card=card)
            environment.partner.reset(seed=card["seed"])
            guard = ArmClearance(cad.model, tolerance=0.)
            shadow = mujoco.MjData(cad.model)
            environment.sample_clearance(True)
            environment._cad_reader = guard, shadow
            hits = record(environment, row, opening_side=1)
            if digest_json(environment.point_placements()) != digest_json(row["placements"]):
                raise ValueError("Physical opponent placements changed on action replay")
            if digest_json(audit(environment.events, opening_side=1)) != digest_json(row["score"]):
                raise ValueError("Referee scores changed on replay")
            clearance = environment.clearance_result()
            if clearance["samples"] != round(environment.data.time / environment.model.opt.timestep):
                raise ValueError("Missing physics step in CAD check")
            return dict(card=card, own=row["score"]["learner"], returns=row["score"]["returns"],
                        event_sha256=digest_json(environment.events), contacts=hits,
                        simulation_s=environment.data.time, launches=environment.launches,
                        clearance=clearance)
        finally:
            environment.close()
            cad.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "summary", "audit"))
    parser.add_argument("--folder", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "select":
        select(args.folder)
    elif args.phase == "summary":
        summarize(args.folder)
    else:
        replay_selected(args.folder)
