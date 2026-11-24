"""Freeze the deep pupil on familiar play, then summarize the held physical comparisons."""

import argparse
from collections import Counter
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_capture import OUTPUT
from .tt37_joint_deep import MODES
from .tt37_joint_report import paired, read_complete, replay_case


DECKS = ("held_openings", "fresh_replies")


def totals(doc):
    near = [r["score"]["learner"] for r in doc["rows"]]
    alternating = [r["score"]["returns"] for r in doc["rows"]]
    lanes = {part: Counter() for part in ("opening_offers", "opening_contacts", "opening_legal",
                                          "later_offers", "later_contacts", "later_legal")}
    for row in doc["rows"]:
        for p in row["placements"]["delivered"]:
            part = "opening" if p["leg"] == 0 else "later"
            lane = p["observed_lane"] or "unmeasured"
            lanes[part + "_offers"][lane] += 1
            if p["near_contact"] is not None:
                lanes[part + "_contacts"][lane] += 1
            if p["near_landed"]:
                lanes[part + "_legal"][lane] += 1
    measured_legal = sum(sum(lanes[part].values()) for part in ("opening_legal", "later_legal"))
    if measured_legal != sum(near):
        raise ValueError(f"Placement bookkeeping and legal referee differ: {measured_legal} != {sum(near)}")
    return dict(points=len(near), first=sum(v > 0 for v in near), own=sum(near),
                later=sum(max(0, v - 1) for v in near), longest_own=max(near, default=0),
                longest_alternating=max(alternating, default=0), alternating=sum(alternating),
                points_with_reply=sum(v > 1 for v in near), lanes=lanes,
                field_ticks=sum(r["field_ticks"] for r in doc["rows"]),
                cpu_s=sum(r["runtime_cpu_s"] for r in doc["rows"]))


def read_deck(folder, deck, mode, sha256):
    path = folder / f"joint-deep-{deck}-{mode}.json.gz"
    doc = read_complete(path)
    if doc["source_sha256"] != sha256 or doc["deck"] != deck or doc["mode"] != mode:
        raise ValueError(f"Unexpected frozen source or comparison condition in {path.name}")
    return path, doc


def select(folder):
    receipt = _read(folder / "joint-deep-observations.json")
    deck = {m: read_deck(folder, "familiar", m, receipt["bank_sha256"]) for m in MODES}
    cards = [r["card"] for r in receipt["practice"]]
    if any(doc["cards"] != cards or doc["ready"] != receipt["ready"] for _, doc in deck.values()):
        raise ValueError("Calibration must use the verified training contexts and posture")
    stats = {m: totals(doc) for m, (_, doc) in deck.items()}
    mode = max(MODES, key=lambda m: (stats[m]["own"] + stats[m]["later"], stats[m]["own"], -MODES.index(m)))
    practice = dict(mode=mode, stats=stats, mentor_near=sum(r["mentor_returns"] for r in receipt["practice"]),
                    sources={m: _hash(path) for m, (path, _) in deck.items()},
                    bank_sha256=receipt["bank_sha256"],
                    rule="Familiar training-context legal near plus later, then total near, then local")
    _write(folder / "joint-deep-choice.json", practice)
    print(practice, flush=True)


def summary(folder):
    selection = _read(folder / "joint-deep-choice.json")
    results = {}
    for deck in DECKS:
        active, baseline = (read_deck(folder, deck, mode, selection["bank_sha256"])
                            for mode in (selection["mode"], "ready"))
        match = paired(baseline[1], active[1])
        results[deck] = dict(pupil=totals(active[1]), ready=totals(baseline[1]), paired=match,
                             source_sha256={doc["mode"]: _hash(path) for path, doc in (active, baseline)})
        print(deck, results[deck]["pupil"], results[deck]["ready"], flush=True)
    left_path = folder / "joint-held_openings-reach.json.gz"
    if left_path.is_file():
        left = read_complete(left_path)
        _, pupil = read_deck(folder, "held_openings", selection["mode"], selection["bank_sha256"])
        matchup = results["held_openings"]
        matchup["left_ready_reference"] = dict(scores=totals(left), paired=paired(left, pupil),
                                                source_sha256=_hash(left_path))
    _write(folder / "joint-deep-results.json", dict(calibration=selection, held=results))


def audit(folder):
    choice = _read(folder / "joint-deep-choice.json")
    cases = []
    for deck in ("familiar", *DECKS):
        path, doc = read_deck(folder, deck, choice["mode"], choice["bank_sha256"])
        i = max(range(len(doc["rows"])), key=lambda i: doc["rows"][i]["score"]["learner"])
        cases.append(dict(deck=deck, source_sha256=_hash(path),
                          **replay_case(doc["ready"], "random", doc["cards"][i], doc["rows"][i])))
    _write(folder / "joint-deep-audit.json", dict(rows=cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "summary", "audit"))
    parser.add_argument("--folder", type=Path, default=OUTPUT)
    args = parser.parse_args()
    {"select": select, "summary": summary, "audit": audit}[args.phase](args.folder)
