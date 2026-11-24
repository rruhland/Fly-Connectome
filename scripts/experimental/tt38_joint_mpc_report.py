"""Paired live outcomes and post-hoc forecast audits for direct joint receding search."""

import argparse
from collections import Counter
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_joint_deep_report import totals
from .tt37_joint_report import paired, read_complete, replay_case
from .tt38_joint_mpc import DT, OUT


def metrics(doc):
    result = totals(doc)
    planner, sources, ends = Counter(), Counter(), Counter()
    due, contacted = 0, 0
    for row in doc["rows"]:
        planner.update(row.get("planner", {}))
        physical = [e["time"] for e in row["events"] if e["kind"] == "contact" and e["side"] == 1]
        searches = row.get("searches", [])
        last = next(iter(row["placements"]["delivered"][-1:]), None)
        if last is None:
            ends["no_far_offer"] += 1
        elif last["near_landed"]:
            ends["after_legal_near"] += 1
        else:
            swipes = [s for s in searches if s["tick"] * DT > last["near_bounce"]["time"] - .05]
            kind = "contact_without_return" if last["near_contact"] is not None else "without_contact"
            source = "legal_nominal_found" if any(s["chosen"] for s in swipes) else "no_legal_nominal"
            ends[f"{source}_{kind}"] += 1
        for i, s in enumerate(searches):
            if s["chosen"] is None:
                continue
            sources[s["chosen"]] += 1
            if s["contact_delay_s"] > .08334:
                continue
            prediction = s["tick"] * DT + s["contact_delay_s"]
            if any(t["tick"] * DT < prediction for t in searches[i + 1:i + 2]):
                continue
            due += 1
            contacted += any(abs(t - prediction) <= .042 for t in physical)
    result.update(planner=dict(planner), accepted_sources=dict(sources),
                  imminent_predictions=due, actual_nearby_contacts=contacted, terminal_offer=dict(ends),
                  simulated_control_s=sum(len(row["actions"]) for row in doc["rows"]) * DT)
    return result


def selection(folder):
    paths = {k: folder / f"joint-mpc-development-{k}.json.gz" for k in ("raw", "live", "elastic")}
    docs = {k: read_complete(p) for k, p in paths.items()}
    if any(d["cards"] != docs["raw"]["cards"] for d in docs.values()):
        raise ValueError("Practice lineup changed between controls")
    stats = {k: metrics(v) for k, v in docs.items()}
    variant = max(("live", "elastic"), key=lambda k: (
        stats[k]["later"], stats[k]["own"], stats[k]["longest_own"], -stats[k]["cpu_s"]))
    selected = dict(variant="fragment" if variant == "live" else "elastic", stats=stats,
                    sources={k: _hash(p) for k, p in paths.items()},
                    selection="Max later legal returns, then total own, then largest run, then lower CPU on practice")
    _write(folder / "joint-mpc-choice.json", selected)
    print(variant, stats, flush=True)


def compare(folder, decks):
    choice = _read(folder / "joint-mpc-choice.json")
    result = dict(practice=choice, held={})
    for deck in decks:
        paths = {kind: folder / f"joint-mpc-{deck}-{kind}.json.gz" for kind in ("raw", "frozen", "live")}
        present = {k: read_complete(p) for k, p in paths.items() if p.is_file()}
        if "raw" not in present or not {"frozen", "live"} & present.keys():
            raise ValueError(f"Need both a raw control and a planner trace on {deck}")
        for name, doc in present.items():
            if (name != "raw" and doc["variant"] != choice["variant"]
                    or doc["source_sha256"] != present["raw"]["source_sha256"]
                    or doc["optical_sha256"] != present["raw"]["optical_sha256"]
                    or doc["protocol_sha256"] != present["raw"]["protocol_sha256"]):
                raise ValueError("Body search, priors or the shared optics changed")
            if name != "raw":
                epochs = [p["memory"] for p in doc["rows"]]
                if (epochs[0]["before"]["rows"] != 0
                        or any(a["after"] != b["before"] for a, b in zip(epochs, epochs[1:]))
                        or any(e["after"]["rows"] > 192 for e in epochs)
                        or (name == "frozen" and any(e["before"] != e["after"] for e in epochs))):
                    raise ValueError("Live joint-action credit crossed the frozen deck or memory boundary")
        entry = dict(stats={k: metrics(d) for k, d in present.items()}, comparisons={},
                     source_sha256={k: _hash(paths[k]) for k in present})
        for name, doc in present.items():
            if name != "raw":
                entry["comparisons"][name] = paired(present["raw"], doc)
        if "live" in present and "frozen" in present:
            entry["comparisons"]["live_against_frozen"] = paired(present["frozen"], present["live"])
        original_path = folder.parent / "tt37" / f"long-rgb-{deck}.json.gz"
        if original_path.exists():
            original = read_complete(original_path)
            count = len(present["raw"]["cards"])
            original = dict(original, rows=original["rows"][:count], cards=original["cards"][:count])
            if original["source_sha256"] == present["raw"]["source_sha256"]:
                entry["original_camera_reference"] = dict(
                    stats=metrics(original), source_sha256=_hash(original_path),
                    comparisons={name: paired(original, doc) for name, doc in present.items()},
                )
        result["held"][deck] = entry
        print(deck, entry["stats"], flush=True)
    _write(folder / "joint-mpc-results.json", result)


def audit(folder, decks):
    output = []
    for deck in decks:
        for mode in ("frozen", "live"):
            path = folder / f"joint-mpc-{deck}-{mode}.json.gz"
            if not path.exists():
                continue
            doc = read_complete(path)
            k = max(range(len(doc["rows"])), key=lambda i: doc["rows"][i]["score"]["learner"])
            if doc["rows"][k]["score"]["learner"]:
                output.append(dict(deck=deck, mode=mode, source_sha256=_hash(path),
                                   **replay_case(doc["ready"], "random", doc["cards"][k], doc["rows"][k])))
    _write(folder / "joint-mpc-audit.json", dict(rows=output))


def extend(folder):
    """Validate that the longer natural replay did not alter any of the scored prefix."""
    short_path = folder / "joint-mpc-development-elastic.json.gz"
    long_path = folder / "joint-mpc-extended.json.gz"
    short, long = read_complete(short_path), read_complete(long_path)
    count = len(long["rows"])
    if (short["cap"] != 36 or long["cap"] != 200 or short["cards"][:count] != long["cards"]
            or short["source_sha256"] != long["source_sha256"]
            or short["optical_sha256"] != long["optical_sha256"]):
        raise ValueError("The natural extended point changed the course or the learner prior")
    matches = []
    for i, (before, after) in enumerate(zip(short["rows"][:count], long["rows"], strict=True)):
        split = i == count - 1
        expected_events = (after["events"][:len(before["events"])] if split else after["events"])
        expected_actions = (after["actions"][:len(before["actions"])] if split else after["actions"])
        matches.append(digest_json(before["events"]) == digest_json(expected_events)
                       and digest_json(before["actions"]) == digest_json(expected_actions))
    if not all(matches) or short["rows"][count - 1]["score"]["returns"] != short["cap"]:
        raise ValueError("Autonomous extension diverged before the original referee cutoff")
    card, long_row = long["cards"][-1], long["rows"][-1]
    proof = replay_case(long["ready"], "random", card, long_row)
    _write(folder / "joint-mpc-extended-audit.json", dict(
        short_sha256=_hash(short_path), extended_sha256=_hash(long_path), exact_prefix=matches,
        earlier_point_count=count - 1, old_cap=short["cap"], new_cap=long["cap"],
        search=long_row["planner"], runtime_cpu_s=long_row["runtime_cpu_s"],
        found_credited=sum(str(s["chosen"]).startswith("credited") for s in long_row["searches"]),
        **proof,
    ))


def compare_guarded(folder, *, audit_best=False):
    paths = {kind: folder / f"joint-mpc-guarded-challenge-{kind}.json.gz"
             for kind in ("raw", "elastic", "guarded")}
    docs = {mode: read_complete(p) for mode, p in paths.items()}
    raw = docs["raw"]
    for mode, doc in docs.items():
        if (doc["source_sha256"] != raw["source_sha256"] or doc["optical_sha256"] != raw["optical_sha256"]
                or doc["protocol_sha256"] != raw["protocol_sha256"] or doc["cards"] != raw["cards"]):
            raise ValueError("The incumbent challenge is not actually matched")
        if mode != "raw" and (doc["mode"] != "live" or doc["variant"] != mode
                              or doc["rows"][0]["memory"]["before"]["rows"] != 0):
            raise ValueError("Guarded and elastic must start with the same empty live memory")
    d = dict(
        results={k: metrics(v) for k, v in docs.items()},
        against_raw={k: paired(raw, v) for k, v in docs.items() if k != "raw"},
        against_elastic=paired(docs["elastic"], docs["guarded"]),
        sources={mode: _hash(p) for mode, p in paths.items()},
    )
    _write(folder / "joint-mpc-guarded-results.json", d)
    if audit_best:
        doc = docs["guarded"]
        k = max(range(len(doc["cards"])), key=lambda i: doc["rows"][i]["score"]["learner"])
        proof = replay_case(doc["ready"], "random", doc["cards"][k], doc["rows"][k])
        _write(folder / "joint-mpc-guarded-audit.json", dict(
            source_sha256=_hash(paths["guarded"]), elastic_own=docs["elastic"]["rows"][k]["score"]["learner"],
            raw_own=raw["rows"][k]["score"]["learner"], **proof,
        ))


def within_point(folder):
    fixed_path = folder / "joint-mpc-within-frozen.json.gz"
    active_path = folder / "joint-mpc-held_openings-live.json.gz"
    f, a = read_complete(fixed_path), read_complete(active_path)
    cap = len(f["cards"])
    a = dict(a, rows=a["rows"][:cap], cards=a["cards"][:cap])
    result = paired(f, a)
    early_control, early_active = f["rows"][0], a["rows"][0]
    active, control = a["rows"][-1], f["rows"][-1]
    if (cap != 2 or f["variant"] != a["variant"] or f["source_sha256"] != a["source_sha256"]
            or f["optical_sha256"] != a["optical_sha256"]
            or early_control["actions"] != early_active["actions"]
            or early_control["events"] != early_active["events"]
            or active["memory"]["before"] != control["memory"]["before"]
            or active["memory"]["before"]["rows"] != 0):
        raise ValueError("Within-point comparison did not start from the exact same reward-empty state")
    changed = next((i for i, (x, y) in enumerate(zip(active["actions"], control["actions"])) if x != y), None)
    elapsed = changed * DT if changed is not None else None
    awarded = None if elapsed is None else sum(s["side"] == 1 and s["landing"] <= elapsed + 1e-9
                                                for s in active["score"]["shots"])
    if changed is not None and awarded == 0:
        raise ValueError("Controllers differed before any live reward on the target point")
    _write(folder / "joint-mpc-within-results.json", dict(
        stage="follow-up causal diagnostic chosen after observing the held result",
        card=a["cards"][-1], exact_earlier_point=True, equal_empty_memory=True,
        first_different_action=changed, simulation_s_at_difference=elapsed,
        already_awarded_actual_near=awarded, fixed=metrics(f), active=metrics(a), paired=result,
        sources=dict(active=_hash(active_path), fixed=_hash(fixed_path)),
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("selection", "comparison", "audit", "extension", "guarded", "within"))
    p.add_argument("--folder", type=Path, default=OUT)
    p.add_argument("--decks", nargs="+", default=("held_openings", "held_replies"))
    p.add_argument("--audit-best", action="store_true")
    args = p.parse_args()
    if args.phase == "selection":
        selection(args.folder)
    elif args.phase == "comparison":
        compare(args.folder, args.decks)
    elif args.phase == "audit":
        audit(args.folder, args.decks)
    elif args.phase == "guarded":
        compare_guarded(args.folder, audit_best=args.audit_best)
    elif args.phase == "within":
        within_point(args.folder)
    else:
        extend(args.folder)
