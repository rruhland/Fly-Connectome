"""Paired live outcomes and post-hoc forecast audits for direct joint receding search."""

import argparse
from collections import Counter
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
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
                  imminent_predictions=due, actual_nearby_contacts=contacted, terminal_offer=dict(ends))
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("selection", "comparison", "audit"))
    p.add_argument("--folder", type=Path, default=OUT)
    p.add_argument("--decks", nargs="+", default=("held_openings", "held_replies"))
    args = p.parse_args()
    if args.phase == "selection":
        selection(args.folder)
    elif args.phase == "comparison":
        compare(args.folder, args.decks)
    else:
        audit(args.folder, args.decks)
