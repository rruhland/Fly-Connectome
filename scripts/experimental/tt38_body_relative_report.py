"""Compare unchanged-opponent body feedback trials and replay the strongest changed rally."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _write
from .tt37_joint_deep_report import totals
from .tt37_joint_report import paired, read_complete, replay_case


def condition(doc):
    fit = doc.get("candidate_source")
    return fit.get("augmentation", fit["body_coordinates"]) if fit else "absolute"


def report(control_path, conditions, output, *, replay=False):
    control = read_complete(control_path)
    control_name = condition(control)
    sources = {control_name: _hash(control_path)}
    scores = {control_name: totals(control)}
    pair = {}
    proof = {}
    for path in conditions:
        candidate = read_complete(path)
        for key in ("source_sha256", "plan_sha256", "deck", "cap", "ready"):
            if candidate[key] != control[key]:
                raise ValueError(f"Control mismatch at {key}: {path.name}")
        name = condition(candidate)
        if name in scores:
            raise ValueError(f"Duplicate condition {name}")
        sources[name] = _hash(path)
        pair[name] = paired(control, candidate)
        scores[name] = totals(candidate)
        if replay:
            better = [i for i, row in enumerate(pair[name]["rows"]) if row["near"][1] > row["near"][0]]
            if better:
                i = max(better, key=lambda i: candidate["rows"][i]["score"]["learner"])
                proof[name] = replay_case(candidate["ready"], "random", candidate["cards"][i], candidate["rows"][i])
        print(name, "own", scores[name]["own"], "first", scores[name]["first"],
              "later", scores[name]["later"], "best", scores[name]["longest_own"], flush=True)
    _write(output, dict(deck=control["deck"], source_sha256=control["source_sha256"],
                        tapes=sources, scores=scores, paired=pair, replay=proof))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("control", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("conditions", type=Path, nargs="+")
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    report(a.control, a.conditions, a.output, replay=a.replay)
