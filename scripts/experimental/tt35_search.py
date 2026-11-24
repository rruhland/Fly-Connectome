"""Fit five shared joint biases from paired on-court rewards, then freeze them."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_curriculum_summary import compare_delivery, first_deliveries
from .tt31_ready import centered
from .tt33_joint import point, resources, verify


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
OUTPUT = ARTIFACTS / "tt35"
BASE = ARTIFACTS / "tt33" / "joint-raw.json.gz"
PRACTICE = tuple(ARTIFACTS / "tt31" / f"{name}.json.gz"
                 for name in ("recovery-full", "confirm-recovery", "recovery-periodic"))
HELD = {
    "novel": (
        ARTIFACTS / "tt34" / "body-novel-plan.json.gz",
        ARTIFACTS / "tt34" / "body-novel-stock.json.gz",
    ),
    "fresh": (
        ARTIFACTS / "tt32" / "fresh-plan.json.gz",
        ARTIFACTS / "tt33" / "joint-fresh-raw.json.gz",
    ),
}
STEPS = (.075, .065, .055, .045)
MAX_BIAS = .16
SEED = 350883
CAP = 16


def roster(seed):
    rng = np.random.default_rng(seed)
    full, confirmation, periodic = (_read(path)["cards"] for path in PRACTICE)
    groups = [rng.permutation(n) for n in (len(full), len(confirmation), len(periodic))]
    batches = []
    for i in range(len(STEPS)):
        cards = [full[j] for j in groups[0][2 * i:2 * i + 2]]
        cards.extend((confirmation[groups[1][i]], periodic[groups[2][i]]))
        batches.append([cards[j] for j in rng.permutation(len(cards))])
    fit_ids = {c["id"] for group in batches for c in group}
    holdout_ids = {c["id"] for plan, _ in HELD.values() for c in _read(plan)["cards"]}
    if fit_ids & holdout_ids:
        raise ValueError("A release appears in practice and a final assessment")
    return batches


def adjusted(base, bias):
    if not np.any(bias):
        return base
    weights = np.array(base["weights"])
    weights[:, :5] += bias
    return dict(base, weights=weights.tolist())


@contextmanager
def station(feeds, base):
    shots = measured_launches(feeds)
    with centered(base["ready"]):
        arena = replay_arena([])
        try:
            stereo, rig, stride, lower, upper = resources(arena)
            try:
                def play(card, model):
                    arena.cards = iter([card])
                    return point(arena, stereo, rig, model, stride, lower, upper,
                                 shots[card["id"]], card, cap=CAP)

                yield play
            finally:
                stereo.close()
        finally:
            arena.close()


def objectives(rows):
    legal = [r["score"]["learner"] for r in rows]
    return dict(legal=sum(legal), later=sum(max(0, count - 1) for count in legal),
                alternating=sum(r["score"]["returns"] for r in rows))


def preference(scores):
    return scores["legal"] + scores["later"]


def fit(feeds, output):
    model = _read(BASE)
    base_hash = _hash(BASE)
    batches = roster(SEED)
    ordered = np.random.default_rng(SEED + 1)
    bias = np.zeros(5)
    report = dict(
        schema="tt35-search-1", base_sha256=base_hash, feed_sha256=_hash(feeds),
        sources={p.name: _hash(p) for p in PRACTICE}, cards=batches,
        seed=SEED, step_rad=STEPS, cap=CAP, rounds=[],
    )
    practice_path = output / "search-practice.json.gz"
    with station(feeds, model) as play:
        for step, cards in zip(STEPS, batches, strict=True):
            direction = ordered.choice((-1., 1.), size=len(bias))
            proposals = [bias.copy(), np.clip(bias + step * direction, -MAX_BIAS, MAX_BIAS),
                         np.clip(bias - step * direction, -MAX_BIAS, MAX_BIAS)]
            candidates = [adjusted(model, p) for p in proposals]
            points = [[] for candidate in candidates]
            orders = []
            for card in cards:
                order = ordered.permutation(len(candidates)).tolist()
                orders.append(order)
                for index in order:
                    points[index].append(play(card, candidates[index]))
            scores = [objectives(trial) for trial in points]
            accepted = 0
            best = max(range(len(candidates)),
                       key=lambda i: (preference(scores[i]), -np.linalg.norm(proposals[i])))
            if preference(scores[best]) > preference(scores[0]):
                accepted = best
            bias = proposals[accepted]
            report["rounds"].append(dict(
                cards=cards, trial_order=orders, direction=direction.tolist(), candidates_rad=[p.tolist() for p in proposals],
                scores=scores, winner=accepted, retained_rad=bias.tolist(), rows=points,
            ))
            _write(practice_path, report)
            print(len(report["rounds"]), scores, "accepted", accepted, bias.tolist(), flush=True)
    _write(output / "search-checkpoint.json", dict(
        base_sha256=base_hash, practice_sha256=_hash(practice_path), bias_rad=bias.tolist(),
        practice_release_ids=sorted({c["id"] for cards in batches for c in cards}),
        objective="legal_returns_plus_later_legal_returns",
        episodes=sum(len(rows) for trial in report["rounds"] for rows in trial["rows"]),
        cpu_s=sum(row["runtime_cpu_s"] for trial in report["rounds"] for rows in trial["rows"] for row in rows),
    ))


def assess(feeds, output, cohort):
    checkpoint_path = output / "search-checkpoint.json"
    checkpoint = _read(checkpoint_path)
    if checkpoint["base_sha256"] != _hash(BASE) or checkpoint["practice_sha256"] != _hash(output / "search-practice.json.gz"):
        raise ValueError("Does not match the completed frozen checkpoint")
    base = _read(BASE)
    model = adjusted(base, np.asarray(checkpoint["bias_rad"]))
    plan_path, control_path = HELD[cohort]
    plan = _read(plan_path)
    baseline = _read(control_path)
    cards = plan["cards"]
    if cards != baseline["cards"] or _hash(feeds) != plan["feed_sha256"]:
        raise ValueError("The fixed control differs from the requested cohort")
    if {c["id"] for c in cards}.intersection(checkpoint["practice_release_ids"]):
        raise ValueError("Final assessment overlaps practice releases")
    report = dict(
        cards=cards, ready=base["ready"], opening_side=1, cohort=cohort, rows=[],
        checkpoint_sha256=_hash(checkpoint_path), plan_sha256=_hash(plan_path),
        control_sha256=_hash(control_path), bias_rad=checkpoint["bias_rad"], cap=CAP,
    )
    path = output / f"search-held-{cohort}.json.gz"
    with station(feeds, base) as play:
        for card in cards:
            row = play(card, model)
            report["rows"].append(row)
            _write(path, report)
            print(cohort, card["id"], row["score"]["learner"], row["score"]["returns"], flush=True)


def first_contact_counts(rows):
    first = [row["placements"]["delivered"][0] for row in rows if row["placements"]["delivered"]]
    return dict(delivered=len(first), touched=sum(p["near_contact"] is not None for p in first),
                landed=sum(p["near_landed"] for p in first))


def summary(output):
    checkpoint_path = output / "search-checkpoint.json"
    checkpoint = _read(checkpoint_path)
    practice_path = output / "search-practice.json.gz"
    if checkpoint["practice_sha256"] != _hash(practice_path):
        raise ValueError("Practice and the learned checkpoint differ")
    practice = _read(practice_path)
    all_scores = {}
    for cohort, (_, baseline) in HELD.items():
        fixed = _read(baseline)
        held = _read(output / f"search-held-{cohort}.json.gz")
        if len(held["rows"]) != len(fixed["rows"]) or held["cards"] != fixed["cards"]:
            raise ValueError("Held result and control differ in releases")
        if held["control_sha256"] != _hash(baseline) or held["checkpoint_sha256"] != _hash(checkpoint_path):
            raise ValueError("The held run used a different checkpoint or control")
        changed = [row["score"]["learner"] - control["score"]["learner"]
                   for row, control in zip(held["rows"], fixed["rows"], strict=True)]
        all_scores[cohort] = dict(
            fixed=objectives(fixed["rows"]), learned=objectives(held["rows"]),
            fixed_first_contacts=first_contact_counts(fixed["rows"]),
            learned_first_contacts=first_contact_counts(held["rows"]),
            first_delivery=compare_delivery(first_deliveries(fixed["rows"]), first_deliveries(held["rows"])),
            wins=sum(x > 0 for x in changed), ties=changed.count(0), losses=sum(x < 0 for x in changed),
            identical_actions=sum(row["actions"] == control["actions"]
                                  for row, control in zip(held["rows"], fixed["rows"], strict=True)),
            active_ticks=sum(row["activity"]["active_steps"] for row in held["rows"]),
            max_measured_rate_rad_s=max(max(row["motor_audit"]["max_measured_rate_rad_s"]) for row in held["rows"]),
            per_point=dict(zip((c["id"] for c in held["cards"]), changed, strict=True)),
            cpu_s=sum(r["runtime_cpu_s"] for r in held["rows"]),
        )
    practice_matches = [
        [compare_delivery(first_deliveries(t["rows"][0]), first_deliveries(candidate))
         for candidate in t["rows"][1:]]
        for t in practice["rounds"]
    ]
    _write(output / "search-results.json", dict(
        practice=dict(rounds=len(practice["rounds"]), accepted=sum(t["winner"] > 0 for t in practice["rounds"]),
                      scores=[t["scores"] for t in practice["rounds"]], first_deliveries=practice_matches,
                      cpu_s=checkpoint["cpu_s"],
                      episodes=checkpoint["episodes"], retained_rad=checkpoint["bias_rad"]),
        held=all_scores,
    ))
    print(all_scores, flush=True)


def audit_positive(source, output):
    doc = _read(source)
    positive = [(c, r) for c, r in zip(doc["cards"], doc["rows"], strict=True) if r["score"]["learner"]]
    if not positive:
        _write(output, dict(source_sha256=_hash(source), points=[]))
        return
    with TemporaryDirectory(prefix="tt35-search-") as tmp:
        selected = Path(tmp) / "positive.json.gz"
        verified = Path(tmp) / "audit.json"
        cards, rows = map(list, zip(*positive, strict=True))
        _write(selected, dict(doc, cards=cards, rows=rows))
        verify(selected, verified, cad=True)
        checked = _read(verified)
    _write(output, dict(checked, source_sha256=_hash(source)))



def audit_practice(directory):
    """Replay both commands on the practice card with the largest accepted repeat gain."""
    source = directory / "search-practice.json.gz"
    practice = _read(source)
    best = None
    for number, trial in enumerate(practice["rounds"], 1):
        winner = trial["winner"]
        if winner == 0:
            continue
        for card, before, after in zip(trial["cards"], trial["rows"][0], trial["rows"][winner], strict=True):
            gain = after["score"]["learner"] - before["score"]["learner"]
            if gain > 0 and after["score"]["learner"] > 1 and (best is None or gain > best[0]):
                best = (gain, number, card, before, after)
    output = directory / "search-practice-repeat-audit.json"
    if best is None:
        _write(output, dict(source_sha256=_hash(source), points=[]))
        return
    _, number, card, before, after = best
    with TemporaryDirectory(prefix="tt35-search-") as tmp:
        selected = Path(tmp) / "matched.json.gz"
        verified = Path(tmp) / "audit.json"
        _write(selected, dict(ready=_read(BASE)["ready"], cards=[card, card], rows=[before, after]))
        verify(selected, verified, cad=True)
        result = _read(verified)
    result["points"][0]["condition"] = "incumbent"
    result["points"][1]["condition"] = "accepted"
    _write(output, dict(result, source_sha256=_hash(source), round=number, card=card))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    fit_parser = phases.add_parser("fit")
    fit_parser.add_argument("feeds", type=Path)
    fit_parser.add_argument("--output", type=Path, default=OUTPUT)
    held_parser = phases.add_parser("assess")
    held_parser.add_argument("feeds", type=Path)
    held_parser.add_argument("cohort", choices=HELD)
    held_parser.add_argument("--output", type=Path, default=OUTPUT)
    summary_parser = phases.add_parser("summary")
    summary_parser.add_argument("--output", type=Path, default=OUTPUT)
    replay_parser = phases.add_parser("audit")
    replay_parser.add_argument("source", type=Path)
    replay_parser.add_argument("output", type=Path)
    practice_replay = phases.add_parser("audit-practice")
    practice_replay.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "fit":
        args.output.mkdir(parents=True, exist_ok=True)
        fit(args.feeds, args.output)
    elif args.phase == "assess":
        assess(args.feeds, args.output, args.cohort)
    elif args.phase == "summary":
        summary(args.output)
    elif args.phase == "audit":
        audit_positive(args.source, args.output)
    else:
        audit_practice(args.output)


if __name__ == "__main__":
    main()
