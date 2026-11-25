"""Learn next-return coverage from physically realized broad TT37 games."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height

ROOT = Path("artifacts/tt43_preparation")
TRAIN = [
    "tt42/learning-24",
    "tt39/credit-carry",
    "tt39/credit-novel",
    "tt39/commitment-replication-stock",
    "tt41/early-optics-carry",
]
EXCLUDE = {14000}
SCALE = np.array([.15, .16, .23, .16, .17, .16, .60])
SOLVE_L2 = 30.
EXTRA_COST = 4.2
MIN_GAIN = .17
MAX_STOCK_LOSS = .66
BASE = mpc.MPC


def design(x):
    """Near-linear flight cues plus axis-local penalties, without high-order interactions."""
    x = np.asarray(x)
    normalized = (x - [0., .63, .35, 0., 1.29, 1.185, 2.30]) / SCALE
    u = np.clip(normalized, -3, 3)
    return np.c_[
        u, np.abs(u), np.maximum(0., (1.96 - x[:, 6]) / .45),
        np.maximum(0., (.21 - x[:, 2]) / .18), np.maximum(0., (x[:, 2] - .50) / .20),
        np.maximum(0., (1.14 - x[:, 4]) / .24),
    ]


def ridge(x, y):
    t = design(x)
    x0 = t.mean(axis=0)
    y0 = np.mean(y, axis=0)
    t -= x0
    b = np.linalg.solve(t.T @ t + SOLVE_L2 * np.eye(t.shape[1]), t.T @ (y - y0))
    return x0, y0, b


def estimate(fit, x):
    x0, y0, b = fit
    return np.clip(y0 + (design(x) - x0) @ b, 0., 1.)


def gather(exclude):
    """Creditable labels pair actual answered shots to the last actually issued prediction."""
    rows = {}
    receipts = {}
    for n in TRAIN:
        path = Path(f"artifacts/{n}.json.gz")
        doc = _read(path)
        receipts[str(path)] = _hash(path)
        for card, row in zip(doc["cards"], doc["rows"]):
            if card["id"] in exclude:
                continue
            deliveries = row["placements"]["delivered"]
            eligible = [s for s in row["searches"] if s.get("target_errors")]
            for i, leg in enumerate(deliveries[:-1]):
                future = deliveries[i + 1]
                if not leg["near_landed"] or not future.get("receive_line"):
                    continue
                time = leg["near_contact"]["time"] / mpc.DT
                decisions = [s for s in eligible if 0 < time - s["tick"] < 8]
                if not decisions:
                    continue
                move = decisions[-1]
                if abs(time - move["tick"] - move["contact_delay_s"] / mpc.DT) > 4:
                    continue
                # Remove literal duplicates across archives, not multiple new trials on the same release.
                impact = leg["near_contact"]
                key = (card["id"], *np.round(impact["ball"] + impact["incoming"], 6),
                       *np.round(move["target_errors"], 5))
                h = projected_height(future["receive_line"])
                rows.setdefault(key, dict(id=card["id"], features=move["target_errors"],
                                           height=h, low=h < .99, missed=not future["near_landed"]))
    return list(rows.values()), receipts


def metrics(p, y):
    n = len(y)
    hi = np.argsort(p[:, 0], kind="stable")[-n // 5:]
    return dict(brier=np.mean((p - y) ** 2, axis=0).tolist(),
                low_count_in_riskiest_fifth=int(y[hi, 0].sum()),
                misses_in_riskiest_fifth=int(y[hi, 1].sum()),
                mean=np.mean(p, axis=0).tolist())


def fit_bank(output):
    excluded = {c["id"] for c in _read(course.PLAN)["assessment"]} | EXCLUDE
    rows, receipts = gather(excluded)
    g = np.array([r["id"] for r in rows])
    x = np.array([r["features"] for r in rows])
    y = np.array([[r["low"], r["missed"]] for r in rows], dtype=float)
    groups = np.random.default_rng(430302).permutation(np.unique(g))
    scores = {}
    for name in ("prevalence", "ridge"):
        p = np.empty_like(y)
        for group in np.array_split(groups, 7):
            test = np.isin(g, group)
            p[test] = y[~test].mean(axis=0) if name == "prevalence" else estimate(
                ridge(x[~test], y[~test]), x[test]
            )
        scores[name] = metrics(p, y)
    fit = ridge(x, y)
    _write(output, dict(
        role="predict risk of a low or missed next delivery after one actually issued stroke",
        receipts=receipts, excluded=sorted(excluded), rows=rows, fit=[v.tolist() for v in fit],
        validation=scores, target_means=y.mean(axis=0).tolist(), releases=len(groups),
        groups=groups.tolist(), cap_censoring="only a subsequently delivered ball with a measured arrival",
        code_sha256=_hash(Path(__file__)), plan_sha256=_hash(course.PLAN),
        parameters=dict(extra_cost=EXTRA_COST, minimum_gain=MIN_GAIN,
                        maximum_stock_loss=MAX_STOCK_LOSS, ridge_l2=SOLVE_L2),
    ))
    print(len(y), "offers", len(groups), "releases", scores, flush=True)


class Coverage:
    def __init__(self, receipt):
        self.fit = tuple(map(np.asarray, receipt["fit"]))
        rows = np.asarray([r["features"] for r in receipt["rows"]])
        self.tree = cKDTree(rows / SCALE)

    def risk(self, x):
        d, _ = self.tree.query(x / SCALE, k=6)
        p = estimate(self.fit, x)
        return (.60 * p[:, 0] + .40 * p[:, 1]), d[:, -1] < 1.85


def choice(agreement, quality, risk, support):
    """Keep the ordinary winner unless an equally robust supported alternative is materially safer."""
    k = int(np.argmin(quality))
    better = EXTRA_COST * (risk[k] - risk) - np.maximum(0., quality - quality[k])
    candidates = ((agreement >= agreement[k]) & (agreement >= 1) & support
                  & (quality < quality[k] + MAX_STOCK_LOSS) & (better > MIN_GAIN))
    if support[k] and candidates.any():
        return int(np.argmax(np.where(candidates, better, -np.inf))), k
    return k, k


class Actor(BASE):
    def __init__(self, *args, coverage=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage = coverage

    def begin(self):
        super().begin()
        self.reranks = []

    def search(self, tick, x, q, reference, base):
        self.metrics = {}
        self.in_rank = False
        self.decision_tick = tick
        return super().search(tick, x, q, reference, base)

    def trial(self, goals, *, long=False, traces=False):
        report = super().trial(goals, long=long, traces=traces)
        if long and not self.in_rank:
            for g, yes, y in zip(goals, report[1], report[3], strict=True):
                if yes:
                    self.metrics[np.asarray(g).tobytes()] = y[3:10].copy()
        return report

    def robust(self, goals, rank, nominal):
        self.in_rank = True
        try:
            agree, quality = self.uncertainty(goals, nominal)
        finally:
            self.in_rank = False
        x = np.asarray([self.metrics[np.asarray(g).tobytes()] for g in goals])
        risk, support = self.coverage.risk(x)
        i, stock = choice(agree, quality, risk, support)
        if i != stock:
            self.reranks.append(dict(tick=self.decision_tick, prior=int(rank[stock]),
                                     selected=int(rank[i]), original=x[stock].tolist(),
                                     replacement=x[i].tolist(), quality=quality[[stock, i]].tolist(),
                                     risks=risk[[stock, i]].tolist(),
                                     robustness=agree[[stock, i]].tolist()))
        return int(rank[i]), int(agree[i]), float(quality[i])


def run(args):
    baseline = _read(args.control)
    bank = _read(args.bank)
    ids = {c["id"] for c in baseline["cards"]}
    if ids.intersection(r["id"] for r in bank["rows"]):
        raise ValueError("Coverage estimator includes an evaluation release")
    if (baseline["mode"] != "frozen" or baseline["carry_sha256"] is not None
            or baseline["variant"] != "elastic" or baseline["source_sha256"] != _hash(mpc.BANK)
            or baseline["optical_sha256"] != _hash(course.OPTIC) or baseline["cap"] < 200):
        raise ValueError("Expected the frozen stock economic comparison")
    cards = baseline["cards"][: len(baseline["rows"])]
    if cards != _read(course.PLAN)["assessment"][:len(cards)]:
        raise ValueError("The frozen control does not follow the broad assessment roster")
    ready = baseline["ready"]
    receipt = dict(
        role="matched native broad-random frozen points, only final rank changes",
        cards=cards, cap=baseline["cap"], ready=ready, budget=baseline["budget"],
        baseline_sha256=_hash(args.control), ranker_sha256=_hash(args.bank),
        code_sha256=_hash(Path(__file__)), rows=[], complete=False,
        source_sha256=_hash(mpc.BANK), optical_sha256=_hash(course.OPTIC),
    )
    if args.output.exists():
        old = _read(args.output)
        if any(old[k] != v for k, v in receipt.items() if k not in ("rows", "complete")):
            raise ValueError("Incompatible continuation")
        receipt = old
    with np.load(mpc.BANK) as data:
        raw = mpc.DeepField(data, "local")
    with mpc.centered(ready):
        arena = mpc.arena()
        resources = mpc.resources(arena)
        with patch.object(mpc, "Fragments", lambda file, engine: Carried(
                file, engine, practice=[course.PRACTICE], held=ids)):
            model = Actor(mpc.BANK, raw, variant="elastic", budget=baseline["budget"],
                          coverage=Coverage(bank))
        try:
            end = len(cards) if args.limit is None else min(args.limit, len(cards))
            for i in range(len(receipt["rows"]), end):
                card, stock = cards[i], baseline["rows"][i]
                if stock["memory"]["before"] != model.priors.digest():
                    raise ValueError("Matched stock prior memory was not restored")
                model.serial = i
                row = mpc.play(arena, *resources, ready, card, stock["shot"], raw, model,
                               course.OPTIC, "frozen", baseline["cap"], 10)
                row["reranks"] = model.reranks
                first = next((k for k,(a,b) in enumerate(zip(stock["actions"], row["actions"]))
                              if a != b), None)
                if not model.reranks:
                    if row["actions"] != stock["actions"] or digest_json(row["events"]) != digest_json(stock["events"]):
                        raise ValueError("The unmodified path did not reproduce stock")
                elif first is not None and first < model.reranks[0]["tick"]:
                    raise ValueError("Commands changed before the first ranking intervention")
                for k, requested in enumerate(row["placements"]["requested"]):
                    if list(course.draw(card, k, "random", None)[0]) != requested["target"]:
                        raise RuntimeError("The independent TT37 placement changed")
                row["stock_learner"] = stock["score"]["learner"]
                row["first_changed_action"] = first
                receipt["rows"].append(row)
                receipt["complete"] = len(receipt["rows"]) == len(cards)
                _write(args.output, receipt)
                print(i+1, card["id"], stock["score"]["learner"],
                      row["score"]["learner"], len(row["reranks"]), first, flush=True)
        finally:
            model.close()
            resources[0].close()
            arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "run"))
    parser.add_argument("--bank", type=Path, default=ROOT / "outgoing-risk.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outgoing-coverage-live.json.gz")
    parser.add_argument("--control", type=Path, default=course.ROOT / "no-more-practice.json.gz")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "fit":
        fit_bank(args.bank)
    else:
        run(args)
