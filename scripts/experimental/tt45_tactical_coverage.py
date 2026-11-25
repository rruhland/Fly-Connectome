"""Separate a coverable next reply from artifacts of incoming difficulty and random aim."""

import argparse
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_opponent as live
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt39_opportunity_map import projected_height
from .tt42_learning_course import PLAN
from .tt44_paired_credit import State

ROOT = Path("artifacts/tt45_tactics")
SOURCES = (
    "tt42/learning-48", "tt42/diverse-learning-48", "tt42/graded-practice",
    "tt42/uniform-practice", "tt37/integrated-practice", "tt37/mix-rollin",
    "tt39/credit-carry", "tt39/credit-reset", "tt39/credit-novel",
    "tt39/credit-observed", "tt39/commitment-replication-stock",
    "tt39/commitment-replication-carry", "tt39/commitment-replication-commit",
    "tt41/early-optics-carry",
    *(f"tt39/endurance-{i:02d}" for i in range(1, 9)),
)
TARGET_LO = np.array([-.28, -1.34, 1.01, 3.15])
TARGET_HI = np.array([.28, -1.22, 1.11, 3.98])
SCALES = {
    "incoming": ([0., 1.02, 0., -3.75, 1.05], [.21, .065, .45, .60, .65]),
    "request": ((TARGET_HI + TARGET_LO) / 2, (TARGET_HI - TARGET_LO) / np.sqrt(12)),
    "outgoing": ([0., .65, 2.65, 3.20], [.17, .20, .64, .95]),
}
THRESHOLD = .99
HARD = 1.018


def arc_shape(p, v):
    """Common observable flight coordinates, computed identically for trials and scored departures."""
    p, v = map(lambda x: np.asarray(x, dtype=float), (p, v))
    with np.errstate(invalid="ignore"):
        damp = .133 * np.clip(np.linalg.norm(v, axis=-1), 2.5, 12.)
        lo, hi = np.zeros_like(damp), np.full_like(damp, 1.8)
        for _ in range(14):
            mid = .5 * (lo + hi)
            travel = -np.expm1(-damp * mid) / damp
            z = p[..., 2] + travel * v[..., 2] - 9.81 * (mid - travel) / damp
            lo, hi = np.where(z > .78, mid, lo), np.where(z > .78, hi, mid)
        t = .5 * (lo + hi)
        travel = -np.expm1(-damp * t) / damp
        loc = p[..., :2] + travel[..., None] * v[..., :2]
        vy = np.exp(-damp * t) * v[..., 1]
        vz = -np.exp(-damp * t) * v[..., 2] + 9.81 * travel
    okay = ((p[..., 2] > .775) & (v[..., 1] > .7) & (t < 1.79)
            & (loc[..., 1] > -.1) & (loc[..., 1] < 2.))
    return np.where(okay[..., None], np.concatenate((loc, vy[..., None], vz[..., None]), axis=-1), np.nan)


def source_rows():
    held = {c["id"] for c in _read(PLAN)["assessment"]} | {14000}
    result, receipts = {}, {}
    for key in SOURCES:
        path = Path("artifacts") / (key + ".json.gz")
        doc = _read(path)
        receipts[key] = _hash(path)
        for i, (card, row) in enumerate(zip(doc["cards"], doc["rows"], strict=True)):
            if card["id"] in held or card.get("profile") != "reach":
                continue
            placements = row["placements"]
            if any(r["draw"] != "random" for r in placements["requested"]):
                continue
            deliveries = placements["delivered"]
            by_own = [e for e in row["events"] if e.get("collider") == "learner_racket"]
            own_lands = [e for e in row["events"] if e["kind"] == "legal_return" and e["side"] == 1][1:]
            awarded = [v for v in deliveries if v["near_landed"]]
            if len(own_lands) != len(awarded):
                raise ValueError("Actual returned bounces and incoming deliveries differ")
            done = row["events"][-1].get("kind") == "point"
            for offer, land in zip(awarded, own_lands, strict=True):
                j = offer["leg"]
                current, contact = offer["receive_line"], offer["near_contact"]
                if current is None:
                    continue
                dep = next((e for e in by_own if contact["time"] < e["time"] < land["time"]), None)
                if dep is None:
                    continue
                observed = arc_shape(dep["ball"], dep["velocity"])
                if not np.isfinite(observed).all():
                    continue
                aft = deliveries[j + 1] if j + 1 < len(deliveries) else None
                if aft is None and not done:
                    continue
                requests = placements["requested"]
                if j + 1 >= len(requests):
                    continue
                req = requests[j + 1]["target"]
                # Commands, scorer and draw must describe one physically realized cycle.
                key0 = (card["id"], j, card["seed"], card["reply_seed"],
                        digest_json((dep["ball"], dep["velocity"], dep["spin"], req)))
                incident = np.r_[current["xyz"][0], projected_height(current), current["velocity"]]
                if key0 not in result:
                    h = projected_height(aft["receive_line"]) if aft and aft["receive_line"] else None
                    current_key = digest_json((offer["far_contact"], offer["near_bounce"]))
                    result[key0] = dict(
                        source=key, row=i, id=card["id"], card=card, leg=j, incoming=incident.tolist(),
                        request=req, outgoing=observed.tolist(), height=h, answered=aft is not None,
                        current_key=current_key, asked=bool(aft and aft["receive_line"]),
                        actual_landing=land["location"], impact_tick=contact["step"], departure_tick=dep["step"],
                        departure_sha256=digest_json(dep), separation_s=dep["time"]
                    )
    return list(result.values()), receipts


def standardized(rows, name):
    center, scale = SCALES[name]
    return np.clip((np.array([r[name] for r in rows]) - center) / scale, -4., 4.)


def features(rows, blocks, interactions=False):
    w = {k: standardized(rows, k) for k in blocks}
    parts = [f for x in w.values() for f in (x, np.abs(x), np.minimum(6., x * x))]
    if "outgoing" in w:
        x = w["outgoing"]
        # Two curved directions of the arc, plus divergence at the table edges.
        parts.extend([x[:, [1, 1, 2]] * x[:, [2, 3, 3]], np.abs(x[:, :1]) * x[:, 1:]])
    if interactions:
        for key in ("incoming", "request"):
            if key in w:
                x, z = w["outgoing"], w[key]
                parts.append(.6 * (x[:, :, None] * z[:, None, :]).reshape(len(rows), -1))
    return np.concatenate(parts, axis=1)


def ridge(x, y, damping):
    mu, cy = np.mean(x, axis=0), np.mean(y, axis=0)
    c = x - mu
    beta = np.linalg.solve(c.T @ c + damping * np.eye(c.shape[1]), c.T @ (y - cy))
    return mu, cy, beta


def predict(fit, x):
    mu, cy, beta = fit
    return cy + (x - mu) @ beta


def grouped(rows, labels):
    ids = np.array([r["id"] for r in rows])
    group = np.random.default_rng(450621).permutation(np.unique(ids))
    reports = {}
    for key, blocks in (
        ("prior", ()), ("incoming", ("incoming",)), ("request", ("request",)),
        ("incoming_request", ("incoming", "request")), ("outgoing", ("outgoing",)),
        ("incoming_outgoing", ("incoming", "outgoing")),
        ("all", ("incoming", "request", "outgoing")),
        ("all_interactions", ("incoming", "request", "outgoing")),
    ):
        x = features(rows, blocks, key == "all_interactions") if blocks else np.zeros((len(rows), 1))
        out = np.empty_like(labels)
        for g in np.array_split(group, 8):
            use = np.isin(ids, g)
            out[use] = predict(ridge(x[~use], labels[~use], 65.), x[use])
        out[:, 1:] = np.clip(out[:, 1:], 0, 1)
        top = np.argsort(out[:, 1])[-max(1, len(rows) // 5):]
        hard = np.array([r["incoming"][1] < HARD for r in rows])
        reports[key] = dict(rmse_m=float(np.mean((out[:, 0] - labels[:, 0])**2)**.5),
                            low_brier=float(np.mean((out[:, 1] - labels[:, 1])**2)),
                            hard_low_brier=float(np.mean((out[hard, 1] - labels[hard, 1])**2)),
                            top_fifth_lows=int(labels[top, 1].sum()))
    return reports


def match_plan(rows, score, ids):
    """Choose actual alternative shots on identical original incoming events, without their follow-up outcomes."""
    groups = defaultdict(list)
    for j, r in enumerate(rows):
        if r["id"] in ids:
            groups[(r["id"], r["leg"], r["current_key"])].append(j)
    potential = {}
    for (ident, leg, k), g in groups.items():
        if len(g) < 2 or leg > 7:
            continue
        low, high = min(g, key=lambda j: score[j]), max(g, key=lambda j: score[j])
        gap = score[high] - score[low]
        # Mix equally among currently hard and ordinary inbound releases when there are enough.
        if gap > .02 and (ident not in potential or potential[ident][0] < gap):
            potential[ident] = gap, low, high
    cases = []
    used = set()
    for hard in (True, False):
        for ident, (gap, low, high) in sorted(potential.items(), key=lambda item: -item[1][0]):
            if (rows[low]["incoming"][1] < HARD) != hard or ident in used:
                continue
            cases.append(dict(id=ident, hard=hard, low_risk=rows[low], high_risk=rows[high],
                              scores=[float(score[low]), float(score[high])]))
            used.add(ident)
            if sum(c["hard"] == hard for c in cases) == 4:
                break
    return cases


def fit(output):
    rows, receipts = source_rows()
    train = [r for r in rows if r["height"] is not None]
    labels = np.array([[r["height"], r["height"] < THRESHOLD] for r in train])
    releases = np.random.default_rng(450671).permutation(sorted({r["id"] for r in rows}))
    # Reserve every fourth release for causal matched alternatives, including the draw matrix.
    drawn = set(releases[::4])
    base = [r for r in train if r["id"] not in drawn]
    result = ridge(features(base, ("incoming", "outgoing")), np.array([
        [r["height"], r["height"] < THRESHOLD] for r in base]), 65.)
    score = predict(result, features(rows, ("incoming", "outgoing")))[:, 1]
    cases = match_plan(rows, score, drawn)
    empirical = audit_cells(rows)
    folds = grouped(train, labels)
    receipt = dict(sources=receipts, cases=cases, rows=rows, fit=[v.tolist() for v in result],
                   fit_releases=sorted(set(r["id"] for r in base)), folds=folds,
                   matrix_releases=sorted(map(int, drawn)), empirical=empirical,
                   low_rule=THRESHOLD, hard_rule=HARD, version=_hash(Path(__file__)),
                   region_sampling="candidate pairs chosen by fitted risk gap at exact incoming-event matches on separate releases",
                   draw_seeds=np.random.default_rng(451701).integers(1, 2**30, 12).tolist())
    _write(output, receipt)
    print(len(rows), len(train), len(set(releases)), folds,
          [(r["id"], r["hard"], r["scores"]) for r in cases], flush=True)


def audit_cells(rows):
    out = []
    # Geometry cells fixed before collecting counterfactuals. Counts include legal replies
    # where the receiver crossed the measuring line, and separate answerability counts.
    for dx in (.13, .26):
        for low, high in ((.34, .57), (.57, .76), (.76, .95)):
            for speed_low, speed_high in ((1.5, 2.10), (2.10, 2.65), (2.65, 3.20), (3.20, 4.6)):
                take = [r for r in rows if abs(r["outgoing"][0]) < dx and low < r["outgoing"][1] < high
                        and speed_low < r["outgoing"][2] < speed_high]
                tall = [r["height"] for r in take if r["height"] is not None]
                if not take:
                    continue
                hard = [r["height"] for r in take if r["incoming"][1] < HARD and r["height"] is not None]
                out.append(dict(dx=dx, depth=[low, high], speed=[speed_low, speed_high], n=len(take),
                                answered=sum(r["answered"] for r in take), tracked=len(tall),
                                low=sum(h < THRESHOLD for h in tall), hard_tracked=len(hard),
                                hard_low=sum(h < THRESHOLD for h in hard)))
    return out


def observe_reply(arena, checkpoint, row, shot, seed):
    """Evaluator-only native fork. Keep own contact and departure, redraw just the next request."""
    checkpoint.restore(arena)
    card, leg = row["card"], row["leg"] + 1
    stock_draw = live.draw

    def requested(c, j, protocol, rng):
        return stock_draw(dict(c, reply_seed=seed) if seed is not None and j == leg else c,
                          j, protocol, rng)

    forced = row["departure_tick"] + 2
    neutral = np.zeros(6)
    with patch.object(live, "draw", requested):
        for i in range(row["impact_tick"] - 2, forced + 150):
            action = shot["actions"][i] if i < min(forced, len(shot["actions"])) else neutral
            _, _, ended, capped, _ = arena.exchange(action, FAR_COMMAND)
            if ended or capped or leg + 1 in arena.crossings:
                break
    marks = [e for e in arena.events if e.get("collider") == "learner_racket"]
    own = next((e for e in reversed(marks) if digest_json(e) == row["departure_sha256"]), None)
    if own is None:
        raise ValueError("Counterfactual draw changed the matched actual outgoing arm stroke")
    placements = arena.point_placements()
    reqs = placements["requested"]
    if len(reqs) <= leg or any(r["target"] != list(requested(card, k, "random", None)[0])
                               for k, r in enumerate(reqs)):
        raise ValueError("Native fork changed the independent target protocol")
    ed = placements["delivered"]
    hit = ed[leg] if len(ed) > leg else None
    line = hit["receive_line"] if hit else None
    return dict(seed=seed, target=reqs[leg]["target"], answered=hit is not None,
                height=None if line is None else projected_height(line),
                far_contact=None if hit is None else hit["far_contact"],
                near_bounce=None if hit is None else hit["near_bounce"],
                reason=arena.events[-1].get("reason"), elapsed_s=float(arena.data.time))


def sweep(plan, output, limit):
    d = _read(plan)
    receipt = dict(plan_sha256=_hash(plan), results=[], complete=False)
    if output.exists():
        receipt = _read(output)
        if receipt["plan_sha256"] != _hash(plan):
            raise ValueError("Different physical matrix plan")
    cache = {}
    cases = d["cases"] if limit is None else d["cases"][:limit]
    for c in cases[len(receipt["results"]):]:
        branch = dict(id=c["id"], worlds={})
        for key in ("low_risk", "high_risk"):
            r = c[key]
            if r["source"] not in cache:
                p = Path("artifacts") / (r["source"] + ".json.gz")
                if _hash(p) != d["sources"][r["source"]]:
                    raise ValueError("Changed physical source")
                cache[r["source"]] = _read(p)
            doc = cache[r["source"]]
            shot = doc["rows"][r["row"]]
            card = r["card"]
            with centered(doc["ready"]):
                arena = live.arena(width=64, height=48)
                try:
                    arena.launch(shot["shot"], card=card)
                    arena.partner.reset(seed=card["seed"])
                    cut = r["impact_tick"] - 2
                    for u in shot["actions"][:cut]:
                        arena.exchange(u, FAR_COMMAND)
                    prefix = [e for e in shot["events"] if e["time"] <= arena.data.time + 1e-9]
                    if digest_json(prefix) != digest_json(arena.events):
                        raise ValueError("Matched real command prefix changed")
                    checkpoint = State(arena)
                    orig = observe_reply(arena, checkpoint, r, shot, None)
                    offers = shot["placements"]["delivered"]
                    expected = offers[r["leg"] + 1] if len(offers) > r["leg"] + 1 else None
                    if (orig["target"] != r["request"] or orig["answered"] != (expected is not None)
                            or expected and any(digest_json(orig[k]) != digest_json(expected[k])
                                                for k in ("far_contact", "near_bounce"))
                            or (r["height"] is not None and orig["height"] != r["height"])):
                        raise ValueError("Original draw did not reproduce the original receiver height")
                    trials = [orig, *(observe_reply(arena, checkpoint, r, shot, s) for s in d["draw_seeds"])]
                    branch["worlds"][key] = dict(row=r, score=c["scores"][key == "high_risk"], trials=trials)
                finally:
                    arena.close()
        m, z = [branch["worlds"][k]["trials"][1:] for k in ("low_risk", "high_risk")]
        if any(a["target"] != b["target"] for a,b in zip(m,z,strict=True)):
            raise ValueError("Paired challenge did not keep the same reply target")
        receipt["results"].append(branch)
        receipt["complete"] = len(receipt["results"]) == len(d["cases"])
        _write(output, receipt)
        print(c["id"], [(sum(r["answered"] for r in x),
                        sum(r["height"] is not None and r["height"] < THRESHOLD for r in x))
                       for x in (m,z)], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "sweep"))
    p.add_argument("--plan", type=Path, default=ROOT / "physical-arc-plan.json.gz")
    p.add_argument("--output", type=Path, default=ROOT / "physical-replies.json.gz")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    if a.phase == "fit":
        fit(a.plan)
    else:
        sweep(a.plan, a.output, a.limit)
