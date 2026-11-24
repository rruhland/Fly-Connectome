"""Learn local corrections to raw visual motor output from paired physical misses."""

import argparse
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_fast_camera import CachedStereoRenderer
from .tt29_rally_physics import measured_launches
from .tt29_vision import Rig
from .tt31_curriculum import card, neutral_delivery, opening_candidates
from .tt31_curriculum_opponent import replay_arena
from .tt31_curriculum_summary import compare_delivery, first_deliveries, held_difference
from .tt31_ready import centered
from .tt33_local import ROOT, Local, Signals


OUTPUT = ROOT / "tt34"
MODEL = ROOT / "tt33" / "local-model.npz"
FROZEN_REFERENCE = ROOT / "tt33" / "local-fresh.json.gz"
FAR = np.zeros(5)
OFFSETS = np.array([[0, 0, 0, 0, 0], [.12, 0, 0, 0, 0], [-.12, 0, 0, 0, 0],
                    [0, .12, 0, 0, 0], [0, -.12, 0, 0, 0], [0, 0, 0, .13, 0], [0, 0, 0, -.13, 0]])
SCALE = np.array([.12, .23, .11, .9, 1.65, 1.9, .8, .85, .85, .85, 1.1])


def prepare(feeds, output):
    names = tuple(ROOT / "tt31" / (f + ".json.gz") for f in ("full-plan", "confirm-plan", "recovery-periodic"))
    train = [c for path in names[:2] for c in _read(path)["cards"]]
    train.extend(_read(names[2])["cards"][:2])
    fresh = _read(ROOT / "tt32" / "fresh-plan.json.gz")["cards"]
    demos = _read(MODEL.with_suffix(".json"))["records"]
    excluded = {c["id"] for c in train + fresh + demos}
    prior = _read(ROOT / "tt31" / "course-plan.json.gz")
    excluded.update(c["id"] for c in prior["held"] + prior["staged"])
    ready = _read(ROOT / "tt31" / "left-ready.json")["ready"]
    candidates = [s for s in opening_candidates(measured_launches(feeds)) if s["id"] not in excluded][:250]
    shots = {s["id"]: s for s in candidates}
    neutral = []
    quota = {"left": 3, "middle": 2, "right": 3}
    counts = Counter()
    seed = 340617
    with centered(ready):
        arena = replay_arena([])
        try:
            for i, j in enumerate(np.random.default_rng(seed).permutation(len(candidates))):
                entry = card(candidates[j]["id"], "full", i, seed + i)
                offer = neutral_delivery(arena, shots, entry)
                if offer is None or counts[offer["observed_lane"]] == quota[offer["observed_lane"]]:
                    continue
                counts[offer["observed_lane"]] += 1
                neutral.append(dict(card=entry, delivery=offer))
                if counts == quota:
                    break
        finally:
            arena.close()
    if counts != quota:
        raise ValueError(f"Insufficient held offers: {counts}")
    _write(output / "visual-plan.json.gz", dict(feed_sha256=_hash(feeds), ready=ready, base_sha256=_hash(MODEL),
           practice=train, held=dict(fresh=fresh, new=[d["card"] for d in neutral]), neutral_new=neutral, offsets=OFFSETS.tolist()))
    print("planned", len(train), len(fresh), len(neutral), flush=True)


class Correction:
    """Keep a chosen direct joint offset during an observed incoming motion."""

    def __init__(self, choose):
        self.choose = choose
        self.current = None
        self.choices = []

    def delta(self, tick, observed):
        if observed[4] > .9:
            self.current = None
        if observed[4] >= -.85:
            return None
        distance = np.linalg.norm((observed[:3] - [0., -1.30, 1.00]) / [.49, .67, .46])
        weight = np.clip((1. - distance) / .38, 0., 1.)
        if weight == 0:
            return None
        if self.current is None:
            self.current = int(self.choose(observed[:11]))
            self.choices.append(dict(tick=tick, action=self.current, features=observed[:11].tolist()))
        return weight * OFFSETS[self.current] if self.current else None


def policy(checkpoint, losses):
    x = np.asarray(checkpoint["contexts"])
    values = np.asarray(checkpoint["paired_uplifts"], dtype=float)
    if not losses:
        values = np.maximum(0., values)

    def choose(observed):
        weights = np.exp(-.5 * np.sum(np.square((x - observed) / SCALE), axis=1))
        scores = weights @ values / (1.25 + weights.sum())
        action = int(np.argmax(scores))
        return action if scores[action] >= .045 else 0

    return choose


def play(arena, renderer, rig, model, shot, entry, choose, cap):
    started = process_time()
    arena.launch(shot, card=entry)
    arena.partner.reset(seed=entry["seed"])
    sense = Signals(rig)
    correction = Correction(choose)
    axes = arena._arm_ids[1][2]
    stride = arena._joint_increments[axes] * arena.control_dt / arena.model.opt.timestep
    lo, hi = arena._joint_low[axes], arena._joint_high[axes]
    prev = arena.joints.copy()
    cmd = arena.data.ctrl[axes].copy()
    actions = []
    for tick in range(3600):
        q = arena.joints.copy()
        x = sense.get(tick, arena, renderer, q, (q - prev) / arena.control_dt)
        u = np.zeros(6)
        if x is not None:
            target = model.get(x)
            delta = correction.delta(tick, x)
            if delta is not None:
                target = target + delta
            u[:5] = np.clip((target - cmd[:5]) / (3. * stride[:5]), -.8, .8)
        _, _, end, capped, _ = arena.exchange(u, FAR)
        actions.append(u.tolist())
        cmd = np.clip(cmd + u * stride, lo, hi)
        prev = q
        if end or capped or arena.legal_returns + arena.rival_returns - 1 >= cap:
            break
    score = audit(arena.events, opening_side=1)
    return dict(shot=shot, opponent_seed=entry["seed"], actions=actions, choices=correction.choices,
                events=arena.events.copy(), placements=arena.point_placements(),
                score={k: v for k, v in score.items() if k != "shots"}, process_cpu_s=process_time()-started)


def run(feeds, output, phase, arm):
    deck_path = output / "visual-plan.json.gz"
    deck = _read(deck_path)
    if deck["feed_sha256"] != _hash(feeds) or deck["base_sha256"] != _hash(MODEL):
        raise ValueError("Feed or frozen direct policy changed")
    if phase == "practice":
        cards, cap = deck["practice"], 2
        choose = lambda _: arm
        name = f"visual-practice-{arm}.json.gz"
    else:
        cards, cap = sum(deck["held"].values(), []), 16
        path = output / "visual-checkpoint.json"
        checkpoint = _read(path)
        if checkpoint["plan_sha256"] != _hash(deck_path):
            raise ValueError("Learned table used a different practice deck")
        choose = (lambda _: 0) if phase == "fixed" else policy(checkpoint, phase == "signed")
        name = f"visual-held-{phase}.json.gz"
    shots = measured_launches(feeds)
    with np.load(MODEL) as source:
        model = Local(dict(source))
    result = dict(ready=deck["ready"], opening_side=1, phase=phase, arm=arm if phase == "practice" else None,
                  cap=cap, plan_sha256=_hash(deck_path), model_sha256=_hash(MODEL), cards=cards, rows=[])
    with centered(deck["ready"]):
        arena = replay_arena([])
        renderer = CachedStereoRenderer(arena)
        rig = Rig.from_arena(arena)
        try:
            for i, entry in enumerate(cards):
                row = play(arena, renderer, rig, model, shots[entry["id"]], entry, choose, cap)
                result["rows"].append(row)
                _write(output / name, result)
                print(name, i+1, entry["id"], row["score"]["learner"], row["score"]["returns"],
                      [r["action"] for r in row["choices"]], flush=True)
        finally:
            renderer.close()
            arena.close()


def fit(output):
    deck_path = output / "visual-plan.json.gz"
    deck = _read(deck_path)
    paths = [output / f"visual-practice-{k}.json.gz" for k in range(len(OFFSETS))]
    docs = [_read(p) for p in paths]
    for doc in docs:
        if doc["cards"] != deck["practice"] or len(doc["rows"]) != len(deck["practice"]) or doc["plan_sha256"] != _hash(deck_path):
            raise ValueError("Incomplete or changed exploration")
    contexts, labels, kept = [], [], []
    for entry, rows in zip(deck["practice"], zip(*(doc["rows"] for doc in docs), strict=True), strict=True):
        replies = [r["placements"]["delivered"] for r in rows]
        if not any(replies):
            continue
        if not all(len(r) == 1 for r in replies):
            raise ValueError("Practice exposure was not paired")
        for field in ("target", "far_contact", "near_bounce"):
            if len({digest_json(r[0][field]) for r in replies}) != 1:
                raise ValueError(f"Training first {field} changed")
        if not any(r["choices"] for r in rows):
            continue
        if not all(r["choices"] for r in rows):
            raise ValueError("Practice was observed by only some conditions")
        features = [r["choices"][0]["features"] for r in rows]
        if len({digest_json(x) for x in features}) != 1:
            raise ValueError("The first decision inputs changed between exploration arms")
        contexts.append(features[0])
        labels.append([int(reply[0]["near_landed"]) for reply in replies])
        kept.append(entry)
    values = np.asarray(labels, dtype=int)
    if not len(values):
        raise ValueError("No matched physically delivered reward trials")
    uplift = values - values[:, :1]
    _write(output / "visual-checkpoint.json", dict(plan_sha256=_hash(deck_path), model_sha256=_hash(MODEL),
           training_sha256=[_hash(p) for p in paths], cards=kept, contexts=contexts, rewards=labels,
           paired_uplifts=uplift.tolist(), offsets=OFFSETS.tolist(), scale=SCALE.tolist()))
    print("trial cards", len(kept), "successes", values.sum(axis=0).tolist(),
          "gained", np.sum(uplift > 0, axis=0).tolist(), "lost", np.sum(uplift < 0, axis=0).tolist(), flush=True)


def stats(rows):
    per = [r["score"]["learner"] for r in rows]
    return dict(points=len(rows), arm=sum(per), both=sum(r["score"]["returns"] for r in rows), per_point=per,
                first=sum(bool(r["placements"]["delivered"]) and r["placements"]["delivered"][0]["near_landed"] for r in rows),
                choices=dict(Counter(c["action"] for r in rows for c in r["choices"])), cpu_s=sum(r["process_cpu_s"] for r in rows))


def summary(output):
    deck = _read(output / "visual-plan.json.gz")
    reference = _read(FROZEN_REFERENCE)
    names = ("fixed", "signed", "positive")
    papers = {p: _read(output / f"visual-held-{p}.json.gz") for p in names}
    total = sum(map(len, deck["held"].values()))
    for doc in papers.values():
        if len(doc["rows"]) != total or doc["cards"] != sum(deck["held"].values(), []):
            raise ValueError("Incomplete assessment or changed held deck")
    fixed = papers["fixed"]["rows"]
    for a, b in zip(fixed[:len(reference["rows"])], reference["rows"], strict=True):
        if any(digest_json(a[k]) != digest_json(b[k]) for k in ("actions", "events", "placements")):
            raise ValueError("Zero correction changed the reference pupil")
    comparison = {}
    loss_ablation = {}
    start = 0
    for name, cards in deck["held"].items():
        section = slice(start, start+len(cards))
        start += len(cards)
        base = fixed[section]
        comparison[name] = {p: dict(totals=stats(d["rows"][section]),
            paired=held_difference(d["rows"][section], base), first_offer=compare_delivery(
                first_deliveries(d["rows"][section]), first_deliveries(base))) for p, d in papers.items()}
        signed, positive = (papers[p]["rows"][section] for p in ("signed", "positive"))
        loss_ablation[name] = dict(paired=held_difference(signed, positive), exact_repetitions={
            k: sum(digest_json(a[k]) == digest_json(b[k]) for a, b in zip(signed, positive, strict=True))
            for k in ("choices", "actions", "events", "placements")})
    hashes = {p: _hash(output / f"visual-held-{p}.json.gz") for p in names}
    _write(output / "visual-summary.json", dict(checkpoint_sha256=_hash(output / "visual-checkpoint.json"),
           frozen_reference_equivalent=len(reference["rows"]), evaluated_sha256=hashes, held=comparison,
           loss_ablation=loss_ablation))
    print({name: {p: d["totals"]["arm"] for p, d in rows.items()} for name, rows in comparison.items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("phase", choices=("prepare", "practice", "fit", "fixed", "signed", "positive", "summary"))
    p.add_argument("--arm", type=int, choices=range(len(OFFSETS)), default=0)
    p.add_argument("--output", type=Path, default=OUTPUT)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase == "prepare":
        prepare(a.feeds, a.output)
    elif a.phase == "fit":
        fit(a.output)
    elif a.phase == "summary":
        summary(a.output)
    else:
        run(a.feeds, a.output, a.phase, a.arm)


if __name__ == "__main__":
    main()
