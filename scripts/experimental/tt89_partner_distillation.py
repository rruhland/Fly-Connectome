"""Distill independently simulated partner replies into a fast pre-contact action ranking."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import tt56_scored_motor_atlas as archive
from . import tt82_partner_model as shadow
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt89_partner_distillation")
PLAN = ROOT / "protocol.json"
POLICY = ROOT / "policy.npz"
pairs, mpc = shadow.pairs, shadow.mpc
N = 256
DRAWS = 8
SHARDS = 4


def feature(decision):
    return np.r_[decision["target_errors"], decision["p"], decision["contact_delay_s"]]


def collect():
    prior = _read(pairs.PLAN)
    forbidden = {c["id"] for group in ("train", "held") for c in prior[group]}
    forbidden.update(c["id"] for c in _read(archive.dose.PLAN)["practice"])
    forbidden.update(c["id"] for c in _read(archive.season.PLAN)["practice"])
    rows = {}
    for file, h in _read(archive.ROOT / "fit.json")["sources"].items():
        p = Path(file)
        if h != _hash(p):
            raise ValueError("Changed physical experience")
        d = _read(p)
        for card, r in zip(d["cards"], d["rows"]):
            if (
                card["id"] in forbidden
                or card.get("profile") != "reach"
                or card.get("opening") != "random"
            ):
                continue
            for leg, offer in enumerate(r["placements"]["delivered"]):
                contact = offer.get("near_contact")
                if not offer.get("near_landed") or not contact:
                    continue
                tick = contact["time"] / mpc.DT
                options = [
                    q
                    for q in r["searches"]
                    if q.get("target_errors")
                    and q.get("p")
                    and 0 < tick - q["tick"] < 8
                ]
                if not options:
                    continue
                q = options[-1]
                if abs(tick - q["tick"] - q["contact_delay_s"] / mpc.DT) > 4:
                    continue
                dep = next(
                    (
                        e
                        for e in r["events"]
                        if e.get("collider") == "learner_racket"
                        and contact["time"] < e["time"] < contact["time"] + 0.087
                    ),
                    None,
                )
                if dep is None:
                    continue
                key = (
                    card["id"],
                    *np.round(dep["ball"] + dep["velocity"] + dep["spin"], 5),
                )
                rec = dict(
                    card=card,
                    ready=d["ready"],
                    depart=dep,
                    visible=feature(q).tolist(),
                    file=file,
                    leg=leg,
                )
                rows.setdefault(key, rec)
    return list(rows.values()), sorted(forbidden)


def freeze():
    rows, forbidden = collect()
    rng = np.random.default_rng(895115)
    keys = np.array([r["visible"] for r in rows])
    idx = rng.permutation(len(rows))
    # Half natural draws, half farthest-point sampling in observable response space.
    # Approximate integration is local at deployment; these are action prototypes.
    take = list(idx[: N // 2])
    z = (keys - np.median(keys, axis=0)) / np.maximum(np.std(keys, axis=0), 0.02)
    near = cKDTree(z[take]).query(z)[0] ** 2
    for _ in range(N - len(take)):
        i = int(np.argmax(near))
        take.append(i)
        near = np.minimum(near, np.sum((z - z[i]) ** 2, axis=1))
    ROOT.mkdir(exist_ok=True)
    out = [dict(rows[i], seeds=rng.integers(2**30, size=DRAWS).tolist()) for i in take]
    _write(
        PLAN,
        dict(
            cases=out,
            forbidden=forbidden,
            available=len(rows),
            release_count=len(set(r["card"]["id"] for r in rows)),
            receiver=_hash(shadow.MODEL),
            method="The private partner receives actual historical scored departures and eight independent continuous draws with neutral initial carriage. A conditional local readout uses ONLY causal private candidate metrics and previous observed ball position before deciding. The training shadow receives post-contact state, never the actor. Compare predicted choices against actual matched completely independent TT55 branches.",
        ),
    )


def shard(i):
    design = _read(PLAN)
    dest = ROOT / f"survey-{i}.json"
    prior = _read(dest) if dest.exists() else dict(protocol=_hash(PLAN), rows=[])
    if prior["protocol"] != _hash(PLAN) or design["receiver"] != _hash(shadow.MODEL):
        raise ValueError("Changed course")
    indices = list(range(i, len(design["cases"]), SHARDS))
    with np.load(shadow.MODEL) as f:
        receiver = [f[n] for n in ("mu", "sigma", "mean", "weight")]
    a = None
    ready = None
    try:
        for j in indices[len(prior["rows"]) :]:
            case = design["cases"][j]
            with enabled(), mpc.centered(case["ready"]):
                if ready != case["ready"]:
                    if a is not None:
                        a.close()
                    a = mpc.arena(width=64, height=48)
                    a.model.body("learner_base").pos[0] += 2.0
                    ready = case["ready"]
                arrivals = [
                    shadow.survey(a, case["card"], case["depart"], s)
                    for s in case["seeds"]
                ]
            values = [
                float(
                    np.clip(
                        shadow.response.predict(
                            receiver, shadow.response.recv_feature(np.asarray(v)[None])
                        )[0],
                        0,
                        1,
                    )
                )
                if v is not None
                else 0.0
                for v in arrivals
            ]
            prior["rows"].append(
                dict(index=j, returnability=np.mean(values), arrivals=arrivals)
            )
            _write(dest, prior)
            if len(prior["rows"]) % 8 == 0:
                print(i, len(prior["rows"]), flush=True)
    finally:
        if a is not None:
            a.close()


class Critic:
    def __init__(self, x, y):
        self.mu = np.median(x, axis=0)
        self.scale = np.maximum(0.025, np.std(x, axis=0))
        self.inputs = (x - self.mu) / self.scale
        self.tree = cKDTree(self.inputs)
        self.y = y

    def value(self, fx):
        z = (np.atleast_2d(fx) - self.mu) / self.scale
        ds, idx = self.tree.query(z, k=16)
        w = np.exp(-2.4 * (ds / np.maximum(0.05, ds[:, -1:])) ** 2)
        return (w * self.y[idx]).sum(1) / w.sum(1)


def evaluate():
    design = _read(PLAN)
    rows = [r for i in range(SHARDS) for r in _read(ROOT / f"survey-{i}.json")["rows"]]
    rows.sort(key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(len(design["cases"]))):
        raise ValueError("Incomplete surrogate training")
    x = np.array([r["visible"] for r in design["cases"]])
    y = np.array([r["returnability"] for r in rows])
    fit = Critic(x, y)
    np.savez_compressed(POLICY, x=x, y=y)
    context = shadow.effects.old.Coverage(_read(pairs.season.fresh.RISK))
    output = {}
    for name in ("train", "held"):
        actual, complete = shadow.effects.collect(name)
        if not complete:
            raise ValueError("Unfinished physical contrast")
        credit = []
        existing = []
        raw = []
        for case in actual:
            if any(v is None for v in case["candidates"]):
                continue
            a, b = case["candidates"]
            score = fit.value(np.asarray([feature(a), feature(b)]))
            risk, support = context.risk(
                np.array([a["target_errors"], b["target_errors"]])
            )
            reference = np.sign(risk[0] - risk[1]) if np.all(support) else 0.0
            truth = case["contrasts"]["all"]["physical_next_legal"]
            credit.append(0.5 * np.sign(score[1] - score[0]) * truth)
            existing.append(0.5 * reference * truth)
            raw.append(
                dict(id=case["id"], scores=score.tolist(), observed_advantage=truth)
            )
        output[name] = dict(
            vs_no_preference=shadow.effects.robust_average(credit),
            vs_existing=shadow.effects.robust_average(np.asarray(credit) - existing),
            rows=raw,
        )
    _write(ROOT / "test.json", dict(groups=output, policy=_hash(POLICY)))
    print({k: v["vs_existing"] for k, v in output.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op", choices=("plan", "survey", "test"))
    p.add_argument("--shard", choices=range(SHARDS), type=int, default=0)
    a = p.parse_args()
    if a.op == "plan":
        freeze()
    elif a.op == "survey":
        shard(a.shard)
    else:
        evaluate()
