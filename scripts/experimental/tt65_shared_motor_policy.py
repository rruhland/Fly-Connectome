"""Optimize a single sensor-conditioned motor policy across multiple attained physical arrivals."""

import argparse
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np
from . import tt57_motor_school as school
from .tt4_transfer_trial import _hash, _read, _write
from .tt44_paired_credit import State

mpc = school.mpc
ROOT = Path("artifacts/tt65_shared_motor_policy")
PLAN = ROOT / "protocol.json"
MODEL = ROOT / "policy.npz"
H = school.H
TRAIN = [
    school.season.ROOT / "learning-96.json.gz",
    school.AUX / "live-from-12.json.gz",
]
TEST = [school.season.ROOT / "after96.json.gz", school.season.ROOT / "after128.json.gz"]
STAGES = ("untouched", "touched", "routine")
RAMP = np.minimum(1, school.PHASE / 7)
BASIS = np.column_stack(
    (RAMP, RAMP * (2 * school.PHASE / H - 1), RAMP * np.sin(np.pi * school.PHASE / H))
)
D = 9
P = 18


def candidates(paths):
    result = {k: [] for k in STAGES}
    for p in paths:
        d = _read(p)
        for i, (c, r) in enumerate(zip(d["cards"], d["rows"])):
            if c.get("profile") != "reach" or any(
                z["draw"] != "random" for z in r["placements"]["requested"]
            ):
                continue
            for o in r["placements"]["delivered"]:
                if not all(
                    o.get(k) for k in ("far_contact", "near_bounce", "receive_line")
                ):
                    continue
                kind = (
                    "routine"
                    if o["near_landed"]
                    else ("touched" if o["near_contact"] else "untouched")
                )
                z = school.projected_height(o["receive_line"])
                if kind == "routine" and not 1.005 < z < 1.12:
                    continue
                if kind == "untouched" and z > 1.015:
                    continue
                a = o["far_contact"]["step"] + 3
                b = (o.get("near_contact") or r["events"][-1])["step"] - 7
                sel = next(
                    (
                        n
                        for n in r["searches"]
                        if n.get("p")
                        and -0.99 < n["p"][1] < -0.51
                        and a < n["tick"] < b
                    ),
                    None,
                )
                if sel:
                    result[kind].append(
                        dict(
                            source=str(p),
                            i=i,
                            id=c["id"],
                            leg=o["leg"],
                            tick=sel["tick"],
                            kind=kind,
                            baseline=bool(o["near_landed"]),
                        )
                    )
    return result


def draw(pool, n, rng, exclude):
    chosen = []
    used = Counter()
    for j in range(n):
        for kind in STAGES:
            options = [
                r for r in pool[kind] if r not in chosen and r["id"] not in exclude
            ]
            if not options:
                continue
            order = rng.permutation(len(options))
            r = min(
                (options[i] for i in order),
                key=lambda r: (used[r["id"]], r["tick"] // 700),
            )
            chosen.append(r)
            used[r["id"]] += 1
    return chosen


def plan():
    rng = np.random.default_rng(65011788)
    test = draw(candidates(TEST), 5, rng, set())
    held = {v["id"] for v in test}
    train = draw(candidates(TRAIN), 8, rng, held)
    with np.load(school.atlas.MODEL) as z:
        if held & set(z["release"].tolist()):
            raise ValueError("Held releases in previous visual motor archive")
    if len(train) != 24 or len(test) != 15:
        raise ValueError("Incomplete physical roster")
    ROOT.mkdir(parents=True, exist_ok=True)
    _write(
        PLAN,
        dict(
            training=train,
            test=test,
            sources={str(p): _hash(p) for p in TRAIN + TEST},
            atlas=_hash(school.atlas.MODEL),
            rule="One shared RGB/encoder policy is optimized across eight untouched ordinary misses, eight touched misses and eight returns. A single unconditional forward pass commands the same arm at evaluation. Compare one policy attempt with the same untrained actor and the archived original on five unused offers of each category. Native states only initialize practice plants or evaluate. No hidden state enters the policy. No opening release from evaluation enters the joint action training bank.",
        ),
    )


def inputs(x, q, ref, bounced):
    return np.r_[x[:6], q[0], q[3], float(bounced)]


def context(x, mu, scale):
    z = np.clip((x - mu) / scale, -2, 2)
    return np.r_[
        1.0, 0.43 * z[:5], 0.4 * z[5], 0.26 * z[6], 0.26 * z[7]
    ]  # bounded context; state 8 indicates bounce for donor selection


def decode(weights, phi, roots):
    # The three temporal functions can express individual joints moving in
    # either direction or reversing. No predefined hitting pose or intercept.
    params = np.einsum("bdp,d->bp", weights, phi)
    logits = params[:, -3:].copy() + np.array([0.65, 0, -0.65])
    coef = np.exp(logits - logits.max(1, keepdims=True))
    mix = np.einsum("bi,ihj->bhj", coef / coef.sum(1, keepdims=True), roots)
    delta = np.einsum("ht,btj->bhj", BASIS, params[:, :15].reshape(-1, 3, 5))
    mix[:, :, :5] += 0.35 * np.tanh(delta)
    return mix


def collect(a, cam, rig, specs, docs, bank):
    out = []
    axes = a._arm_ids[1][2]
    for s in specs:
        doc = docs[s["source"]]
        c, r = doc["cards"][s["i"]], doc["rows"][s["i"]]
        eye, x, q, ref, prev, _, _ = school.observed_prefix(
            a, cam, rig, c, r, s["tick"], axes
        )
        if x is None:
            raise ValueError("Missing registered RGB at training cut")
        b = bool(eye.eye.latest.bounced)
        k = school.atlas.key(s["tick"], x[:3], b, ref, prev)
        donors, _ = bank.paths(k, x, c["id"])
        if len(donors) < 3:
            raise ValueError("Insufficient independent previous movements")
        out.append(
            dict(
                spec=s,
                raw=inputs(x, q, ref, b),
                ref=ref.copy(),
                roots=np.stack([m for _, m in donors[:3]]),
                state=State(a),
            )
        )
    return out


def value(legal, y, fb):
    # Simulator-based teaching only. Mild approach and physical exchange feedback
    # can break ties between movements that all currently earn zero.
    separation = fb[:, 1] > 0.5
    dense = np.where(
        separation,
        0.10 + 0.10 * np.clip(fb[:, 3] / 4, 0, 1),
        0.08 * np.exp(-np.clip(fb[:, 0], 0, 1) / 0.06),
    )
    good = np.where(legal, 0.90 + 0.1 * np.clip((0.083 - y[:, 12]) / 0.06, 0, 1), 0)
    return dense + good


def train(a, engine, cases):
    rng = np.random.default_rng(650771)
    states = np.array([r["raw"] for r in cases])
    mu = states.mean(0)
    scale = np.maximum(
        states.std(0), [0.08, 0.08, 0.03, 0.24, 0.32, 0.34, 0.1, 0.12, 0.30]
    )
    center = np.zeros((D, P))
    best = center.copy()
    bestscore = -10
    std = 0.45
    records = []
    start = process_time()
    for epoch in range(15):
        # Orthogonal antithetic perturbations; all siblings see all 24
        # *identical* physical situations in a generation.
        noise = rng.normal(size=(15, D * P))
        orth = np.linalg.qr(noise.T)[0].T.reshape(-1, D, P) * np.sqrt(D * P)
        population = np.concatenate(
            ([center, best], center + std * orth, center - std * orth)
        )
        reward = []
        valid = []
        for case in cases:
            case["state"].restore(a)
            paths = decode(population, context(case["raw"], mu, scale), case["roots"])
            _, y, fb, _, ok = school.native(engine, paths, case["ref"])
            reward.append(value(ok, y, fb))
            valid.append(ok)
        reward = np.array(reward)
        valid = np.array(valid)
        score = reward.mean(0)
        elite = np.argsort(score)[-5:][::-1]
        if score[elite[0]] > bestscore:
            bestscore = float(score[elite[0]])
            best = population[elite[0]].copy()
        center = 0.48 * center + 0.52 * np.einsum(
            "b,bdp->dp", [0.37, 0.25, 0.18, 0.12, 0.08], population[elite]
        )
        records.append(
            dict(
                epoch=epoch,
                center_legal=int(valid[:, 0].sum()),
                top_legal=int(valid[:, elite[0]].sum()),
                best_value=bestscore,
                top_each={
                    k: int(
                        valid[[r["spec"]["kind"] == k for r in cases], elite[0]].sum()
                    )
                    for k in STAGES
                },
            )
        )
        std = max(0.14, 0.45 * (0.92 ** (epoch + 1)))
        np.savez_compressed(MODEL, weights=best, mu=mu, scale=scale)
        _write(
            ROOT / "training.json",
            dict(
                protocol=_hash(PLAN),
                model=_hash(MODEL),
                epochs=records,
                cpu=process_time() - start,
                complete=epoch == 14,
            ),
        )
        print(records[-1], flush=True)


def assess(a, engine, cases, docs, guard):
    report = dict(protocol=_hash(PLAN), model=_hash(MODEL), rows=[])
    if not _read(ROOT / "training.json")["complete"]:
        raise ValueError("Unfinished practice")
    with np.load(MODEL) as z:
        weights, mu, scale = z["weights"], z["mu"], z["scale"]
    for case in cases:
        s = case["spec"]
        doc = docs[s["source"]]
        c, r = doc["cards"][s["i"]], doc["rows"][s["i"]]
        ref = case["ref"]
        paths = decode(
            np.stack((np.zeros_like(weights), weights)),
            context(case["raw"], mu, scale),
            case["roots"],
        )
        case["state"].restore(a)
        us, y, fb, loss, ok = school.native(engine, paths, ref)
        results = {}
        for i, who in enumerate(("zero", "shared")):
            # Convert only the desired joint path to real actions. The native
            # rollout never selects a held movement or alters the issuing rule.
            stride = a._joint_increments[engine.axes] * engine.micro
            own_ref = ref.copy()
            actual = []
            for target in ref + paths[i]:
                command = np.clip(
                    (
                        np.clip(
                            target,
                            a._joint_low[engine.axes],
                            a._joint_high[engine.axes],
                        )
                        - own_ref
                    )
                    / stride,
                    -0.8,
                    0.8,
                )
                own_ref = np.clip(
                    own_ref + stride * command,
                    a._joint_low[engine.axes],
                    a._joint_high[engine.axes],
                )
                actual.append(command)
            reported = np.flatnonzero(np.any(us[i, :H] != 0.0, axis=1))
            if len(reported) and not np.allclose(
                np.array(actual)[: reported[-1] + 1],
                us[i, : reported[-1] + 1],
                atol=1e-7,
            ):
                raise ValueError(
                    "Commanded prefix differs from the independent native rate limiter"
                )
            tape = np.vstack([actual, np.zeros((95, 6))])
            v = school.physical(a, case["state"], c, r, s, tape, guard)
            results[who] = dict(**v, nominal=bool(ok[i]))
        report["rows"].append(dict(case=s, arms=results))
        _write(ROOT / "physical.json.gz", report)
        print(
            s["id"],
            s["kind"],
            [(k, v["landed"], v["answered"]) for k, v in results.items()],
            flush=True,
        )


def run(phase):
    d = _read(PLAN)
    specs = d["training" if phase == "train" else "test"]
    docs = {k: _read(Path(k)) for k in {s["source"] for s in specs}}
    if d["atlas"] != _hash(school.atlas.MODEL) or any(
        _hash(Path(k)) != d["sources"][k] for k in docs
    ):
        raise ValueError("Source changed")
    ready = next(iter(docs.values()))["ready"]
    if any(v["ready"] != ready for v in docs.values()):
        raise ValueError("Body or posture changed")
    with school.enabled(), mpc.centered(ready):
        a = mpc.arena()
        cam, rig, *_ = mpc.resources(a)
        try:
            cases = collect(a, cam, rig, specs, docs, school.Donors())
            engine = school.Rollout(a, 1)
            if phase == "train":
                train(a, engine, cases)
            else:
                cad = mpc.arena(width=64, height=48, detail="cad")
                try:
                    assess(
                        a,
                        engine,
                        cases,
                        docs,
                        school.CourtClearance(cad.model, articulated=True),
                    )
                finally:
                    cad.close()
        finally:
            cam.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan", "train", "assess"))
    args = p.parse_args()
    plan() if args.phase == "plan" else run(args.phase)
