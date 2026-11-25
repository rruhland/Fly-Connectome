"""Can a private partner forecast which of two achieved departures leaves a playable reply?"""

import argparse
from pathlib import Path

import numpy as np

from . import tt51_afterstate as response
from . import tt55_paired_afterstate as pairs
from . import tt55_paired_preference as effects
from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _read, _write, _hash
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt82_partner_model")
PLAN = ROOT / "protocol.json"
MODEL = ROOT / "receiver.npz"
mpc = pairs.mpc


def plan():
    source = _read(pairs.PLAN)
    excluded = {r["id"] for k in ("train", "held") for r in source[k]}
    obs, reward, ids = response.receiver_corpus(excluded)
    receiver = response.fit(response.recv_feature(obs), reward, 70.0)
    ROOT.mkdir(exist_ok=True)
    np.savez(
        MODEL,
        **{
            k: v
            for k, v in zip(("mu", "sigma", "mean", "weight"), receiver, strict=True)
        },
    )
    rng = np.random.default_rng(8268514)
    d = {
        k: [
            dict(id=r["id"], seeds=rng.integers(2**30, size=10).tolist())
            for r in source[k]
        ]
        for k in ("train", "held")
    }
    _write(
        PLAN,
        dict(
            **d,
            receiver=_hash(MODEL),
            pairs=_hash(pairs.PLAN),
            train_rows=len(obs),
            train_ids=len(set(ids)),
            rule="Both arms use the same independently generated private goals and an identical neutral far carriage. Start the surrogate from a real achieved departure to test its control authority with favorable information. The receipt is diagnostic. Official games continue normally. Do not deploy if matched legal next returns do not favor the predicted actions.",
        ),
    )


def survey(a, card, departure, seed):
    p = departure["ball"]
    v = departure["velocity"]
    spin = departure["spin"]
    ident = card["id"]
    shot = dict(
        id=ident,
        pos_x=p[0],
        pos_y=p[1],
        pos_z=p[2] - TABLE_HEIGHT,
        **dict(zip(("vel_x", "vel_y", "vel_z"), v, strict=True)),
        **dict(zip(("w_vel_x", "w_vel_y", "w_vel_z"), spin, strict=True)),
    )
    c = dict(card, seed=seed, reply_seed=seed)
    a.launch(shot, card=c)
    a.partner.reset(seed=seed)
    stop = np.zeros(6)
    for _ in range(175):
        _, _, done, limit, _ = a.exchange(stop, mpc.FAR_COMMAND)
        if done or limit or 1 in a.crossings:
            break
    if any(e.get("reason") == "double_hit" and e["time"] < 0.05 for e in a.events):
        raise ValueError("Private replay contaminated by the near arm")
    delivered = a.point_placements()["delivered"]
    record = delivered[0] if delivered else None
    if not record or not record["receive_line"]:
        return None
    line = record["receive_line"]
    return response.projected(np.r_[line["xyz"], line["velocity"]]).tolist()


def run(name):
    design = _read(PLAN)
    if design["pairs"] != _hash(pairs.PLAN) or design["receiver"] != _hash(MODEL):
        raise ValueError("Different training or candidate packet")
    src = _read(pairs.PLAN)[name]
    out = ROOT / ("forecast-" + name + ".json")
    data = _read(out) if out.exists() else dict(protocol=_hash(PLAN), rows=[])
    if data["protocol"] != _hash(PLAN):
        raise ValueError("Incompatible resume")
    with np.load(MODEL) as m:
        receiver = [m[k] for k in ("mu", "sigma", "mean", "weight")]
    for idx in range(len(data["rows"]), len(src)):
        case = src[idx]
        entry = design[name][idx]
        if entry["id"] != case["id"]:
            raise ValueError("Changed matched physical fork")
        branches = []
        # No recorded partner pose, evaluator target, or actual next ball is restored.
        with enabled(), mpc.centered(case["ready"]):
            a = mpc.arena(width=64, height=48)
            # The shot has already departed the near blade. The source of that
            # shot is outside the predicted partner interaction. A reset near
            # paddle can otherwise hit the imported ball again on substep one.
            a.model.body("learner_base").pos[0] += 2.0
            try:
                for option in case["arms"]:
                    preds = [
                        survey(a, case["card"], option["depart"], int(s))
                        for s in entry["seeds"]
                    ]
                    scores = [
                        0.0
                        if x is None
                        else float(
                            np.clip(
                                response.predict(
                                    receiver, response.recv_feature(np.asarray(x)[None])
                                )[0],
                                0,
                                1,
                            )
                        )
                        for x in preds
                    ]
                    branches.append(
                        dict(
                            returnability=float(np.mean(scores)),
                            arrived=sum(x is not None for x in preds),
                            forecasts=preds,
                        )
                    )
            finally:
                a.close()
        data["rows"].append(dict(id=entry["id"], branches=branches))
        _write(out, data)
        print(
            idx + 1,
            entry["id"],
            [round(b["returnability"], 3) for b in branches],
            flush=True,
        )


def report():
    output = {}
    for name in ("train", "held"):
        rows, complete = effects.collect(name)
        d = _read(ROOT / ("forecast-" + name + ".json"))["rows"]
        if not complete or [r["id"] for r in rows] != [r["id"] for r in d]:
            raise ValueError("Unfinished paired survey")
        scores = np.asarray(
            [np.diff([b["returnability"] for b in r["branches"]])[0] for r in d]
        )
        truth = np.array([r["contrasts"]["all"]["physical_next_legal"] for r in rows])
        optimal_height = np.array(
            [r["contrasts"]["all"]["height_m"] for r in rows], dtype=float
        )
        height = np.array(
            [
                np.diff(
                    [
                        np.mean([v[0] for v in b["forecasts"] if v is not None])
                        if b["arrived"]
                        else np.nan
                        for b in r["branches"]
                    ]
                )[0]
                for r in d
            ]
        )
        output[name] = dict(
            n=len(rows),
            estimated_next_win=effects.robust_average(0.5 * np.sign(scores) * truth),
            forecast_height_contrast=effects.signs(height, optimal_height),
            actual_score_contrast=effects.signs(scores, truth),
            distinct_departures=len(rows) * 2,
            receiver_arrivals=sum(b["arrived"] for r in d for b in r["branches"]),
        )
    _write(ROOT / "report.json", output)
    print(output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op", choices=("plan", "train", "held", "report"))
    a = p.parse_args()
    if a.op == "plan":
        plan()
    elif a.op == "report":
        report()
    else:
        run(a.op)
