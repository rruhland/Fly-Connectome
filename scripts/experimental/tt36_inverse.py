"""Adapt each attempted joint correction to its observed spatial and outgoing-velocity errors."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import resources
from .tt36_course import OUT
from .tt36_local import BASE, IDS, SCALE, trial


def fit_prior():
    source = OUT / "hit-practice.json.gz"
    d = _read(source)
    inputs = []
    delta = []
    wins = []
    approach = []
    for rows in d["rounds"]:
        for i, row in enumerate(rows):
            outcomes = row["optical_outgoing"]
            pulses = [
                e["time"]
                for e in row["events"]
                if e["kind"] == "legal_return" and e["side"] == 1
            ][1:]
            for j, y in enumerate(outcomes):
                bound = min(
                    y["reversal_time"] + 1.25,
                    outcomes[j + 1]["reversal_time"]
                    if j + 1 < len(outcomes)
                    else np.inf,
                )
                if any(y["reversal_time"] < t < bound for t in pulses):
                    wins.append(y["velocity"])
                    approach.append(y["incoming"])
            a = row["optical_first_turn"]
            if a:
                for k in range(i):
                    b = rows[k]["optical_first_turn"]
                    if b:
                        inputs.append(
                            np.subtract(d["offsets_rad"][i], d["offsets_rad"][k])
                        )
                        delta.append(np.subtract(a["velocity"], b["velocity"]))
    x, y = np.array(inputs), np.array(delta)
    # Five-axis linear response regularized in directions with few motor probes.
    jac = np.linalg.solve(x.T @ x + 0.035 * np.eye(5), x.T @ y).T
    _write(
        OUT / "inverse-prior.json",
        {
            "input_sha256": _hash(source),
            "velocity_jacobian": jac.tolist(),
            "rewarded_velocities": wins,
            "approaches": approach,
            "matched_velocity_pairs": len(x),
        },
    )


def measured(ep):
    return ep["row"]["optical"]


def velocity(row):
    return row["observed_outgoing_mps"]


def target(prior, ctx):
    # Compare incoming directions in the previously rewarded observations.
    incoming = np.asarray(prior["approaches"])[:, 3:6]
    dist = np.linalg.norm((incoming - ctx[3:6]) / [0.8, 1.2, 1.8], axis=1)
    indices = np.argsort(dist)[:7]
    return np.median(np.asarray(prior["rewarded_velocities"])[indices], axis=0)


def rank(ep, goal):
    saw = measured(ep)
    if saw is None:
        quality = -3.0
    elif velocity(saw) is not None:
        quality = 1.0 - 0.16 * np.linalg.norm(np.subtract(velocity(saw), goal))
    else:
        quality = -saw["gap_m"]
    return ep["row"]["score"]["learner"], quality, -np.linalg.norm(ep["offset_rad"])


def candidate(history, prior, goal, step):
    valid = [e for e in history if measured(e)]
    best = max(valid, key=lambda e: rank(e, goal))
    origin = np.asarray(best["offset_rad"])
    x = np.array([e["offset_rad"] for e in valid]) - origin
    saw = measured(best)
    speeds = [e for e in valid if velocity(measured(e)) is not None]
    if velocity(saw) is not None:
        prior_j = np.asarray(prior["velocity_jacobian"])
        inputs = np.array([v["offset_rad"] for v in speeds]) - origin
        outputs = np.array([velocity(measured(e)) for e in speeds]) - velocity(saw)
        # Start with a transferable action-to-flight model; use local errors to adjust it.
        strength = 0.035
        jac = np.linalg.solve(
            inputs.T @ inputs + strength * np.eye(5),
            inputs.T @ outputs + strength * prior_j.T,
        ).T
        residual = goal - np.array(velocity(saw))
        damp = 0.25
    else:
        # Closest observed separation is noisy in depth (30 Hz). Give lateral and vertical
        # directions priority but keep an explicit small measured depth term.
        weight = np.array([1.0, 0.25, 1.0])
        outputs = (
            np.array([measured(e)["gap_xyz_m"] for e in valid]) - saw["gap_xyz_m"]
        ) * weight
        jac = np.linalg.solve(x.T @ x + 0.007 * np.eye(5), x.T @ outputs).T
        residual = -np.array(saw["gap_xyz_m"]) * weight
        damp = 0.002
    move = jac.T @ np.linalg.solve(jac @ jac.T + damp * np.eye(3), residual)
    radius = (0.38, 0.54, 0.48, 0.65)[step]
    move *= min(1.0, radius / max(1e-12, np.linalg.norm(move)))
    proposal = np.clip(origin + move, -0.68, 0.68)
    if min(np.linalg.norm(proposal - e["offset_rad"]) for e in history) < 0.045:
        # Probe an unmeasured action direction if repeated observations yield the same step.
        _, _, basis = np.linalg.svd(x, full_matrices=False)
        proposal = np.clip(
            proposal + (0.20 if step % 2 == 0 else -0.20) * basis[-1], -0.68, 0.68
        )
    return proposal


def practice(feeds):
    prior = _read(OUT / "inverse-prior.json")
    plan = _read(OUT / "local-plan.json.gz")
    shots = measured_launches(feeds)
    model = _read(BASE)
    records = []
    memory = []
    with centered(model["ready"]):
        arena = replay_arena([])
        renderer, rig, stride, lo, hi = resources(arena)
        try:
            for case in plan["cases"]:
                card = case["original"]
                shared = OUT / f"local-practice-{card['id']}.json.gz"
                history = _read(shared)["common"].copy()
                ctx = history[0]["row"]["picks"][0]["observation"]
                goal = target(prior, ctx)
                extra = []
                for i in range(4):
                    action = candidate(history, prior, goal, i)
                    row = trial(
                        arena,
                        renderer,
                        rig,
                        stride,
                        lo,
                        hi,
                        model,
                        shots,
                        card,
                        lambda _, action=action: action,
                        case["practice_reply_seeds"][7 + i],
                    )
                    ep = {"attempt": 8 + i, "offset_rad": action.tolist(), "row": row}
                    history.append(ep)
                    extra.append(ep)
                winner = max(history, key=lambda e: rank(e, goal))
                memory.append(
                    {
                        "context": ctx,
                        "offset": winner["offset_rad"],
                        "own": winner["row"]["score"]["learner"],
                    }
                )
                records.append(
                    {
                        "practice_sha256": _hash(shared),
                        "goal_mps": goal.tolist(),
                        "new": extra,
                    }
                )
                _write(
                    OUT / "inverse-practice.json.gz",
                    {
                        "cases": records,
                        "ids": IDS[: len(records)],
                        "prior_sha256": _hash(OUT / "inverse-prior.json"),
                    },
                )
                print(
                    card["id"],
                    [
                        (
                            e["row"]["score"]["learner"],
                            e["row"]["placements"]["delivered"][0]["near_contact"]
                            is not None,
                        )
                        for e in extra
                    ],
                    flush=True,
                )
        finally:
            renderer.close()
            arena.close()
    _write(
        OUT / "inverse-checkpoint.json",
        {
            "memory": memory,
            "scale": SCALE.tolist(),
            "source_sha256": _hash(OUT / "inverse-practice.json.gz"),
        },
    )


def evaluate(feeds):
    plan = _read(OUT / "local-plan.json.gz")
    ck = _read(OUT / "inverse-checkpoint.json")
    model = _read(BASE)
    context = np.array([e["context"] for e in ck["memory"]])
    offsets = np.array([e["offset"] for e in ck["memory"]])

    def choose(observed):
        distance = np.linalg.norm((context - observed) / SCALE, axis=1)
        i = int(np.argmin(distance))
        return (
            offsets[i] * np.exp(-0.5 * (distance[i] / 1.25) ** 2)
            if distance[i] < 2.35
            else np.zeros(5)
        )

    schedule = [
        ("reply", c["original"], s) for c in plan["cases"] for s in c["reply_seeds"]
    ]
    schedule += [
        ("first", c["changed"], 365061 + i) for i, c in enumerate(plan["cases"])
    ]
    shots = measured_launches(feeds)
    data = {
        "cards": [c for _, c, _ in schedule],
        "schedule": [{"kind": k, "reply_seed": s} for k, _, s in schedule],
        "rows": [],
        "ready": model["ready"],
        "opening_side": 1,
    }
    with centered(model["ready"]):
        arena = replay_arena([])
        renderer, rig, stride, lo, hi = resources(arena)
        try:
            for _, card, seed in schedule:
                data["rows"].append(
                    trial(
                        arena,
                        renderer,
                        rig,
                        stride,
                        lo,
                        hi,
                        model,
                        shots,
                        card,
                        choose,
                        seed,
                        progress=False,
                    )
                )
            _write(OUT / "inverse-held.json.gz", data)
        finally:
            renderer.close()
            arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "practice", "evaluate"))
    p.add_argument("--feeds", type=Path)
    a = p.parse_args()
    if a.phase == "fit":
        fit_prior()
    elif a.phase == "practice":
        practice(a.feeds)
    else:
        evaluate(a.feeds)
