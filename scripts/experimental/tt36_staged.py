"""Retain a learned reach and tune a second joint correction as the ball approaches the visible blade."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt36_local as local
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt33_joint import resources
from .tt34_latency import Map
from .tt34_translate import LearnedBody
from .tt36_course import OUT
from .tt36_inverse import candidate, target
from .tt36_local import BASE, IDS, SCALE

APPROACH = local.Approach


def body_model():
    with np.load(OUT.parent / "tt34/lag-bank.npz") as bank:
        body = LearnedBody(bank)
    return Map(body.q, body.p, body.scale)


def movement(body, choose_late, include_reach=True):
    class NearBlade(APPROACH):
        def update(self, x, q):
            early = super().update(x, q)
            if self.current is None or x is None:
                return early if include_reach else None
            late = choose_late(np.array(self.picks[-1]["observation"]))
            distance = np.linalg.norm(x[:3] - body(q[:5]))
            raw = np.clip((0.58 - distance) / 0.32, 0.0, 1.0)
            gain = raw**2 * (3.0 - 2.0 * raw)
            return (self.current if include_reach else 0.0) + gain * late

    return patch.object(local, "Approach", NearBlade)


def choose_from(state, mode):
    centers = np.array([s["context"] for s in state])
    actions = np.array([s[mode] for s in state])

    def choose(x):
        distances = np.linalg.norm((centers - x) / SCALE, axis=1)
        i = int(np.argmin(distances))
        return (
            actions[i] * np.exp(-0.5 * (distances[i] / 1.25) ** 2)
            if distances[i] < 2.35
            else np.zeros(5)
        )

    return choose


def practice(feeds):
    body = body_model()
    previous = _read(OUT / "local-checkpoint.json")
    early = local.choose_from(previous, "dense")
    prior = _read(OUT / "inverse-prior.json")
    plan = _read(OUT / "local-plan.json.gz")
    model = _read(BASE)
    shots = measured_launches(feeds)
    # Choose two joint-space directions by their observed effects on outgoing velocity.
    axes = np.linalg.svd(np.array(prior["velocity_jacobian"]), full_matrices=False)[2][
        :2
    ]
    probes = [
        np.zeros(5),
        0.30 * axes[0],
        -0.30 * axes[0],
        0.30 * axes[1],
        -0.30 * axes[1],
    ]
    records = []
    states = []
    with centered(model["ready"]):
        arena = local.replay_arena([])
        renderer, rig, stride, lo, hi = resources(arena)
        try:
            for k, case in enumerate(plan["cases"]):
                episodes = []
                ctx = previous["observations"][k]["context"]
                goal = target(prior, ctx)
                for i in range(8):
                    offset = (
                        probes[i] if i < 5 else candidate(episodes, prior, goal, i - 5)
                    )
                    with movement(body, lambda _, action=offset: action):
                        row = local.trial(
                            arena,
                            renderer,
                            rig,
                            stride,
                            lo,
                            hi,
                            model,
                            shots,
                            case["original"],
                            early,
                            366010 + 31 * k + i,
                        )
                    episodes.append({"offset_rad": offset.tolist(), "row": row})
                winners = {
                    m: local.winner(episodes, m)["offset_rad"]
                    for m in ("dense", "score")
                }
                states.append({"context": ctx, **winners})
                records.append({"id": IDS[k], "episodes": episodes})
                _write(
                    OUT / "staged-practice.json.gz",
                    {
                        "cases": records,
                        "basis": axes.tolist(),
                        "source_sha256": _hash(OUT / "local-checkpoint.json"),
                    },
                )
                print(
                    IDS[k], [e["row"]["score"]["learner"] for e in episodes], flush=True
                )
        finally:
            renderer.close()
            arena.close()
    _write(
        OUT / "staged-checkpoint.json",
        {"states": states, "practice_sha256": _hash(OUT / "staged-practice.json.gz")},
    )


def held(feeds, mode):
    body = body_model()
    early = local.choose_from(_read(OUT / "local-checkpoint.json"), "dense")
    states = _read(OUT / "staged-checkpoint.json")["states"]
    choose = choose_from(states, "score" if mode == "score" else "dense")
    plan = _read(OUT / "local-plan.json.gz")
    model = _read(BASE)
    shots = measured_launches(feeds)
    schedule = [
        ("reply", c["original"], s) for c in plan["cases"] for s in c["reply_seeds"]
    ]
    schedule += [
        ("first", c["changed"], 365061 + i) for i, c in enumerate(plan["cases"])
    ]
    result = {
        "cards": [c for _, c, _ in schedule],
        "schedule": [{"kind": k, "reply_seed": s} for k, _, s in schedule],
        "mode": mode,
        "ready": model["ready"],
        "rows": [],
        "opening_side": 1,
    }
    with (
        centered(model["ready"]),
        movement(body, choose, include_reach=mode != "late-only"),
    ):
        arena = local.replay_arena([])
        renderer, rig, stride, lo, hi = resources(arena)
        try:
            for _, c, seed in schedule:
                row = local.trial(
                    arena,
                    renderer,
                    rig,
                    stride,
                    lo,
                    hi,
                    model,
                    shots,
                    c,
                    early,
                    seed,
                    progress=False,
                )
                result["rows"].append(row)
            _write(OUT / f"staged-held-{mode}.json.gz", result)
        finally:
            renderer.close()
            arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("practice", "dense", "score", "late-only"))
    p.add_argument("feeds", type=Path)
    args = p.parse_args()
    if args.mode == "practice":
        practice(args.feeds)
    else:
        held(args.feeds, args.mode)
