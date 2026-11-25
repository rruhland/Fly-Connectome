"""Isolate the causal effect of continued learning inside each complete point with identical prior memory."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried


def replay_memory(arena, tools, priors, card, r, home, optic, index):
    renderer, rig, stride, lo, hi = tools
    arena.launch(r["shot"], card=card)
    arena.partner.reset(seed=card["seed"])
    eye = mpc.Vision(rig, arena.control_dt, raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye, optic)
    ref = np.asarray(home).copy()
    ring = deque(maxlen=150)
    last = 0
    first = None
    for tick, u in enumerate(r["actions"]):
        q = arena.joints
        x = mpc.observed(eye, tick, renderer.capture() if tick % 2 == 0 else None, q)
        sent = np.clip(ref + stride * np.asarray(u), lo, hi)
        _, _, _, _, info = arena.exchange(u, mpc.FAR_COMMAND)
        ring.append((tick, x, q, ref, sent))
        if any(e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]):
            score = mpc.audit(arena.events, opening_side=1)["learner"]
            if score > last:
                priors.keep(list(ring), ("live", index, score))
                if first is None:
                    first = tick
                last = score
        ref = sent
    if digest_json(arena.events) != digest_json(r["events"]):
        raise ValueError("Different live physics in the recorded online point")
    if priors.digest() != r["memory"]["after"]:
        raise ValueError("Reconstructed online learning changed")
    return first


def run(a):
    d = _read(a.source)
    optic = mpc.ROOT / "tt37/rebound.npz"
    start = 0
    before = None
    if getattr(a, "boundary", None) is not None:
        before = _read(a.boundary)
        start = len(before["rows"])
        if before["cards"] != d["cards"] or not 0 < start < len(d["rows"]) <= len(d["cards"]):
            raise ValueError("Expected two consecutive snapshots of the same course")
        if before["rows"][-1]["memory"]["after"] != d["rows"][start]["memory"]["before"]:
            raise ValueError("Course snapshots have different boundary memories")
    cards = d["cards"][start : len(d["rows"])]
    if (
        d["mode"] != "live"
        or d["source_sha256"] != _hash(mpc.BANK)
        or d["optical_sha256"] != _hash(optic)
        or (before is None and len(d["cards"]) != len(d["rows"]))
    ):
        raise ValueError("Expected a complete compatible live course")
    if d["previous_practice"] != {p.name: _hash(p) for p in a.practice}:
        raise ValueError("Wrong pretraining")
    with np.load(mpc.BANK) as z:
        raw = mpc.DeepField(z, "local")
    doc = {
        "cards": cards,
        "ready": d["ready"],
        "source_sha256": _hash(a.source),
        "rows": [],
        "pairs": [],
        "mode": "frozen within each rally; identical prior earned experience restored between rallies",
    }
    cached = _read(a.out) if a.out.exists() else None
    if cached and (cached["source_sha256"] != doc["source_sha256"] or cached["cards"] != doc["cards"]):
        raise ValueError("Partial audit belongs to a different source course")
    with mpc.centered(d["ready"]):
        arena = mpc.arena()
        res = mpc.resources(arena)
        model = mpc.MPC(mpc.BANK, raw, variant="elastic", budget=d["budget"])
        try:
            model.priors = Carried(
                mpc.BANK,
                model.engine,
                practice=a.practice,
                held={c["id"] for c in d["cards"]},
            )
            if before is not None:
                from .tt41_early_optics_carry import resume
                resume(model, a.boundary.with_suffix("").with_suffix(".npz"), before)
            for i, (c, live) in enumerate(zip(cards, d["rows"][start:], strict=True)):
                if model.priors.digest() != live["memory"]["before"]:
                    raise ValueError(
                        "Live and frozen branches differ before first exposure"
                    )
                if cached and i < len(cached["rows"]):
                    frozen = cached["rows"][i]
                    model.serial += 1
                else:
                    frozen = mpc.play(
                        arena,
                        *res,
                        d["ready"],
                        c,
                        live["shot"],
                        raw,
                        model,
                        optic,
                        "frozen",
                        d["cap"],
                    )
                first = replay_memory(
                    arena, res, model.priors, c, live, d["ready"], optic, start + i + 1
                )
                same_until = next(
                    (
                        j
                        for j, (x, y) in enumerate(
                            zip(live["actions"], frozen["actions"])
                        )
                        if x != y
                    ),
                    min(len(live["actions"]), len(frozen["actions"])),
                )
                if first is None:
                    if (
                        live["actions"] != frozen["actions"]
                        or digest_json(live["events"]) != digest_json(frozen["events"])
                    ):
                        raise ValueError(
                            f"Unrewarded game {c['id']}: prefix {same_until}, lengths {len(live['actions'])}/{len(frozen['actions'])}, same events {digest_json(live['events']) == digest_json(frozen['events'])}"
                        )
                elif same_until <= first:
                    raise ValueError("Controllers diverged before learning was enabled")
                pair = {
                    "id": c["id"],
                    "online": live["score"]["learner"],
                    "frozen": frozen["score"]["learner"],
                    "same_action_prefix_ticks": same_until,
                    "first_own_reward_tick": first,
                    "initial_memory": live["memory"]["before"],
                }
                doc["rows"].append(frozen)
                doc["pairs"].append(pair)
                _write(a.out, doc)
                print(
                    i, c["id"], pair["online"], pair["frozen"], same_until, flush=True
                )
        finally:
            res[0].close()
            model.close()
            arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--boundary", type=Path)
    p.add_argument(
        "--practice",
        nargs="+",
        type=Path,
        default=[Path("artifacts/tt37/camera-time-school.npz")],
    )
    run(p.parse_args())
