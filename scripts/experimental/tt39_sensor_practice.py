"""Recover proprioception, RGB and reward from recorded play without simulator contact labels."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_joint_capture import observed
from .tt37_opponent import arena
from .tt37_visual_inertia import LearnedOptics
from .tt38_scored_joints import ScoredJoints
from .tt39_preparation import Preparation


def harvest(source, output, optics, *, fitted=False):
    data = _read(source)
    if len(data["rows"]) != len(data["cards"]):
        raise ValueError("Incomplete game archive")
    if output.parent.resolve() != source.parent.resolve():
        raise ValueError("Keep the movement bank beside its source tape")
    reporter = ScoredJoints(fitted_time=fitted)
    ratio = float(bool(data.get("pupil_rollin")))
    contexts = Preparation(fixed=ratio, learn=False)
    descriptors = []
    all_refs = []
    goals = []
    receipt = []
    with centered(data["ready"]):
        a = arena()
        renderer, rig, stride, lo, hi = resources(a)
        try:
            for card, row in zip(data["cards"], data["rows"], strict=True):
                a.launch(row["shot"], card=card)
                a.partner.reset(seed=card["seed"])
                vision = Vision(rig, a.control_dt, raw=True)
                vision.eye = LearnedOptics(vision.eye, optics)
                reporter.begin_point()
                contexts.begin()
                reference = np.array(data["ready"])
                incoming = []
                extracted = local_wins = 0
                for tick, u in enumerate(row["actions"]):
                    q = a.joints
                    state = observed(
                        vision, tick, renderer.capture() if tick % 2 == 0 else None, q
                    )
                    contexts.observe(state, q)
                    reference = np.clip(reference + stride * np.asarray(u), lo, hi)
                    reporter.step(tick, state, q, reference)
                    if state is not None and state[4] < -0.9:
                        if incoming and tick - incoming[-1][0] > 22:
                            incoming = []
                        incoming.append((tick, np.r_[state, q[:5]], reference.copy()))
                    _, _, done, _, info = a.exchange(u, FAR_COMMAND)
                    scored = any(
                        e["kind"] == "legal_return" and e["side"] == 1
                        for e in info["events"]
                    )
                    contexts.feedback(scored, done)
                    if scored:
                        reporter.credit()
                    if scored or done:
                        if len(incoming) > 6:
                            ts = np.array([e[0] for e in incoming])
                            for i, (t, desc, ref) in enumerate(incoming[:-4]):
                                # A future actually issued command can supervise a lead motor policy.
                                # All current and future frames here are on the same observed approach.
                                j = int(np.searchsorted(ts, t + 6))
                                if j >= len(ts) or ts[j] - t > 8:
                                    continue
                                descriptors.append(desc)
                                all_refs.append(np.r_[ref, incoming[j][2]])
                                goals.append(int(scored))
                                extracted += 1
                            local_wins += int(scored)
                        incoming = []
                if digest_json(a.events) != digest_json(row["events"]):
                    raise ValueError("Replayed scores or event history changed")
                reporter.complete(card)
                receipt.append(
                    {"card": card, "frames": extracted, "converted": local_wins}
                )
                print(
                    card["id"],
                    row["score"]["learner"],
                    reporter.strikes,
                    extracted,
                    flush=True,
                )
        finally:
            renderer.close()
            a.close()
    reporter.save(output, data["ready"], -1.56, source)
    alt = output.with_name(output.stem + "-outcomes.npz")
    np.savez_compressed(
        alt,
        states=np.asarray(descriptors),
        references=np.asarray(all_refs),
        reward=np.asarray(goals, dtype=np.uint8),
    )
    _write(
        output.with_name(output.stem + "-outcomes.json"),
        {
            "source": source.name,
            "sha256": _hash(source),
            "data_sha256": _hash(alt),
            "optics_sha256": _hash(optics),
            "practice": receipt,
            "examples": contexts.new,
            "practice_ids": [c["id"] for c in data["cards"]],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--optics", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    p.add_argument("--fitted-time", action="store_true")
    a = p.parse_args()
    harvest(a.source, a.output, a.optics, fitted=a.fitted_time)
