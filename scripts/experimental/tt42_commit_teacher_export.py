"""Distill scored teacher commands into ordinary RGB and encoder training fragments."""

import argparse
from collections import deque
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt39_turn_time import turn_time
from .tt42_commit_teacher import OUT, PLAN

OPTIC = Path("artifacts/tt37/rebound.npz")


def upcoming_games(doc, plans):
    accepted = {(r["id"], r["seed"], r["reply_seed"]) for r in plans["practice"]}
    excluded = {r["id"] for r in plans["assessment"]}
    for i, (c, row) in enumerate(zip(doc["cards"], doc["rows"], strict=True)):
        if c["id"] in excluded or (c["id"], c["seed"], c["reply_seed"]) not in accepted:
            raise ValueError("Wrong practice cohort for sensory motor export")
        yield i, c, row


def score_clips(events, claims):
    score = 0
    for e in events:
        if e["kind"] != "legal_return" or e["side"] != 1:
            continue
        score += 1
        if score in claims and e["step"] == claims[score]["landing"]["step"]:
            yield score, e


def extract(source, output, protocol):
    d = _read(source)
    if (
        d.get("mode") != "live"
        or not d.get("complete")
        or not d["role"].startswith("training only")
    ):
        raise ValueError(
            "Use the complete training-only physical point, not native search candidates or a fork"
        )
    if output.parent != source.parent:
        raise ValueError("Place the fragment bank with its scored source tape")
    plan = _read(protocol)
    started = process_time()
    rows = []
    tracks = []
    marks = []
    goals = []
    issued = []
    positions = []
    with mpc.centered(d["ready"]):
        a = mpc.arena()
        camera, rig, stride, lo, hi = mpc.resources(a)
        try:
            for i, c, r in upcoming_games(d, plan):
                accepted = {
                    item["scored_before"] + 1: item
                    for item in r["teacher_searches"]
                    if item.get("physical_legal") and item.get("answered")
                }
                if not accepted:
                    continue
                credited = dict(score_clips(r["events"], accepted))
                if set(credited) != set(accepted):
                    raise ValueError(
                        "A committed teacher clip did not receive native legal credit"
                    )
                a.launch(r["shot"], card=c)
                a.partner.reset(seed=c["seed"])
                eye = mpc.Vision(rig, a.control_dt, raw=True)
                eye.eye = mpc.LearnedOptics(eye.eye, OPTIC)
                reference = np.asarray(d["ready"]).copy()
                history = deque(maxlen=150)
                used = 0
                clips = []
                unmatched = 0
                j = 0
                for tick, u in enumerate(r["actions"]):
                    q = a.joints.copy()
                    x = mpc.observed(
                        eye, tick, camera.capture() if tick % 2 == 0 else None, q
                    )
                    sent = np.clip(reference + stride * np.asarray(u), lo, hi)
                    _, _, _, _, info = a.exchange(u, mpc.FAR_COMMAND)
                    history.append((tick, x, q, reference.copy(), sent.copy(), u))
                    reference = sent
                    if any(
                        e["kind"] == "legal_return" and e["side"] == 1
                        for e in info["events"]
                    ):
                        j += 1
                        if j not in credited:
                            continue
                        h = turn_time([(t, p) for t, p, *_ in history])
                        if h is None:
                            unmatched += 1
                            continue
                        ret = accepted[j]
                        take = [
                            (t, p, q, v, g, a)
                            for t, p, q, v, g, a in history
                            if p is not None
                            and p[4] < -0.9
                            and p[1] < -0.62
                            and h - 22 <= t < h - 1
                            and t >= ret["tick"] - 3
                        ]
                        if not take:
                            unmatched += 1
                            continue
                        for t, p, q, v, g, u in take:
                            tracks.append(np.r_[p, q[:5]])
                            marks.append([t, (t - h) * mpc.DT])
                            goals.append(g)
                            issued.append(u)
                            positions.append(v)
                        used += len(take)
                        clips.append(
                            {
                                "score": j,
                                "frames": len(take),
                                "teacher_family": ret["selection"]["family"],
                                "observed_intersection_tick": float(h),
                            }
                        )
                if digest_json(a.events) != digest_json(r["events"]) or digest_json(
                    audit(a.events, opening_side=1)
                ) != digest_json(r["score"]):
                    raise ValueError(
                        "Camera export did not reproduce its physical teacher record"
                    )
                if used:
                    rows.append(
                        {
                            "card": c,
                            "source": source.name,
                            "source_row": i,
                            "frames": used,
                            "clips": clips,
                            "unobservable_credit_events": unmatched,
                            "exact_physical_replay": True,
                        }
                    )
                print(c["id"], len(accepted), len(clips), used, flush=True)
        finally:
            camera.close()
            a.close()
    if not rows:
        raise ValueError(
            "No answered teacher hit with a visible turn; no bank should be exported"
        )
    np.savez_compressed(
        output,
        states=np.array(tracks),
        marks=np.array(marks),
        targets_now=np.array(goals),
        issued_rates=np.array(issued),
        reference_before=np.array(positions),
    )
    _write(
        output.with_suffix(".json"),
        {
            "schema": "tt42-score-earned-visual-fragments-v1",
            "ready": d["ready"],
            "bank_sha256": _hash(output),
            "sources": {source.name: _hash(source)},
            "practice": rows,
            "features": "causal RGB flight estimate and own joint encoders, no native state or future images",
            "labels": "actually issued normalized controls and target references. All selected clips physically landed and were answered. Credit is aligned by camera motion intersection, detected after the action.",
            "native_codes_not_exported": True,
            "time_cpu_s": process_time() - started,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=OUT / "live.json.gz")
    p.add_argument("--output", type=Path, default=OUT / "earned-camera.npz")
    p.add_argument("--protocol", type=Path, default=PLAN)
    a = p.parse_args()
    extract(a.source, a.output, a.protocol)
