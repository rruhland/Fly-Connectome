"""Isolate late receiving failures by branching from archived physical action prefixes."""

import argparse
from math import ceil
from pathlib import Path
from time import process_time

import numpy as np

from . import tt42_direct_execution as execution
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_joint_report import read_complete


mpc = execution.mpc
REFERENCE = Path("artifacts/tt39/commitment-replication-stock.json.gz")
OUT = Path("artifacts/tt42/direct-execution-late.json.gz")


def failures(doc):
    found = {}
    for card, row in zip(doc["cards"], doc["rows"], strict=True):
        offered = row["placements"]["delivered"]
        if row["score"]["learner"] < 4 or not offered:
            continue
        last = offered[-1]
        if last["near_landed"]:
            continue
        after_far = ceil(last["far_contact"]["time"] / mpc.DT - 1e-8)
        selected = any(s.get("chosen") for s in row["searches"] if s["tick"] >= after_far)
        contact = last["near_contact"] is not None
        category = {(True, True): "selected_contact_fault", (True, False): "selected_no_contact",
                    (False, False): "unsupported_no_contact"}.get((selected, contact))
        if category and category not in found:
            found[category] = dict(card=card, row=row, fork_tick=after_far)
    if len(found) != 3:
        raise ValueError("The completed source must contain all three physical failure types")
    return found


def branch(arena, renderer, rig, stride, low, high, ready, card, row, fork_tick, raw, actor):
    arena.launch(row["shot"], card=card)
    arena.partner.reset(seed=card["seed"])
    eye = mpc.Vision(rig, arena.control_dt, raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye, execution.course.OPTIC)
    actor.begin()
    reference = np.asarray(ready).copy()
    home = reference.copy()
    began = max(0, fork_tick - 64)
    sensed_inbound = False
    started = process_time()
    prefix = None
    added = []
    for tick in range(arena.max_steps):
        q = arena.joints
        x = mpc.observed(eye, tick, renderer.capture() if tick >= began and tick % 2 == 0 else None, q)
        if x is not None and x[4] < -.9 and x[1] < -.90:
            sensed_inbound = True
        if tick < fork_tick:
            action = np.asarray(row["actions"][tick])
            actor.engine.observe(tick, eye.eye, q, reference.copy())
        else:
            if prefix is None:
                events = [e for e in row["events"] if e["time"] <= tick * mpc.DT + 1e-8]
                signature = digest_json(arena.events)
                if signature != digest_json(events):
                    raise ValueError("The real prefix did not reach the original terminal offer")
                prefix = dict(sha256=signature, count=len(events), q=q.tolist(), reference=reference.tolist())
            goal = home
            if x is not None and (x[4] < -.9 or (sensed_inbound and x[1] < .5)):
                target, confidence = raw.get(x, q)
                if target is not None:
                    goal = home + confidence * (target - home)
            desired = actor.step(tick, eye.eye, x, q, reference, goal)
            action = np.clip((np.clip(desired, low, high) - reference) / stride, -.8, .8)
            added.append(action.tolist())
        _, _, done, capped, info = arena.exchange(action, mpc.FAR_COMMAND)
        reference = np.clip(reference + action * stride, low, high)
        if done or capped or (tick >= fork_tick and any(e["kind"] == "legal_return" and e["side"] == 1
                                                      for e in info["events"])):
            break
    if prefix is None:
        raise ValueError("Point ended before the branch")
    own = audit(arena.events, opening_side=1)["learner"]
    return dict(prefix=prefix, rescued=own > row["score"]["learner"], own=own,
                post_events=arena.events[prefix["count"]:], actions=added, searches=actor.notes,
                planner=dict(actor.diag), runtime_cpu_s=process_time() - started)


def run():
    doc = read_complete(REFERENCE)
    fit = _read(execution.FIT)
    if doc["source_sha256"] != _hash(mpc.BANK) or doc["optical_sha256"] != _hash(execution.course.OPTIC):
        raise ValueError("Older physical trial bank or optics changed")

    class Calibrated(execution.Execution):
        scales = np.asarray(fit["scales"])

    with np.load(mpc.BANK) as data:
        raw = mpc.DeepField(data, "local")
    results = []
    with mpc.centered(doc["ready"]):
        arena = mpc.arena()
        renderer, rig, stride, low, high = mpc.resources(arena)
        actors = {mode: cls(mpc.BANK, raw, variant="elastic", budget=doc["budget"])
                  for mode, cls in (("stock", execution.BASE), ("adaptive", Calibrated))}
        try:
            for category, case in failures(doc).items():
                card, row, fork = case["card"], case["row"], case["fork_tick"]
                branches = {k: branch(arena, renderer, rig, stride, low, high, doc["ready"], card, row, fork,
                                      raw, actor) for k, actor in actors.items()}
                if branches["stock"]["prefix"] != branches["adaptive"]["prefix"]:
                    raise ValueError("The two branches did not see an identical physical prefix")
                results.append(dict(category=category, card=card, fork_tick=fork,
                                    stock_own_before_fork=row["score"]["learner"], branches=branches))
                _write(OUT, dict(rows=results, source_sha256=_hash(REFERENCE), fit_sha256=_hash(execution.FIT),
                                 boundary="Earlier archived physical commands only. The branch mark selects a known "
                                          "failure for this diagnostic, not an input of the held actor. "
                                          "Both controllers draw fresh equal search seeds at the same prefix."))
                print(category, card["id"], {k: v["rescued"] for k, v in branches.items()}, flush=True)
        finally:
            for a in actors.values():
                a.close()
            renderer.close()
            arena.close()


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
