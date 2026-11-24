"""Counterfactual next-return audit of the existing elastic joint MPC's within-point memory."""

import argparse
from collections import Counter, deque
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write

PRIMARY = mpc.OUT / "joint-mpc-held_replies-frozen.json.gz"
DIAGNOSTIC = mpc.OUT / "joint-mpc-within-frozen.json.gz"
OPTIC = mpc.ROOT / "tt37/rebound.npz"
THRESHOLDS = (2, 5, 10)


def json_hash(records):
    return sha256(dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def vector_hash(records):
    return sha256(np.asarray(records, dtype="<f8").tobytes()).hexdigest()


def own_landings(events):
    return [s["landing"] for s in mpc.audit(events, opening_side=1)["shots"] if s["side"] == 1]


def boundary_for(row, score):
    t = own_landings(row["events"])[score - 1]
    return next(e for e in row["events"] if e["kind"] == "legal_return" and e["time"] == t)


def roster(doc, thresholds=THRESHOLDS):
    """Use one deepest preassigned boundary per distinct eligible physical tape, even if it ends."""
    used = set()
    for i, (c, row) in enumerate(zip(doc["cards"], doc["rows"], strict=True)):
        scores = [s for s in thresholds if s <= row["score"]["learner"]]
        if not scores:
            continue
        n = max(scores)
        event = boundary_for(row, n)
        prefix = [e for e in row["events"] if e["step"] <= event["step"]]
        # Do not use card ID alone: same opening archive can give physically different replies.
        key = json_hash(prefix)
        if key not in used:
            yield i, c, row, n, key
            used.add(key)


def episode(events, score, *, terminal=None):
    own = own_landings(events)
    t = own[score - 1]
    next_events = [e for e in events if e["time"] > t]
    far = next((e for e in next_events if e["kind"] == "contact" and e["side"] == -1), None)
    offer = next((e for e in next_events if e["kind"] == "legal_return" and e["side"] == -1), None)
    touch = next((e for e in next_events if e["kind"] == "contact" and e["side"] == 1), None)
    legal = next((e for e in next_events if e["kind"] == "legal_return" and e["side"] == 1), None)
    final = next((e for e in next_events if e["kind"] in ("point", "timeout", "let")), None)
    # Only the following receiving attempt. Discard later scores or referee results.
    if legal is not None:
        final = None
    return loads(dumps(dict(
        outcome=("legal" if legal else "contact_no_return" if offer and touch else
                 "miss" if offer else "no_legal_offer"),
        far_contact=far,
        offer=offer,
        contact=touch,
        legal=legal,
        terminal=final if final is not None else terminal,
    )))


def recent(frame, x, q, prev, sent):
    return frame, None if x is None else x.copy(), q.copy(), prev.copy(), sent.copy()


def sense_hash(digest, x, q, reference):
    digest.update(np.asarray(q, dtype="<f8").tobytes())
    digest.update(np.asarray(reference, dtype="<f8").tobytes())
    digest.update(b"?" if x is None else np.asarray(x, dtype="<f8").tobytes())


def replay(a, resources, model, raw, optic, ready, serial, card, row, boundary, retain):
    """Run the ordinary controller while forcing only the archived physical prefix.

    Write new rows to an isolated buffer via the deployed rule. The ordinary controller
    cannot read those rows until the chosen handoff. Its own warm state and RNG continue.
    """
    renderer, rig, stride, lower, upper = resources
    a.launch(row["shot"], card=card)
    a.partner.reset(seed=card["seed"])
    eye = mpc.Vision(rig, a.control_dt, raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye, optic)
    model.serial = serial - 1
    model.begin()
    empty = deque(maxlen=model.priors.online.maxlen)
    history = deque(maxlen=empty.maxlen)
    model.priors.online = empty
    archive = deque(maxlen=150)
    reference = np.asarray(ready).copy()
    home = reference.copy()
    past_inbound, previous_reward = False, 0
    handoff, cmds, divergent = None, [], []
    sensory = sha256()
    origin = process_time()
    tolerance = 1e-9
    max_error = 0.
    for tick in range(a.max_steps):
        q = a.joints
        x = mpc.observed(eye, tick, renderer.capture() if tick % 2 == 0 else None, q)
        if handoff is None:
            sense_hash(sensory, x, q, reference)
        if x is not None and x[4] < -.9 and x[1] < -.90:
            past_inbound = True
        desired = home
        if x is not None and (x[4] < -.9 or (past_inbound and x[1] < .50)):
            target, confidence = raw.get(x, q)
            if target is not None:
                desired = home + confidence * (target - home)
        desired = model.step(tick, eye.eye, x, q, reference, desired)
        predicted_u = np.clip((np.clip(desired, lower, upper) - reference) / stride, -.8, .8)
        u = predicted_u
        if tick < len(row["actions"]):
            recorded_u = np.asarray(row["actions"][tick])
            difference = float(np.max(abs(recorded_u - predicted_u)))
            if handoff is None:
                u = recorded_u
                max_error = max(max_error, difference)
                if difference > tolerance:
                    raise RuntimeError(f"Frozen actor failed to reproduce its own tape at tick {tick}, delta={difference}")
            elif difference > tolerance:
                divergent.append(tick)
        elif handoff is None:
            raise RuntimeError("Exhausted the tape before handoff")
        else:
            divergent.append(tick)
        sent = np.clip(reference + u * stride, lower, upper)
        _, _, done, capped, info = a.exchange(u, mpc.FAR_COMMAND)
        cmds.append(u.tolist())
        reference, prev = sent, reference
        near_awarded = any(e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"])
        if handoff is None:
            archive.append(recent(tick, x, q, prev, sent))
            if near_awarded:
                reward = mpc.audit(a.events, opening_side=1)["learner"]
                if reward > previous_reward:
                    model.priors.online = history
                    model.priors.keep(list(archive), ("live", model.serial, reward))
                    memory = model.priors.digest()
                    model.priors.online = empty
                    previous_reward = reward
                    if reward == boundary:
                        prefix = [e for e in row["events"] if e["step"] <= len(cmds)]
                        if json_hash(a.events) != json_hash(prefix):
                            raise RuntimeError(f"Physical referee prefix differs at tick {tick}: {a.events[-1]}, {prefix[-1]}")
                        model.priors.online = history if retain else empty
                        handoff = dict(
                            next_tick=tick + 1,
                            sensor_sha256=sensory.hexdigest(),
                            events_sha256=json_hash(a.events),
                            commands_sha256=vector_hash(cmds),
                            strongest_proposed_command_delta=max_error,
                            reconstructed=memory,
                            available=model.priors.digest(),
                            rng_sha256=json_hash(model.rng.bit_generator.state),
                        )
        elif near_awarded and len(own_landings(a.events)) > boundary:
            break
        if done or capped:
            break
    if handoff is None:
        raise RuntimeError("Did not reach the archived score threshold")
    output = episode(a.events, boundary, terminal="capped" if capped else None)
    return dict(
        handoff=handoff,
        next=output,
        first_different_tick=divergent[0] if divergent else None,
        changed_ticks=len(divergent),
        live_actions=cmds[handoff["next_tick"]:],
        new_searches=[r for r in model.notes if r["tick"] >= handoff["next_tick"]],
        cpu_s=process_time() - origin,
    )


def compare(archived, treatment):
    left, right = archived["outcome"], treatment["outcome"]
    if left == "no_legal_offer" and right == "no_legal_offer":
        result = "both_no_offer"
    elif left == "no_legal_offer" or right == "no_legal_offer":
        result = "one_no_offer"
    else:
        count = int(right == "legal") - int(left == "legal")
        result = {1: "retain_only_legal", 0: "tie_legal" if left == "legal" else "tie_nonlegal",
                  -1: "erase_only_legal"}[count]
    return dict(
        result=result,
        same_nonnull_offer=archived["offer"] is not None and archived["offer"] == treatment["offer"],
        same_nonnull_far_contact=(archived["far_contact"] is not None and
                                  archived["far_contact"] == treatment["far_contact"]),
        contact_delta=int(treatment["contact"] is not None) - int(archived["contact"] is not None),
    )


def run(args):
    source = DIAGNOSTIC if args.supplemental else PRIMARY
    d = _read(source)
    if not (d["mode"] == "frozen" and d["variant"] == "elastic" and all(
        row["memory"]["before"]["rows"] == 0 for row in d["rows"]
    )):
        raise ValueError("Need a frozen elastic tape with empty prepoint online memory")
    if d["source_sha256"] != _hash(mpc.BANK) or d["optical_sha256"] != _hash(args.optic):
        raise ValueError("Prior bank or visual filter differs from archived controller")
    cases = list(roster(d, (2,) if args.supplemental else THRESHOLDS))
    code_hashes = {
        n: _hash(Path(__file__).with_name(n))
        for n in (
            Path(__file__).name, "tt38_joint_mpc.py", "tt37_opponent.py",
            "tt37_optical_clinic.py", "tt29_rally_rollout.cpp", "tt29_rally_physics.py",
        )
    }
    document = dict(
        source=str(source), source_sha256=_hash(source), archival_code_sha256=d["model_source_sha256"],
        running_code_sha256=code_hashes,
        archive_prior_sha256=d["source_sha256"], archive_optic_sha256=d["optical_sha256"],
        protocol="random",
        selection=("Known 4640 counterexample, probe after its second real legal return." if args.supplemental else "Each distinct frozen held tape with >=2 own legal returns, in archive order. "
                   "Take its greatest available threshold among 2, 5, and 10. "
                   "Do not condition on the next offer or chosen action."),
        supplemental_known_counterexample=args.supplemental,
        design=("Force exact archived near action prefix through nth own legal landing. "
                "Compare the next physical receiving attempt after adding within-point reward rows or not. "
                "Memory is frozen after handoff. Scores are not inputs except credit for actual past legal returns."),
        roster=[dict(archive_row=i, card=c, after_own=n, prefix_sha256=key) for i, c, _, n, key in cases],
        rows=[],
    )
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, document)
    with mpc.centered(d["ready"]):
        a = mpc.arena(protocol="random")
        resources = mpc.resources(a)
        model = mpc.MPC(mpc.BANK, raw, variant=d["variant"], budget=d["budget"])
        try:
            for case, (i, c, row, n, key) in enumerate(cases):
                if args.case is not None and args.case != case:
                    continue
                archived = episode(row["events"], n)
                retain = replay(a, resources, model, raw, args.optic, d["ready"], i + 1, c, row, n, True)
                if retain["handoff"]["events_sha256"] != key:
                    raise RuntimeError("Frozen and forced branches disagree on the preassigned events")
                fresh = None
                if args.fresh_control or (case == 0 and not args.archive_only):
                    fresh = replay(a, resources, model, raw, args.optic, d["ready"], i + 1, c, row, n, False)
                    same = {k: v for k, v in fresh["handoff"].items() if k != "available"} == {
                        k: v for k, v in retain["handoff"].items() if k != "available"
                    }
                    if not same or fresh["next"] != archived or fresh["changed_ticks"]:
                        raise RuntimeError("Fresh erased branch failed frozen tape parity")
                comparison = compare(archived, retain["next"])
                document["rows"].append(dict(
                    archive_row=i, after_own=n, card=c,
                    erased_archive=archived, retain=retain, erased_fresh=fresh, pair=comparison,
                ))
                document["pair_counts"] = dict(Counter(r["pair"]["result"] for r in document["rows"]))
                _write(args.output, document)
                print(c["id"], c["reply_seed"], n, archived["outcome"], retain["next"]["outcome"],
                      retain["handoff"]["available"]["rows"], comparison, flush=True)
        finally:
            resources[0].close()
            model.close()
            a.close()
    if code_hashes != {n: _hash(Path(__file__).with_name(n)) for n in code_hashes}:
        print("NOTE: at least one on-disk dependency changed while this process was active", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--optic", type=Path, default=OPTIC)
    p.add_argument("--case", type=int)
    p.add_argument("--supplemental", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fresh-control", action="store_true")
    mode.add_argument("--archive-only", action="store_true")
    run(p.parse_args())
