"""Swap only the earned bank after an exact archived far-racket separation."""

import argparse
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt51_transfer_after_point as source
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_joint_report import read_complete
from .tt42_feedback_verification import action_prefix, verify_checkpoint, verify_row


course = source.course
mpc = course.mpc
season = source.season
ROOT = Path("artifacts/tt52_terminal_bank_crossplay")
PLAN = ROOT / "protocol.json"
TAPE = ROOT / "actions.json.gz"
REPORT = ROOT / "comparison.json"
ARMS = ("after", "before")
GAMES = 8
ASSESS = {k: source.ROOT / f"{k}-held.json.gz" for k in ARMS}
BANKS = {k: source.ROOT / f"after-{g:02}.json.gz" for k, g in zip(ARMS, (10, 9), strict=True)}
OTHER = dict(after="before", before="after")


def fork(row):
    delivered = row["placements"]["delivered"]
    if not delivered:
        return dict(role="no_native_offer", cut_tick=None)
    offer = delivered[-1]
    if offer["near_landed"] or not offer["near_bounce"] or row["events"][-1]["kind"] != "point":
        return dict(role="no_failed_bounced_terminal_offer", cut_tick=None)
    # The natural separation is a stronger cut than first contact. It keeps the
    # full far-racket impulse and all outbound return preparation identical.
    release = next((e for e in row["events"] if e.get("collider") == "rival_racket"
                    and offer["far_contact"]["time"] <= e["time"] < offer["near_bounce"]["time"]), None)
    if release is None:
        return dict(role="no_recorded_far_separation", cut_tick=None)
    tick = int(math.ceil(release["time"] / mpc.DT - 1e-7))
    if not 0 < tick < offer["near_bounce"]["step"] or tick >= len(row["actions"]):
        raise ValueError("Invalid terminal fork time")
    return dict(role="terminal_offer", cut_tick=tick, leg=offer["leg"],
                far_separation=release, original_offer=offer)


def load():
    stocks = {k: read_complete(p) for k, p in ASSESS.items()}
    meta = {k: read_complete(p) for k, p in BANKS.items()}
    nums = {k: source.history.load_rows(p) for k, p in BANKS.items()}
    all_cards = _read(source.PLAN)["assessment"]
    comparable = ("deck", "ready", "cap", "variant", "budget", "source_sha256", "previous_practice",
                  "plan_sha256", "optical_sha256")
    if any(stocks["before"][k] != stocks["after"][k] for k in comparable):
        raise ValueError("Assessments do not share their ordinary actor and physical protocol")
    for arm in ARMS:
        run, bank = stocks[arm], meta[arm]
        if (run["mode"] != "frozen" or run["cards"] != all_cards or run["carry_sha256"] != _hash(BANKS[arm])
                or run["source_sha256"] != _hash(mpc.BANK) or run["optical_sha256"] != _hash(course.OPTIC)
                or run["plan_sha256"] != _hash(source.PLAN)
                or bank["exact_source"] != _hash(source.SOURCE) or len(nums[arm]) != 768):
            raise ValueError("Changed archived trajectory or donor-bank provenance")
        stats = verify_checkpoint(BANKS[arm].with_suffix("").with_suffix(".npz"),
                                  [None] * bank["parent_serial"] + bank["rows"], bank["checkpoint_sha256"])
        if any(r["memory"]["before"] != r["memory"]["after"] or
               r["memory"]["before"] != {k: stats[k] for k in ("rows", "state_sha256")} for r in run["rows"]):
            raise ValueError("Assessment did not use the exact frozen bank")
    roster = []
    for i, card in enumerate(all_cards[:GAMES]):
        for arm in ARMS:
            row = stocks[arm]["rows"][i]
            verify_row(card, row)
            roster.append(dict(index=i, card=card, origin=arm, donor=OTHER[arm], **fork(row)))
    paths = (source.PLAN, source.SOURCE, *ASSESS.values(), *BANKS.values(),
             *(p.with_suffix("").with_suffix(".npz") for p in BANKS.values()), mpc.BANK, course.PRACTICE,
             course.OPTIC, season.fresh.RISK, season.fresh.servo.ROOT / "prior.npz")
    code = (__file__, source.__file__, mpc.__file__, season.__file__, season.fresh.__file__,
            season.fresh.servo.__file__, season.fresh.shot.__file__,
            Path(mpc.__file__).with_name("tt45_rollout_reuse.py"),
            Path(mpc.__file__).with_name("tt29_rally_rollout.cpp"))
    protocol = dict(roster=roster, assets={str(p): _hash(p) for p in paths},
                    code={Path(p).name: _hash(Path(p)) for p in code},
                    rule="First eight paired assignments by original sequence, both directions, no replacement. "
                         "The factual archive sets the harness intervention time, after far separation. "
                         "Run the ordinary actor with full RGB from launch and verify every prefix command. "
                         "Only its replay-bank rows change. Use no cut marker or evaluator state in the actor's decisions.")
    return protocol, stocks, meta, nums


class Recorder(season.Player):
    def begin(self):
        super().begin()
        self.installed = []

    def search(self, tick, x, q, reference, base):
        n = len(self.notes)
        super().search(tick, x, q, reference, base)
        if tick >= self.record_at and len(self.notes) != n:
            note = self.notes[-1]
            if note.get("chosen") or note.get("prepare"):
                self.installed.append(dict(tick=tick, chosen=note.get("chosen"), prepare=note.get("prepare"),
                                           robust=note.get("robust"), lease=int(self.lease),
                                           goals=np.asarray(self.best).tolist()))


def branch(arena, resources, actor, case, stock, donor_rows, donor_digest, ready, cfg, *, swap):
    old, cut = stock, case["cut_tick"]
    actor.record_at = cut if cut is not None else len(old["actions"])
    actor.priors.online.clear()
    actor.priors.online.extend(donor_rows[case["origin"]])
    if actor.priors.digest() != old["memory"]["before"]:
        raise ValueError("Prefix does not start from original bank")
    witnessed = {}
    regular, exchange = actor.step, arena.exchange
    command_index = 0

    def step(tick, eye, x, q, reference, base):
        if tick == cut:
            expected = [e for e in old["events"] if e["time"] <= tick * mpc.DT + 1e-8]
            if digest_json(expected) != digest_json(arena.events) or case["far_separation"] not in expected:
                raise ValueError("Far separation or native events diverged before the bank swap")
            # Evaluation-only snapshot. No field from this dictionary is used by the actor.
            blade = arena._racket_ids[1]
            witnessed.update(tick=tick, events=len(expected), events_sha256=digest_json(expected),
                             prefix_sha256=digest_json(old["actions"][:tick]),
                             body=dict(q=q.tolist(), reference=reference.tolist(),
                                       blade_xyz=arena.data.geom_xpos[blade].tolist()),
                             native_ball=dict(p=arena.ball.tolist(), v=arena.velocity.tolist(), spin=arena.spin.tolist()),
                             detected=None if x is None else x.tolist(),
                             rng_sha256=digest_json(actor.rng.bit_generator.state),
                             incumbent_tick=int(actor.tick), next_search=int(actor.next),
                             incumbent_sha256=digest_json(None if actor.best is None else actor.best.tolist()))
            if swap:
                actor.priors.online.clear()
                actor.priors.online.extend(donor_rows[case["donor"]])
                if actor.priors.digest() != donor_digest:
                    raise ValueError("Did not install the exact donor rows")
        return regular(tick, eye, x, q, reference, base)

    def native(near, far):
        nonlocal command_index
        if cut is None or not swap or command_index < cut:
            if command_index >= len(old["actions"]) or near.tolist() != old["actions"][command_index]:
                raise ValueError(f"Fresh actor command differs from archive at tick {command_index}")
        command_index += 1
        return exchange(near, far)

    image, rig, stride, lo, hi = resources
    with patch.object(actor, "step", step), patch.object(arena, "exchange", native):
        result = mpc.play(arena, image, rig, stride, lo, hi, ready, case["card"], old["shot"], actor.raw,
                          actor, course.OPTIC, "frozen", cfg["cap"])
    verify_row(case["card"], result)
    diff, event_prefix = action_prefix(old, result)
    if (cut is not None and not witnessed) or (not swap and
            (diff is not None or digest_json(old["events"]) != digest_json(result["events"]))):
        raise ValueError("Exact factual replay did not reach the fixed cut and terminal outcome")
    if result["memory"]["after"] != (donor_digest if swap else old["memory"]["after"]):
        raise ValueError("Frozen continuation changed the donor's actual memory")
    details = dict(witness=witnessed, first_changed_command=diff, exact_event_prefix=len(event_prefix),
                   accepted_direct=actor.installed, searched=[n for n in result["searches"] if n["tick"] >= actor.record_at],
                   final_event=result["events"][-1], total_own=result["score"]["learner"],
                   total_alternating=result["score"]["returns"], runtime_cpu_s=result["runtime_cpu_s"],
                   commands=result["actions"][cut or 0:], events_after_cut=result["events"][witnessed.get("events", 0):],
                   action_sha256=digest_json(result["actions"]), event_sha256=digest_json(result["events"]),
                   memory=result["memory"])
    if cut is not None:
        offer = next((o for o in result["placements"]["delivered"] if o["leg"] == case["leg"]), None)
        old_offer = case["original_offer"]
        if offer is None or any(digest_json(offer[k]) != digest_json(old_offer[k]) for k in ("far_contact", "target")):
            raise ValueError("The already-issued terminal far impact or placement draw changed")
        same = digest_json(offer["near_bounce"]) == digest_json(old_offer["near_bounce"])
        details.update(offer=offer, same_native_near_bounce=same,
                       same_receive_line=digest_json(offer["receive_line"]) == digest_json(old_offer["receive_line"]),
                       rescue=bool(offer["near_landed"]) if same else None,
                       terminal_offer=(result["placements"]["delivered"] or [None])[-1])
        if same and (result["score"]["learner"] > old["score"]["learner"]) != details["rescue"]:
            raise ValueError("First offered return does not explain the extra legal near count")
    return details


def run():
    protocol, stocks, meta, nums = load()
    ROOT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if protocol != _read(PLAN):
            raise ValueError("Cannot continue with changed archived conditions")
    else:
        _write(PLAN, protocol)
    saved = _read(TAPE) if TAPE.exists() else dict(protocol_sha256=_hash(PLAN), rows=[], complete=False)
    if saved["protocol_sha256"] != _hash(PLAN) or any(
            r["case"] != c for r, c in zip(saved["rows"], protocol["roster"])):
        raise ValueError("Stored trials do not follow the predeclared order")
    ready = stocks["before"]["ready"]
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    with mpc.centered(ready):
        arena = mpc.arena()
        resources = mpc.resources(arena)
        ids = {c["card"]["id"] for c in protocol["roster"]}
        with patch.object(mpc, "Fragments", lambda p, e: course.Carried(p, e, practice=[course.PRACTICE], held=ids)):
            actor = Recorder(mpc.BANK, raw, variant=stocks["before"]["variant"], budget=stocks["before"]["budget"],
                             coverage=season.fresh.shot.Coverage(_read(season.fresh.RISK)))
        try:
            for case in protocol["roster"][len(saved["rows"]):]:
                arm, donor, i = case["origin"], case["donor"], case["index"]
                stock = stocks[arm]["rows"][i]
                digest = meta[donor]["rows"][-1]["memory"]["after"]
                swap = case["cut_tick"] is not None
                actor.serial = meta[arm]["parent_serial"] + len(meta[arm]["rows"]) + i
                altered = branch(arena, resources, actor, case, stock, nums, digest, ready, stocks[arm], swap=swap)
                matched = None
                if altered.get("rescue"):
                    # A fresh no-swap counterpart rules out a hidden reconstruction
                    # error in a positive claim. Both roll through a natural endpoint.
                    actor.serial = meta[arm]["parent_serial"] + len(meta[arm]["rows"]) + i
                    matched = branch(arena, resources, actor, case, stock, nums, digest, ready, stocks[arm], swap=False)
                    if matched["witness"] != altered["witness"] or matched.get("rescue"):
                        raise ValueError("A claimed rescue is not isolated against the exact fresh factual arm")
                saved["rows"].append(dict(case=case, factual_own=stock["score"]["learner"],
                                          counterfactual=altered, fresh_factual=matched))
                saved["complete"] = len(saved["rows"]) == len(protocol["roster"])
                _write(TAPE, saved)
                print(i + 1, case["card"]["id"], f"{arm} <- {donor}", case["role"],
                      stock["score"]["learner"], altered["total_own"], altered.get("rescue"),
                      altered["first_changed_command"], flush=True)
        finally:
            actor.close()
            resources[0].close()
            arena.close()
    report(saved)


def report(data):
    protocol, stocks, _, _ = load()
    if protocol != _read(PLAN) or data["protocol_sha256"] != _hash(PLAN):
        raise ValueError("Archived input or output receipt changed")
    rows = []
    for record, case in zip(data["rows"], protocol["roster"]):
        if record["case"] != case:
            raise ValueError("Case order changed")
        trial = record["counterfactual"]
        old = stocks[case["origin"]]["rows"][case["index"]]
        if (digest_json(old["actions"][:case["cut_tick"] or 0] + trial["commands"]) != trial["action_sha256"]
                or digest_json(old["events"][:trial["witness"].get("events", 0)] + trial["events_after_cut"])
                != trial["event_sha256"]):
            raise ValueError("The saved closed-loop trajectory does not match its full event/command hashes")
        rows.append(dict(index=case["index"], id=case["card"]["id"], origin=case["origin"], donor=case["donor"],
                         role=case["role"], cut_tick=case["cut_tick"], first_changed_command=trial["first_changed_command"],
                         exact_event_prefix=trial["exact_event_prefix"], bounced_exact=trial.get("same_native_near_bounce"),
                         same_receive_line=trial.get("same_receive_line"), rescue=trial.get("rescue"),
                         own=[record["factual_own"], trial["total_own"]], fresh_factual=record["fresh_factual"] is not None,
                         original_touched=bool(case.get("original_offer", {}).get("near_contact")),
                         observed_touched=bool(trial.get("offer", {}).get("near_contact")),
                         accepted=len(trial["accepted_direct"]), terminal=trial["final_event"]))
    stats = {}
    for arm in ARMS:
        r = [v for v in rows if v["origin"] == arm]
        stats[arm] = dict(assignments=len(r), absent=sum(v["cut_tick"] is None for v in r),
                          same_incoming=sum(v["bounced_exact"] is True for v in r),
                          rescued=sum(v["rescue"] is True for v in r),
                          changed_commands=sum(v["first_changed_command"] is not None for v in r))
    _write(REPORT, dict(complete=data["complete"], stats=stats, rows=rows,
                        trajectory_sha256=_hash(TAPE), protocol_sha256=_hash(PLAN)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("verify", "run", "report"))
    a = p.parse_args()
    if a.phase == "verify":
        print([(r["card"]["id"], r["origin"], r["cut_tick"]) for r in load()[0]["roster"]])
    elif a.phase == "run":
        run()
    else:
        report(_read(TAPE))
