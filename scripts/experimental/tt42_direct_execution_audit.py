"""Verify the frozen execution comparison and separate first-offer from later effects."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_report import paired, read_complete
from .tt38_joint_mpc_report import metrics
from .tt42_direct_execution import CALIBRATION, FIT, HELD, STOCK, course
from .tt42_feedback_verification import action_prefix, offer_contrast, verify_checkpoint, verify_row


PER_GAME = ("incumbent_inspections", "incumbent_inspection_failed", "preserved_due_calls",
            "invalidated_motion", "early_replans", "installed_forecast")
MOTOR_LIMITS = ("target_rate_limit_rad_s", "no_load_reference_rad_s", "linear_motoring_envelope",
                "force_limit_nm")


def report(stock, execution, out, *, partial=False):
    a = read_complete(stock)
    b = _read(execution) if partial else read_complete(execution)
    context = ("cards", "mode", "deck", "ready", "cap", "variant", "budget", "source_sha256",
               "previous_practice", "plan_sha256", "optical_sha256", "carry_sha256")
    if any(a[k] != b[k] for k in context) or b["mode"] != "frozen" or b["deck"] != "assessment":
        raise ValueError("Execution comparison changed its observation, motor memory or held context")
    roster = _read(course.PLAN)["assessment"]
    fit = _read(FIT)
    if (a["cards"] != roster or a["plan_sha256"] != _hash(course.PLAN)
            or fit["train_sha256"] != _hash(CALIBRATION)
            or {r["id"] for r in fit["rows"]} & {c["id"] for c in roster}):
        raise ValueError("Archived assignment or release-disjoint execution fit changed")
    n = len(b["rows"])
    if not 0 < n <= len(a["cards"]):
        raise ValueError("Expected a non-empty prefix of the fixed assessment deck")
    complete = n == len(a["cards"])
    protocol = execution.with_suffix("").with_suffix(".protocol.json")
    sources = _read(protocol)
    fit_hash = _hash(FIT)
    if (sources[FIT.name] != fit_hash or (complete and not partial
            and (b["execution_source_sha256"] != sources["tt42_direct_execution.py"]
                 or b["fit_sha256"] != fit_hash))):
        raise ValueError("Execution and calibration protocol receipt changed")
    memory = {}
    for name, path, doc in (("stock", stock, a), ("execution", execution, b)):
        memory[name] = verify_checkpoint(path.with_suffix("").with_suffix(".npz"), doc["rows"],
                                         doc["checkpoint_sha256"])
    first = a["rows"][0]["memory"]["before"]
    if any({k: m[k] for k in first} != first for m in memory.values()):
        raise ValueError("Assessment checkpoint does not reproduce the original frozen memory")
    a = dict(a, cards=a["cards"][:n], rows=a["rows"][:n])
    b = dict(b, cards=b["cards"][:n])
    p = paired(a, b)
    for detail, card, left, right in zip(p["rows"], a["cards"], a["rows"], b["rows"], strict=True):
        for row in (left, right):
            verify_row(card, row)
            if row["memory"] != dict(before=first, after=first):
                raise ValueError("Frozen motor memory changed within or between paired games")
        if left["shot"] != right["shot"]:
            raise ValueError("Paired physical launch changed")
        if any(left["motor_audit"][k] != right["motor_audit"][k] for k in MOTOR_LIMITS):
            raise ValueError("Physical actuator limits changed between the arms")
        fork, prefix = action_prefix(left, right)
        contact = offer_contrast(left, right, fork)
        lt, rt = ({s["tick"] for s in row["searches"] if fork is None or s["tick"] <= fork}
                  for row in (left, right))
        detail.update(first_action_change_tick=fork, exact_prefix_events=len(prefix),
                      identical_launch=True, first_changed_offer=contact,
                      first_stock_only_search_tick=min(lt - rt, default=None),
                      first_execution_only_search_tick=min(rt - lt, default=None),
                      adaptive_cadence={k: right["planner"].get(k, 0) for k in PER_GAME})
    matched_offers = [x for r in p["rows"] if (x := r["first_changed_offer"]) and x["same_incoming"]]
    m = dict(n=len(matched_offers), touched=[sum(o["touches"][k] for o in matched_offers) for k in (0, 1)],
             legal=[sum(o["landed"][k] for o in matched_offers) for k in (0, 1)],
             changed_leg=sum(o["landed"][0] != o["landed"][1] for o in matched_offers))
    _write(out, dict(complete=complete, n=n, held_roster=len(roster),
                     stock=metrics(a), execution=metrics(b), paired=p, matched_first_changed_offers=m,
                     checkpoints=memory, source_sha256=dict(stock=_hash(stock), execution=_hash(execution),
                                                           protocol=_hash(protocol), fit=fit_hash)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, default=STOCK)
    parser.add_argument("--execution", type=Path, default=HELD)
    parser.add_argument("--output", type=Path, default=course.ROOT / "direct-execution-audit.json")
    parser.add_argument("--partial", action="store_true")
    a = parser.parse_args()
    report(a.stock, a.execution, a.output, partial=a.partial)
