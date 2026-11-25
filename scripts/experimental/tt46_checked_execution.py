"""Add sensor-checked commitment to the unchanged TT46 season player."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import tt42_direct_execution as execution
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_joint_report import paired, read_complete
from .tt38_joint_mpc_report import metrics
from .tt42_direct_execution_audit import MOTOR_LIMITS, PER_GAME
from .tt42_feedback_verification import action_prefix, offer_contrast, verify_checkpoint, verify_row


ROOT = Path("artifacts/tt46_checked_execution")
STOCK = season.ROOT / "before.json.gz"
PROTOCOL = ROOT / "protocol.json"
CONTROL = ROOT / "fresh-control.json.gz"
TREATMENT = ROOT / "checked.json.gz"
OUT = ROOT / "comparison.json"
# The first assigned cycle contains every held release exactly once. The next
# two reply streams per release are untouched by this comparison.
N = 8
course = season.course
fresh = season.fresh


class CheckedCadence(season.Player, execution.Execution):
    """Temporal.step adjusts only the fallback, then calls Execution.step."""


def protocol(feeds):
    base = read_complete(STOCK)
    fit = _read(execution.FIT)
    temporal = fresh.servo.ROOT / "prior.npz"
    cards = _read(season.PLAN)["assessment"]
    disjoint = {r["id"] for r in fit["rows"]}
    prior = _read(course.PLAN)
    disjoint.update(c["id"] for deck in ("practice", "assessment") for c in prior[deck])
    if (base["cards"] != cards or len({c["id"] for c in cards[:N]}) != N
            or {c["id"] for c in cards} & disjoint or base["cap"] != 240
            or base["mode"] != "frozen" or base["deck"] != "assessment"
            or base["source_sha256"] != _hash(course.mpc.BANK)
            or base["optical_sha256"] != _hash(course.OPTIC)
            or base["carry_sha256"] != _hash(season.CARRY)
            or base["plan_sha256"] != _hash(season.PLAN)
            or base["temporal"] != _hash(temporal) or base["risk"] != _hash(fresh.RISK)):
        raise ValueError("Expected the fixed, execution-disjoint TT46 frozen season-before control")
    native = Path(course.mpc.__file__).with_name("tt29_rally_rollout.cpp")
    paths = [STOCK, season.PLAN, season.CARRY, temporal, fresh.RISK, execution.FIT, course.mpc.BANK,
             course.OPTIC, course.PRACTICE, feeds, native]
    code = [__file__, season.__file__, course.__file__, course.mpc.__file__, execution.__file__,
            fresh.__file__, fresh.servo.__file__, fresh.shot.__file__,
            Path(__file__).with_name("tt45_rollout_reuse.py")]
    names = dict(assets={str(p): _hash(p) for p in paths},
                 code={Path(p).name: _hash(Path(p)) for p in code})
    return dict(cards=cards[:N], receipt=names, option="Checked cadence only. Existing temporal fallback, "
                "rollout reuse, outgoing ranker, motor bank and speed limits remain fixed.",
                scope="First one of three reply-seed cycles, eight distinct measured releases.")


def args_for(out, n, feeds, *, resume=False):
    return SimpleNamespace(mode="frozen", out=out, deck="assessment", n=n, carry=season.CARRY,
                           feeds=feeds, resume=resume, cap=240)


def check_control(base, fresh_row):
    fields = ("shot", "opponent_seed", "actions", "events", "placements", "score", "memory", "motor_audit")
    changed = [k for k in fields if digest_json(base[k]) != digest_json(fresh_row[k])]
    if changed:
        raise ValueError(f"Fresh stock control changed the season-before archive: {', '.join(changed)}")


def run(feeds):
    receipt = protocol(feeds)
    ROOT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        if receipt != _read(PROTOCOL):
            raise ValueError("Cannot resume with a changed source or fixed setup")
    else:
        _write(PROTOCOL, receipt)
    if not CONTROL.exists():
        season.run(args_for(CONTROL, 1, feeds))
    base = _read(STOCK)
    check_control(base["rows"][0], read_complete(CONTROL)["rows"][0])

    class Frozen(CheckedCadence):
        scales = np.asarray(_read(execution.FIT)["scales"])

    callback = course.mpc.play

    def play(*a, **kw):
        row = callback(*a, **kw)
        row["coverage_edits"] = a[10].reranks
        return row

    with patch.object(season, "Player", Frozen), patch.object(course.mpc, "play", play):
        season.run(args_for(TREATMENT, N, feeds, resume=TREATMENT.exists()))
    report(feeds)


def report(feeds, *, partial=False):
    base, control = map(read_complete, (STOCK, CONTROL))
    candidate = _read(TREATMENT) if partial else read_complete(TREATMENT)
    if protocol(feeds) != _read(PROTOCOL):
        raise ValueError("Expected the assignment captured before running the treatment")
    check_control(base["rows"][0], control["rows"][0])
    same = ("mode", "deck", "ready", "cap", "variant", "budget", "source_sha256", "previous_practice",
            "plan_sha256", "optical_sha256", "carry_sha256")
    if any(base[k] != candidate[k] for k in same) or candidate["cards"] != base["cards"][:N]:
        raise ValueError("The treatment changed stock physical conditions or memory source")
    n = len(candidate["rows"])
    if not 0 < n <= N:
        raise ValueError("Expected checked results on the frozen eight-game roster")
    fixed = base["rows"][0]["memory"]["before"]
    starting_serial = len(_read(season.CARRY)["rows"])
    checkpoint = verify_checkpoint(TREATMENT.with_suffix("").with_suffix(".npz"),
                                   [None] * starting_serial + candidate["rows"], candidate["checkpoint_sha256"])
    if {k: checkpoint[k] for k in fixed} != fixed:
        raise ValueError("Checked treatment changed frozen motor fragments")
    a = dict(base, cards=base["cards"][:n], rows=base["rows"][:n])
    b = dict(candidate, cards=candidate["cards"][:n])
    result = paired(a, b)
    for pair, card, left, right in zip(result["rows"], a["cards"], a["rows"], b["rows"], strict=True):
        for row in (left, right):
            verify_row(card, row)
            if row["memory"] != dict(before=fixed, after=fixed):
                raise ValueError("A held game updated motor memory")
        if left["shot"] != right["shot"] or any(left["motor_audit"][k] != right["motor_audit"][k]
                                               for k in MOTOR_LIMITS):
            raise ValueError("Launch or motor envelope changed")
        k, events = action_prefix(left, right)
        pair.update(first_action_change_tick=k, prefix_events=len(events), offer=offer_contrast(left, right, k),
                    reranks=len(right["coverage_edits"]),
                    cadence={key: right["planner"].get(key, 0) for key in PER_GAME})
    matched = [r["offer"] for r in result["rows"] if r["offer"] and r["offer"]["same_incoming"]]
    _write(OUT, dict(n=n, complete=n == N, stock=metrics(a), checked=metrics(b), pairs=result,
                     matched_changed_offers=dict(n=len(matched),
                         landed=[sum(o["landed"][j] for o in matched) for j in (0, 1)]),
                     checkpoint=checkpoint, source_sha256={p.name: _hash(p) for p in
                                                           (STOCK, CONTROL, TREATMENT, PROTOCOL)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "run", "report"))
    parser.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    parser.add_argument("--partial", action="store_true")
    a = parser.parse_args()
    if a.mode == "verify":
        print([c["id"] for c in protocol(a.feeds)["cards"]])
    elif a.mode == "run":
        run(a.feeds)
    else:
        report(a.feeds, partial=a.partial)
