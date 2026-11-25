"""Learn whether to keep a checked joint buffer for two extra ticks during real play."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import tt42_direct_execution as execution
from . import tt46_checked_execution as checked
from . import tt48_practice_transfer as stream
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_report import paired, read_complete
from .tt38_joint_mpc_report import metrics
from .tt42_feedback_verification import action_prefix, offer_contrast, verify_checkpoint, verify_row


ROOT = Path("artifacts/tt51_commitment")
CONTROL = stream.ROOT / "live-plain.json.gz"
OUTPUT = ROOT / "learning.json.gz"
PROTOCOL = ROOT / "protocol.json"
REPORT = ROOT / "comparison.json"
N = 16
HOLD = 2
LIMIT = 1.25
BAND = 0.6
course, season = stream.course, stream.season
mpc = course.mpc


class Book:
    """One observation/choice/settled outcome per actual inbound ball."""

    def __init__(self, counts=None):
        self.counts = np.zeros((2, 2, 2), dtype=int) if counts is None else np.asarray(counts, dtype=int).copy()
        if self.counts.shape != (2, 2, 2) or np.any(self.counts < 0):
            raise ValueError("Invalid commitment checkpoint")

    def choose(self, error):
        band = int(error > BAND)
        local = self.counts[band]
        pooled = self.counts[1 - band]
        weighted = local + .25 * pooled
        n = weighted.sum(axis=1)
        # A cautious prior. Observe the ordinary arm before making a bounded,
        # nominally legal hold probe. Afterwards risk and confidence both update.
        posterior = (2.5 + weighted[:, 1]) / (3.5 + n)
        value = posterior + .18 / np.sqrt(1. + n) - np.array([0., .018])
        if not self.counts[:, 0].sum():
            arm, kind = 0, "cold-stock"
        elif not local[1].sum() and (local[0].sum() or self.counts[:, 0].sum() >= 3):
            arm, kind = 1, "bounded-hold-probe"
        else:
            arm, kind = int(np.argmax(value)), "outcome-weighted"
        return band, arm, kind, posterior.tolist()

    def observe(self, band, arm, reward):
        self.counts[band, arm, int(reward)] += 1

    def serializable(self):
        return self.counts.tolist()


class Learned(checked.CheckedCadence):
    def __init__(self, *args, book, scales, **kwargs):
        super().__init__(*args, **kwargs)
        self.book, self.scales = book, scales
        self.priors.duration_feedback = self.score_duration

    def begin(self):
        super().begin()
        self.book_before = self.book.serializable()
        self.duration_decisions = []
        self.pending = None

    def update_motion(self, tick, eye, x, q, reference, base):
        if tick < self.next:
            return
        track = eye.filter.track
        fresh = bool(track and abs(tick * mpc.DT - track[-1][0]) < 1e-8)
        if fresh and self.pending is None and self.forecast is not None and self.best is not None and self.support == 2:
            start, trace = self.forecast
            k, age, until = tick - start, tick - self.tick, self.hit - tick
            if (execution.STRIDE <= k < len(trace) and 0 < until <= execution.CLOSE
                    and 0 <= age < mpc.HORIZON - HOLD and np.isfinite(trace[k]).all()):
                levels = execution.innovation(x, q, trace[k]) / self.scales
                error = float(np.max(levels))
                if error <= LIMIT and self.refresh_incumbent(tick):
                    band, arm, method, prior = self.book.choose(error)
                    self.pending = dict(tick=tick, band=band, arm=arm, method=method,
                                        innovation=levels.tolist(), until=until, age=age, prior=prior,
                                        reward=None, settlement_tick=None)
                    self.duration_decisions.append(self.pending)
                    self.diag["duration_choices"] += 1
                    if arm == 1:
                        self.next = tick + HOLD
                        self.lease = min(mpc.HORIZON, max(self.lease, age + HOLD))
                        self.diag["duration_holds"] += 1
                        return
        # Baseline full rank, warm proposals, and outgoing scoring stay intact.
        self.next = tick + execution.NORMAL
        n = len(self.notes)
        self.search(tick, x, q, reference, base)
        if len(self.notes) != n:
            note = self.notes[-1]
            if note.get("chosen"):
                self.support = int(note["robust"])
                self.refresh_incumbent(tick, new=True)
            elif note.get("prepare"):
                self.forecast = None

    def score_duration(self, archive, reward):
        if self.pending is None:
            return
        decision = self.pending
        tick = int(archive[-1][0])
        if tick < decision["tick"]:
            raise ValueError("Physical feedback precedes held duration decision")
        self.book.observe(decision["band"], decision["arm"], reward)
        decision.update(reward=bool(reward), settlement_tick=tick, counts=self.book.serializable())
        self.pending = None
        self.diag["duration_rewards"] += int(reward)
        self.diag["duration_misses"] += int(not reward)

    def missed(self, archive):
        self.score_duration(archive, False)


class Memory(course.Carried):
    def keep(self, archive, leg):
        earned = super().keep(archive, leg)
        self.duration_feedback(archive, True)
        return earned


def protocol(feeds):
    baseline = read_complete(CONTROL)
    source = _read(stream.SOURCE)
    cards = _read(stream.PLAN)["practice"][:N]
    if (baseline["cards"][:N] != cards or len({c["id"] for c in cards}) != N or baseline["actor"] != "plain"
            or baseline["mode"] != "live" or baseline["carry_sha256"] != _hash(stream.SOURCE)
            or baseline["plan_sha256"] != _hash(stream.PLAN)
            or {c["id"] for c in cards} & {c["id"] for c in source["cards"][:len(source["rows"])]}):
        raise ValueError("Expected the archived plain live run on the first independent official stream")
    paths = (CONTROL, stream.PLAN, stream.SOURCE, stream.SOURCE.with_suffix("").with_suffix(".npz"),
             season.CARRY, execution.FIT, season.fresh.RISK, season.fresh.servo.ROOT / "prior.npz", mpc.BANK,
             course.OPTIC, course.PRACTICE, feeds)
    code = (__file__, stream.__file__, stream.retention.__file__, course.__file__, mpc.__file__,
            execution.__file__, checked.__file__, season.__file__, season.fresh.__file__, season.fresh.servo.__file__,
            season.fresh.shot.__file__, Path(mpc.__file__).with_name("tt29_rally_rollout.cpp"),
            Path(mpc.__file__).with_name("tt45_rollout_reuse.py"))
    return dict(cards=cards, assets={str(p): _hash(p) for p in paths},
                code={Path(p).name: _hash(Path(p)) for p in code}, hold_ticks=HOLD, innovation_limit=LIMIT,
                feedback="Original live score gate. One eligible decision per inbound ball. Actual own legal "
                         "return or real terminal unreturned offer settles this policy choice. No virtual rewards.",
                scope="First 16 preassigned games, including the native 63-return stock assignment.")


def run(feeds):
    receipt = protocol(feeds)
    ROOT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        if _read(PROTOCOL) != receipt:
            raise ValueError("Cannot resume changed duration policy or physical assignment")
    else:
        _write(PROTOCOL, receipt)
    resume = OUTPUT.exists()
    book = Book(_read(OUTPUT)["rows"][-1]["duration"]["after"] if resume else None)
    scales = np.asarray(_read(execution.FIT)["scales"])
    ordinary = mpc.play

    def play(*a, **kw):
        row = ordinary(*a, **kw)
        player = a[10]
        row["duration"] = dict(before=player.book_before, after=book.serializable(),
                               decisions=player.duration_decisions)
        row["coverage_edits"] = player.reranks
        return row

    def build(*a, **kw):
        return Learned(*a, book=book, scales=scales, **kw)

    a = SimpleNamespace(actor="plain", out=OUTPUT, n=N, deck="practice", mode="live", carry=stream.SOURCE,
                        policy=stream.SOURCE, resume_from=None, feeds=feeds, cap=240, resume=resume)
    with patch.object(season, "Player", build), patch.object(course, "Carried", Memory), patch.object(mpc, "play", play):
        stream.run(a)
    saved = _read(OUTPUT)
    _write(OUTPUT, dict(saved, actor="plain+learned-duration", duration_protocol_sha256=_hash(PROTOCOL)))
    report(feeds)


def verify_duration(row, book):
    receipt = row["duration"]
    if receipt["before"] != book.serializable():
        raise ValueError("Inconsistent commitment memory before physical point")
    game_scores = [e["step"] for e in row["events"] if e["kind"] == "legal_return" and e["side"] == 1][1:]
    last = -1
    inside = settled = 0
    for d in receipt["decisions"]:
        if (d["tick"] <= last or max(d["innovation"]) > LIMIT
                or tuple([d["band"], d["arm"], d["method"], d["prior"]]) != book.choose(max(d["innovation"]))):
            raise ValueError("Decision does not match causal commitment memory and observed innovation")
        last = d["settlement_tick"] if d["reward"] is not None else len(row["actions"])
        if d["reward"] is None:
            continue
        scores = [s for s in game_scores if d["tick"] < s <= last + 1]
        if bool(scores) != d["reward"] or (scores and abs(scores[-1] - last) > 1):
            raise ValueError("Duration feedback is not the observed native outcome")
        book.observe(d["band"], d["arm"], d["reward"])
        if book.serializable() != d["counts"]:
            raise ValueError("Commitment memory update does not match the causal reward")
        inside += settled > 0 and d["method"] == "outcome-weighted"
        settled += 1
    if receipt["after"] != book.serializable():
        raise ValueError("Commitment weights changed without settled outcome")
    return inside


def report(feeds, *, partial=False):
    a = read_complete(CONTROL)
    b = _read(OUTPUT) if partial else read_complete(OUTPUT)
    if protocol(feeds) != _read(PROTOCOL) or b["cards"] != a["cards"][:N]:
        raise ValueError("Physical comparison changed its predeclared inputs")
    keys = ("mode", "deck", "ready", "cap", "variant", "budget", "source_sha256", "previous_practice",
            "plan_sha256", "optical_sha256", "carry_sha256")
    if any(a[k] != b[k] for k in keys):
        raise ValueError("The learned duration arm changed ordinary live course conditions")
    n = len(b["rows"])
    a = dict(a, cards=a["cards"][:n], rows=a["rows"][:n])
    b = dict(b, cards=b["cards"][:n])
    memory_path = stream.SOURCE.with_suffix("").with_suffix(".npz")
    with np.load(memory_path) as z:
        serial = int(z["serial"])
    motor = verify_checkpoint(OUTPUT.with_suffix("").with_suffix(".npz"), [None] * serial + b["rows"],
                              b["checkpoint_sha256"])
    result, book = paired(a, b), Book()
    previous = a["rows"][0]["memory"]["before"]
    for entry, card, left, right in zip(result["rows"], a["cards"], a["rows"], b["rows"], strict=True):
        for row in (left, right):
            verify_row(card, row)
        if right["memory"]["before"] != previous or left["shot"] != right["shot"]:
            raise ValueError("Motor continuity or native physical launch changed")
        previous = right["memory"]["after"]
        seen = verify_duration(right, book)
        fork, prefix = action_prefix(left, right)
        same_motor = left["memory"]["before"] == right["memory"]["before"]
        first_hold = min((d["tick"] for d in right["duration"]["decisions"] if d["arm"]), default=None)
        if same_motor and fork is not None and (first_hold is None or fork < first_hold):
            raise ValueError("Commands changed before the first supported duration intervention")
        entry.update(first_action_change=fork, exact_prefix_events=len(prefix),
                     same_initial_motor=same_motor, first_hold_tick=first_hold,
                     first_changed_offer=offer_contrast(left, right, fork),
                     scored_choices=sum(d["reward"] is True for d in right["duration"]["decisions"]),
                     missed_choices=sum(d["reward"] is False for d in right["duration"]["decisions"]),
                     within_rally_posterior_choices=seen,
                     held_duration_ticks=HOLD * right["planner"].get("duration_holds", 0))
    _write(REPORT, dict(n=n, complete=n == N, stock=metrics(a), learner=metrics(b), pairs=result, motor=motor,
                        duration_checkpoint=book.serializable(), source_sha256={p.name: _hash(p) for p in
                                                                               (CONTROL, OUTPUT, PROTOCOL)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("verify", "run", "report"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    if a.mode == "verify":
        print([c["id"] for c in protocol(a.feeds)["cards"]])
    elif a.mode == "run":
        run(a.feeds)
    else:
        report(a.feeds, partial=a.partial)
