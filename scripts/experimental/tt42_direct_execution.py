"""Check accepted joint motion against live sensors before asking the unchanged ranker again."""

import argparse
import math
from pathlib import Path
from time import process_time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt37_joint_report import paired, read_complete
from .tt38_joint_mpc_report import metrics
from .tt39_hindsight import flight


BASE = mpc.MPC
CALIBRATION = Path("artifacts/tt39/hindsight-practice.json.gz")
FIT = course.ROOT / "direct-execution-fit.json"
HELD = course.ROOT / "direct-execution-held.json.gz"
STOCK = course.ROOT / "no-more-practice.json.gz"
READ_MIN = np.array([.003, .045, .022])
READ_MAX = np.array([.038, .85, .088])
STRIDE = 2
NORMAL = 5
FAR = 7
CLOSE = 8
MAX_FAST = 3


def innovation(live, encoders, trace):
    return np.array([np.linalg.norm(trace[:3] - live[:3]),
                     np.linalg.norm(trace[3:6] - live[3:]),
                     np.linalg.norm(trace[6:11] - encoders[:5])])


def fit():
    """Estimate normal short-run residuals against actual RGB and encoders, not virtual rewards."""
    doc = read_complete(CALIBRATION)
    roles = _read(course.PLAN)
    if {c["id"] for c in doc["cards"]} & {c["id"] for c in roles["assessment"]}:
        raise ValueError("Execution calibration overlaps the held releases")
    samples = []
    with mpc.centered(doc["ready"]):
        engine = mpc.OpticalClinic(body=True)
        try:
            for card, row in zip(doc["cards"], doc["rows"], strict=True):
                sensory = row["sensory"]
                engine.previous, engine.bounced = None, False
                impact = next((e["time"] for e in row["events"]
                               if e["kind"] == "contact" and e["side"] == 1), math.inf)
                for tick, obs in enumerate(sensory):
                    engine.observe(tick, SimpleNamespace(latest=flight(obs, tick)),
                                   np.asarray(obs["q"]), np.asarray(obs["ref"]))
                    x = obs["x"]
                    if (tick % 2 or not engine.bounced or obs["xyz"] is None or x is None
                            or tick + 6 >= len(sensory) or not -.65 > x[1] > -1.56 or x[4] >= -.9):
                        continue
                    if not engine.sync():
                        continue
                    # The calibration rollout receives only known past observations and the
                    # exact buffered references which were subsequently sent in this archive.
                    goals = np.array([[s["ref"] for s in sensory[tick + 1:tick + 7]]])
                    trace = np.full((1, 6, 17), np.nan)
                    engine.trajectories(goals, mpc.AIM, 6 * mpc.DT, traces=trace)
                    for k in (2, 4):
                        after = sensory[tick + k]
                        state = after["x"]
                        if (state is None or after["xyz"] is None or state[4] >= -.9
                                or (tick + k) * mpc.DT >= impact or not np.isfinite(trace[0, k, :11]).all()):
                            continue
                        err = innovation(np.asarray(state), np.asarray(after["q"]), trace[0, k])
                        samples.append(dict(id=card["id"], good=bool(row["score"]["learner"]),
                                            age=k, tick=tick, values=err.tolist()))
        finally:
            engine.twin.close()
    good = [s["values"] for s in samples if s["good"]]
    missed = [s["values"] for s in samples if not s["good"]]
    if len(good) < 8 or len(missed) < 8:
        raise ValueError("Insufficient real stereo and encoder corroboration on good and missed practice")
    scales = np.clip(np.quantile(good, .85, axis=0), READ_MIN, READ_MAX)
    _write(FIT, dict(scales=scales.tolist(), good_trials=len(good), missed_trials=len(missed),
                     good_median=np.median(good, axis=0).tolist(), missed_median=np.median(missed, axis=0).tolist(),
                     good_exceed=int(np.sum(np.max(np.asarray(good) / scales, axis=1) > 1.)),
                     missed_exceed=int(np.sum(np.max(np.asarray(missed) / scales, axis=1) > 1.)),
                     rows=samples, train_sha256=_hash(CALIBRATION), course_sha256=_hash(course.PLAN),
                     fit_rule="Successful physical practice p/v/q residual component p85 with fixed physical floors. "
                              "Unrewarded practice estimates contrast only.",
                     schedule=dict(normal=NORMAL, far=FAR, close=CLOSE, minimum=STRIDE, fast_max=MAX_FAST)))


class Execution(BASE):
    def begin(self):
        super().begin()
        self.forecast = None
        self.hit = -99
        self.support = 0
        self.last_check = -99
        self.fast = 0

    def refresh_incumbent(self, tick, *, new=False):
        """Roll out just the currently issued motion. Leave candidate/risk ranking untouched."""
        started = process_time()
        age = tick - self.tick
        self.last_check = tick
        if self.best is None or not 0 <= age < mpc.HORIZON - 1 or not self.engine.sync():
            self.forecast = None
            return False
        goals = mpc.tail(self.best[age:], mpc.HORIZON)
        _, legal, states, values, _ = self.trial([goals], long=True, traces=True)
        self.diag["inspection_cpu_s"] += process_time() - started
        self.diag["incumbent_inspections"] += 1
        if not bool(legal[0]) or not 0 < values[0, 1] <= mpc.HORIZON * mpc.DT:
            self.diag["incumbent_inspection_failed"] += 1
            self.forecast = None
            return False
        self.hit = tick + int(np.ceil(values[0, 1] / mpc.DT))
        self.forecast = tick, states[0, :mpc.HORIZON, :11].copy()
        if new:
            self.diag["installed_forecast"] += 1
        return True

    def step(self, tick, eye, x, q, reference, base):
        self.engine.observe(tick, eye, q, reference.copy())
        if x is not None and x[4] > .9:
            self.best = self.forecast = None
            self.fast = 0
            return base
        eligible = x is not None and x[4] < -.9 and -.49 > x[1] > -1.62 and x[2] > .77
        if eligible:
            self.update_motion(tick, eye, x, q, reference, base)
        if self.best is not None and 0 <= tick - self.tick < self.lease:
            self.diag["screened_ticks"] += 1
            return self.best[tick - self.tick]
        return base

    def update_motion(self, tick, eye, x, q, reference, base):
        track = eye.filter.track
        fresh = bool(track and abs(tick * mpc.DT - track[-1][0]) < 1e-8)
        due = tick >= self.next
        due_reason = "clock"
        if self.forecast is not None and self.best is not None and fresh:
            start, trace = self.forecast
            age, until = tick - self.tick, self.hit - tick
            sample = tick - start
            if STRIDE <= sample < len(trace) and np.isfinite(trace[sample]).all():
                error = float(np.max(innovation(x, q, trace[sample]) / self.scales))
                urgent = (0 < until <= CLOSE and (error > 1. or self.support < 2)
                          and tick - self.last_check >= STRIDE and self.fast < MAX_FAST)
                stable = due and 0 < until and error <= 1. and self.support == 2 and age < mpc.HORIZON - 2
                if urgent or stable:
                    if self.refresh_incumbent(tick):
                        if stable:
                            gap = FAR - NORMAL if until > CLOSE else STRIDE
                            self.next = tick + gap
                            self.lease = min(mpc.HORIZON, max(self.lease, age + gap))
                            self.diag["preserved_due_calls"] += 1
                            return
                    else:
                        due = True
                        due_reason = "innovation" if error > 1. else "nominal_failure"
                        self.best = None
                        self.diag["invalidated_motion"] += 1
                        if tick < self.next:
                            self.fast += 1
                            self.diag["early_replans"] += 1
                elif until < -1:
                    # No indefinite command revival if the forecasted hit failed to arrive.
                    self.forecast = None
        if not due:
            return
        self.next = tick + NORMAL
        n = len(self.notes)
        self.search(tick, x, q, reference, base)
        if len(self.notes) == n:
            return
        note = self.notes[-1]
        note["execution_trigger"] = due_reason
        if note.get("chosen"):
            self.support = int(note["robust"])
            note["execution_forecast"] = bool(self.refresh_incumbent(tick, new=True))
        elif note.get("prepare"):
            self.forecast = None


def assess(output, *, deck, n, feeds, resume=False):
    d = _read(FIT)
    if (d["train_sha256"] != _hash(CALIBRATION) or d["course_sha256"] != _hash(course.PLAN)
            or d["schedule"] != dict(normal=NORMAL, far=FAR, close=CLOSE, minimum=STRIDE, fast_max=MAX_FAST)):
        raise ValueError("Calibrated execution protocol changed")
    if deck == "assessment" and n != len(_read(course.PLAN)["assessment"]):
        raise ValueError("The held execution protocol includes every assigned assessment start")
    sources = {Path(p).name: _hash(Path(p)) for p in
               (__file__, course.__file__, mpc.__file__, FIT, feeds, mpc.BANK, course.OPTIC, course.PRACTICE)}
    protocol = output.with_suffix("").with_suffix(".protocol.json")
    if protocol.exists() and resume:
        if _read(protocol) != sources:
            raise ValueError("The requested continuation changed the fixed actor or observations")
    else:
        _write(protocol, sources)

    class FrozenExecution(Execution):
        scales = np.asarray(d["scales"])

    with patch.object(mpc, "MPC", FrozenExecution):
        course.run(SimpleNamespace(mode="frozen", out=output, deck=deck,
                                   n=n, feeds=feeds, carry=None, resume=resume))
    d = read_complete(output)
    _write(output, dict(d, execution_source_sha256=sources[Path(__file__).name], fit_sha256=sources[FIT.name]))


def report():
    left, right = map(read_complete, (STOCK, HELD))
    required = ("cards", "mode", "deck", "ready", "cap", "variant", "budget", "source_sha256",
                "previous_practice", "plan_sha256", "optical_sha256", "carry_sha256", "checkpoint_sha256")
    if any(left[k] != right[k] for k in required) or right["fit_sha256"] != _hash(FIT):
        raise ValueError("Execution comparison changed a live input, matched card or held memory")
    for card, a, b in zip(left["cards"], left["rows"], right["rows"], strict=True):
        for row in (a, b):
            if (row["placements"]["card"] != card or row["memory"]["before"] != row["memory"]["after"]
                    or audit(row["events"], opening_side=1) != row["score"]):
                raise ValueError("Physical assignment, event score or frozen memory failed verification")
    outcomes = paired(left, right)
    for row, a, b in zip(outcomes["rows"], left["rows"], right["rows"], strict=True):
        fork = next((k for k, (x, y) in enumerate(zip(a["actions"], b["actions"])) if x != y), None)
        prefix = [0, 0]
        if fork is not None:
            for side, log in enumerate((a, b)):
                prefix[side] = sum(s["side"] == 1 and s["landing"] <= fork * mpc.DT + 1e-9
                                   for s in log["score"]["shots"])
            if prefix[0] != prefix[1]:
                raise ValueError("Physical score changed before the matched action prefix ended")
        row.update(first_action_change_tick=fork, scored_own_on_exact_prefix=prefix[0])
    _write(course.ROOT / "direct-execution-results.json", dict(
        stock=metrics(left), execution=metrics(right), paired=outcomes,
        stock_sha256=_hash(STOCK), execution_sha256=_hash(HELD), fit_sha256=_hash(FIT)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "practice", "held", "report"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.phase == "fit":
        fit()
    elif args.phase == "report":
        report()
    else:
        practice = args.phase == "practice"
        assess(course.ROOT / "direct-execution-practice.json.gz" if practice else HELD,
               deck="practice" if practice else "assessment", n=3 if practice else 24,
               feeds=args.feeds, resume=args.resume)
