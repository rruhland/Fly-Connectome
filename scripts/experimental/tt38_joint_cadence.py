"""Diagnose whether fresh-frame replans can rescue known four-refresh opening misses."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as core
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_report import paired, read_complete
from .tt38_joint_mpc_report import metrics


IMPLEMENTATION = core.MPC
DECK = "cadence_failures"
# Known diagnostic failures, not an outcome-independent test deck.
TARGETS = (13628, 9203, 7775)


class AdaptiveClock(IMPLEMENTATION):
    """Use the same planner and proposals, triggered by new sensory evidence."""

    def begin(self):
        super().begin()
        self.accelerated_until = None
        self.last_search_frame = -np.inf

    def step(self, tick, eye, x, q, reference, base):
        flight = eye.latest
        stamp = -np.inf if flight is None else float(flight.time)
        old = len(self.notes)
        if x is not None and x[4] > .9:
            self.accelerated_until = None
        wants_fast = self.accelerated_until is not None and tick <= self.accelerated_until
        if wants_fast and tick >= self.next and stamp <= self.last_search_frame + 1e-9:
            self.next = tick + 1
        result = super().step(tick, eye, x, q, reference, base)
        if len(self.notes) == old:
            return result
        note = self.notes[-1]
        self.last_search_frame = stamp
        if wants_fast:
            self.diag["fresh_burst_rechecks"] += 1
        prior_alive = self.best is not None and 0 <= tick - self.tick < self.lease
        missed = (note["chosen"] is None and not prior_alive and bool(note["bounced"])
                  and x is not None and -1.58 < x[1] < -.64)
        if missed and self.accelerated_until is None:
            self.accelerated_until = tick + 8
            self.diag["burst_starts"] += 1
        if missed and tick < self.accelerated_until:
            self.next = tick + 2
        elif (note["chosen"] is not None and note["robust"] == 2
              and note["contact_delay_s"] > .13):
            self.next = tick + 7
            self.diag["stable_rechecks_postponed"] += 1
        note["next_gap_ticks"] = int(self.next - tick)
        return result


def plan(folder):
    source = core.PLAN
    held = _read(source)["held_openings"]
    cards = [next(c for c in held if c["id"] == k) for k in TARGETS]
    _write(folder / "joint-cadence-plan.json", {
        DECK: cards,
        "source_sha256": _hash(source),
        "selection": "Previously observed four-refresh opening errors, for diagnosis only; not held claims.",
        "fixed_gap_ticks": 5,
        "adaptive_gap_ticks": [2, 5, 7],
        "burst_budget_ticks": 8,
    })


def assess(folder, mode, feeds, optic):
    policy = AdaptiveClock if mode == "adaptive" else IMPLEMENTATION
    path = folder / f"joint-cadence-{mode}.json.gz"
    course = folder / "joint-cadence-plan.json"
    with patch.object(core, "MPC", policy):
        core.run(feeds, core.BANK, course, DECK, "frozen", path, optic=optic,
                 budget=16, cap=200, variant="guarded")
    doc = read_complete(path)
    doc["cadence"] = mode
    _write(path, doc)


def report(folder):
    paths = {k: folder / f"joint-cadence-{k}.json.gz" for k in ("fixed", "adaptive")}
    docs = {k: read_complete(p) for k, p in paths.items()}
    before, after = docs["fixed"], docs["adaptive"]
    if (before["source_sha256"], before["optical_sha256"], before["budget"], before["cap"]) != (
            after["source_sha256"], after["optical_sha256"], after["budget"], after["cap"]):
        raise ValueError("The cadence experiment changed the model or sample budget")
    _write(folder / "joint-cadence-results.json", dict(
        stats={k: metrics(v) for k, v in docs.items()},
        paired=paired(before, after),
        controls={mode: _hash(p) for mode, p in paths.items()},
        command_refresh_ticks={mode: [[s["tick"] for s in r["searches"]] for r in d["rows"]]
                               for mode, d in docs.items()},
        scope="Known failure diagnosis only. Same openings and nominal opportunity seeds as prior experiments.",
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "fixed", "adaptive", "report"))
    p.add_argument("--folder", type=Path, default=core.OUT)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--optic", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    a = p.parse_args()
    if a.action == "plan":
        plan(a.folder)
    elif a.action == "report":
        report(a.folder)
    else:
        assess(a.folder, a.action, a.feeds, a.optic)
