"""Learn an answerable continuous flight envelope and feed it into every joint-search stage."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt45_tactical_coverage as science
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_opportunity_map import projected_height

ROOT = science.ROOT
BASE = mpc.MPC
CONTROL = course.ROOT / "live-after12.json.gz"
CHECKPOINT = course.ROOT / "learning-12.json.gz"
LANE = .16
DEPTHS = ((.30, .70), (.34, .76), (.42, .84), (.49, .90))
PACES = ((1.5, 2.1), (2.10, 2.65), (2.65, 3.20), (3.20, 4.60))


def rows_for(data):
    disjoint = set(data["fit_releases"])
    return [r for r in data["rows"] if r["id"] in disjoint]


def fit_goal(source, output):
    """Learn only from physical practice outcomes, never from requests or sweep outcomes."""
    data = _read(source)
    train = rows_for(data)
    scores = []
    for dep in DEPTHS:
        for pace in PACES:
            group = [
                r for r in train if (abs(r["outgoing"][0]) < LANE and dep[0] < r["outgoing"][1] < dep[1]
                                     and pace[0] < r["outgoing"][2] < pace[1])
            ]
            hard = [r for r in group if r["incoming"][1] < science.HARD]
            nr = len({r["id"] for r in group})
            if len(group) < 100 or len(hard) < 14 or nr < 14:
                continue

            def failed(rows):
                return sum(r["height"] is None or r["height"] < science.THRESHOLD for r in rows)

            f, hf = failed(group), failed(hard)
            # Support and hard-receive penalties avoid liking tiny, easy-only windows or opponent faults.
            merit = .71 * (f + 1.5) / (len(group) + 16.) + .29 * (hf + 1.) / (len(hard) + 10.)
            scores.append(dict(depth=dep, pace=pace, support=len(group), failures=f, hard_n=len(hard),
                               hard_failures=hf, distinct_releases=nr, fit_loss=merit,
                               center=np.median([r["outgoing"][:3] for r in group], axis=0).tolist()))
    if not scores:
        raise ValueError("No region had adequate distinct released practice support")
    best = min(scores, key=lambda r: r["fit_loss"])
    _write(output, dict(role="Goal fit from achieved outgoing practice and next physical outcomes only",
                        source_sha256=_hash(source), code_sha256=_hash(Path(__file__)),
                        criterion="Minimum regularized bad-followup rate over physically populated geometry cells",
                        selected=best, comparison=scores, learner_fields=("x_m", "bounce_y_m", "incident_vy_mps"),
                        excluded_draw_matrix=data["matrix_releases"]))
    print(best, flush=True)


class Envelope:
    def __init__(self, data):
        r = data["selected"]
        self.lo = np.r_[-LANE, r["depth"][0], r["pace"][0]]
        self.hi = np.r_[ LANE, r["depth"][1], r["pace"][1]]
        self.center = np.asarray(r["center"])

    def penalty(self, shape):
        a = np.asarray(shape)[:, :3]
        outside = np.maximum(np.maximum(self.lo - a, a - self.hi), 0.)
        distance = np.sqrt(1 + (outside / [.13, .21, .31])**2) - 1
        closer = np.abs(a - self.center) @ [.27, .21, .16]
        return np.minimum(1.35, closer + distance @ [.48, .73, 1.24])

    def contains(self, a):
        return bool(np.all((self.lo < a[:3]) & (a[:3] < self.hi)))


class TacticalMPC(BASE):
    """The same stock direct law, with the practiced envelope grading its earliest candidate trials."""

    def __init__(self, *args, goal, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal = goal

    def search(self, tick, x, q, reference, base):
        self.scored_shapes = {}
        before = len(self.notes)
        super().search(tick, x, q, reference, base)
        if (len(self.notes) > before and self.notes[-1].get("chosen") and self.best is not None):
            key = self.best.tobytes()
            if key in self.scored_shapes:
                self.notes[-1]["upstream_goal"] = self.scored_shapes[key]

    def trial(self, goals, *, long=False, traces=False):
        loss, legal, track, scores, f = super().trial(goals, long=long, traces=traces)
        separated = np.flatnonzero(f[:, 1] > 0)
        if len(separated):
            shape = science.arc_shape(f[separated, 5:8], f[separated, 2:5])
            okay = np.isfinite(shape).all(axis=1)
            use = separated[okay]
            if len(use):
                penalty = self.goal.penalty(shape[okay])
                # Even the initial short anchor screen sees this smooth goal when it has
                # a credible departure. Illegal paths keep the ordinary much larger cost.
                delta = np.where(legal[use], 1.0, .73) * penalty
                loss[use] += delta
                self.diag["goal_scored"] += len(use)
                self.diag["goal_scored_preliminary"] += 0 if long else len(use)
                if long:
                    for j, s, v in zip(use, shape[okay], delta, strict=True):
                        if legal[j]:
                            self.scored_shapes.setdefault(np.asarray(goals[j]).tobytes(),
                                                          dict(shape=s.tolist(), penalty=float(v)))
        return loss, legal, track, scores, f


def run(a):
    data = _read(a.envelope)
    stock = _read(a.control)
    desc = dict(actor_sha256=_hash(Path(__file__)), envelope_sha256=_hash(a.envelope),
                stock_sha256=_hash(a.control), evaluation="matched continuous motor learning",
                learning_boundary="Frozen learned goal from disjoint practice, real joint fragments earned during/after each game.")
    if a.out.exists() and any(_read(a.out).get(k) != v for k,v in desc.items()):
        raise ValueError("Existing native recording used different envelope/control conditions")
    cards = _read(course.PLAN)["assessment"][:a.n]
    if (cards != stock["cards"][:a.n] or len(stock["rows"]) < a.n or stock["mode"] != "live"
            or stock["carry_sha256"] != _hash(a.carry) or stock["cap"] != a.cap):
        raise ValueError("Expected the matched live carried baseline")
    envelope = Envelope(data)

    def save(path, tape):
        _write(path, tape | desc)

    a.mode, a.deck, a.resume = "live", "assessment", a.out.exists()
    with patch.object(mpc, "MPC", lambda *args, **kw: TacticalMPC(*args, goal=envelope, **kw)),             patch.object(course, "_write", save):
        course.run(a)


def compare(a):
    d, stock = _read(a.out), _read(a.control)
    if d["stock_sha256"] != _hash(a.control) or d["cards"] != stock["cards"][:len(d["cards"])]:
        raise ValueError("Incorrect matched control")
    paired = []
    envelope = Envelope(_read(a.envelope))
    for i, (r, s) in enumerate(zip(d["rows"], stock["rows"])):
        card = d["cards"][i]
        for _row in (r, s):
            for j, entry in enumerate(_row["placements"]["requested"]):
                if entry["target"] != list(course.draw(card, j, "random", None)[0]):
                    raise ValueError("Opponent random placement differs")
        if not i and r["memory"]["before"] != s["memory"]["before"]:
            raise ValueError("Different entry memory despite loaded practice checkpoint")
        pair = dict(id=card["id"], own=[s["score"]["learner"], r["score"]["learner"]], changed=None,
                    opening=[bool(z["placements"]["delivered"] and z["placements"]["delivered"][0]["near_landed"])
                             for z in (s, r)],
                    physical_motor_updates=r["planner"].get("credited_fragments", 0),
                    earliest_selected_goal=next((n["upstream_goal"] for n in r["searches"] if "upstream_goal" in n), None),
                    measured_cpu=[s["runtime_cpu_s"],r["runtime_cpu_s"]])
        if s["actions"] != r["actions"]:
            pair["changed"] = next((j for j, (x,y) in enumerate(zip(s["actions"],r["actions"])) if x != y),
                                   min(len(s["actions"]),len(r["actions"])))
        outcome = []
        for z in (s,r):
            p = z["placements"]["delivered"]
            f = [projected_height(o["receive_line"]) for o in p[1:] if o["receive_line"]]
            own = [e for e in z["events"] if e.get("collider") == "learner_racket"]
            landing = [e for e in z["events"] if e["kind"] == "legal_return" and e["side"] == 1][1:]
            arcs = []
            for e in landing:
                dpt = next((x for x in reversed(own) if x["time"] < e["time"]), None)
                if dpt is not None:
                    val = science.arc_shape(dpt["ball"], dpt["velocity"])
                    if np.isfinite(val).all():
                        arcs.append(val)
            outcome.append(dict(known_lows=sum(t < science.THRESHOLD for t in f), tracked=len(f),
                                arcs=len(arcs), pace_in_envelope=sum(envelope.contains(el) for el in arcs)))
        pair["followup"] = outcome
        paired.append(pair)
    y = np.array([r["own"] for r in paired])
    captured = dict(role="Native whole games under identical continuous TT37 draws", rows=paired,
                    total=y.sum(axis=0).tolist(), max=y.max(axis=0).tolist(),
                    complete=len(d["rows"]) == len(d["cards"]),
                    stock_sha256=_hash(a.control), live_sha256=_hash(a.out), envelope_sha256=_hash(a.envelope))
    _write(a.report, captured)
    print(captured, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "run", "compare"))
    p.add_argument("--source", type=Path, default=ROOT / "physical-arc-plan.json.gz")
    p.add_argument("--envelope", type=Path, default=ROOT / "learned-envelope.json")
    p.add_argument("--out", type=Path, default=ROOT / "upstream-live.json.gz")
    p.add_argument("--control", type=Path, default=CONTROL)
    p.add_argument("--carry", type=Path, default=CHECKPOINT)
    p.add_argument("--report", type=Path, default=ROOT / "upstream-report.json")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--cap", type=int, default=240)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    args = p.parse_args()
    args.envelope.parent.mkdir(parents=True, exist_ok=True)
    {"fit": lambda a: fit_goal(a.source, a.envelope), "run": run, "compare": compare}[args.phase](args)
