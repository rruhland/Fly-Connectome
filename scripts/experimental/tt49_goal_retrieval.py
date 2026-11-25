"""Recall conditionally similar physically scored movement into a learned outgoing envelope."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as combined
from . import tt45_tactical_coverage as observed
from . import tt45_tactical_goal as tactic
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt39_turn_time import turn_time
from .tt41_anticipate import physical_projection

ROOT = Path("artifacts/tt49_tactical_retrieval")
STUDY = observed.ROOT / "physical-arc-plan.json.gz"
REF = season.ROOT / "learning-12.json.gz"
PLAN = ROOT / "official.json"
GOAL = ROOT / "envelope.json"
MAX_RESERVED = 2
SAFETY_MIN = 1.002


def receipt():
    schedule = _read(season.PLAN)
    known = {c["id"] for g in ("practice", "assessment") for c in schedule[g]}
    for path in (mpc.BANK, course.PRACTICE):
        occupied = {r["card"]["id"] for r in _read(path.with_suffix(".json"))["practice"]}
        if occupied & known:
            raise ValueError("Season release was used in official motor archive")
    carry = _read(season.CARRY)
    if known & {c["id"] for c in carry["cards"][:len(carry["rows"])]}:
        raise ValueError("Official season hardware practice reuses a test release")
    data = _read(STUDY)
    data["fit_releases"] = [r for r in data["fit_releases"] if r not in known]
    source = ROOT / "disjoint-practice.json.gz"
    data = dict(rows=[r for r in data["rows"] if r["id"] in data["fit_releases"]],
                fit_releases=data["fit_releases"], sources=data["sources"],
                matrix_releases=data["matrix_releases"], restriction_to_fit_nonseason=True,
                initial_source_sha256=_hash(STUDY))
    _write(source, data)
    tactic.fit_goal(source, GOAL)
    ref = _read(REF)
    chosen = schedule["practice"][:8]
    if chosen != ref["cards"][:8] or ref["carry_sha256"] != _hash(season.CARRY):
        raise ValueError("Wrong season reference")
    _write(PLAN, dict(cards=chosen, known=sorted(known), fit_ids=data["fit_releases"],
                      files={str(p): _hash(p) for p in [STUDY, source, GOAL, season.PLAN,
                             season.CARRY, REF, mpc.BANK, course.PRACTICE, combined.RISK]},
                      mode="live", budget=28, cap=240,
                      rule="First eight official season rounds, no filtering by score. Same no-easing wide opponent. Add only physically qualified goal-conditioned motor retrieval to history and risk."))


def useful(row):
    """Legally landed action, then a real answered reply of measurable safe height."""
    result = []
    departures = [e for e in row["events"] if e.get("collider") == "learner_racket"]
    deliveries = row["placements"]["delivered"]
    for j, (r, later) in enumerate(zip(deliveries, deliveries[1:])):
        hit, line = r.get("near_contact"), later.get("receive_line")
        if not r["near_landed"] or not hit or line is None or projected_height(line) < .99:
            continue
        sep = next((e for e in departures if hit["time"] < e["time"] < later["far_contact"]["time"]), None)
        if sep:
            result.append((round(hit["time"] * 480), j+1, observed.arc_shape(sep["ball"], sep["velocity"])))
    return result


def historic(path, goal, exclude):
    """Map the original deterministic fragment tags without altering the replayed commands."""
    receipts, archive, qualified = _read(path.with_suffix(".json")), {}, {}
    for i, trial in enumerate(receipts["practice"]):
        if trial["card"]["id"] in exclude:
            continue
        name = trial["source"]
        if name not in archive:
            origin = path.with_name(name)
            if _hash(origin) != receipts["sources"][name]:
                raise ValueError("Practice motor outcome source changed")
            archive[name] = _read(origin)
        d = archive[name]
        row = d["rows"][trial["source_row"]]
        if d["cards"][trial["source_row"]] != trial["card"]:
            raise ValueError("Physically scored labels and original motor fragment differ")
        for impact, _, arc in useful(row):
            if np.isfinite(arc).all() and goal.contains(arc):
                qualified[i, impact] = tuple(arc)
    return qualified


def optical_departure(archive):
    stamp = turn_time([(t, x) for t,x,*_ in archive])
    if stamp is None:
        return None
    seen = [x for t,x,*_ in archive if x is not None and stamp+4 <= t <= stamp+18 and t % 2 == 0
            and x[4] > .9 and -1.48 < x[1] < -.23]
    if len(seen) < 3:
        return None
    w = np.asarray(seen)
    arcs = observed.arc_shape(w[:, :3], w[:, 3:])
    good = arcs[np.isfinite(arcs).all(axis=1)]
    if len(good) < 3:
        return None
    med = np.median(good, axis=0)
    if np.quantile(np.abs(good - med), .70, axis=0)[1] >= .14:
        return None
    return tuple(map(float, med))


class GoalFragments(Carried):
    def __init__(self, *args, goal, excluded, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal = goal
        a = historic(mpc.BANK, goal, excluded)
        b = historic(course.PRACTICE, goal, excluded)
        rows, labels = [], {}
        for x, g, tag in self.rows:
            if tag in a:
                rows.append((x, g, tag))
        self.good = rows
        self.good_tree = cKDTree(np.stack([x for x, *_ in rows])) if rows else None
        for (i, stamp), shape in b.items():
            labels[("carried", course.PRACTICE.stem, i, stamp)] = shape
        data = _read(season.CARRY)
        for i, (card, row) in enumerate(zip(data["cards"], data["rows"], strict=True), 1):
            if card["id"] not in excluded:
                for _, j, arc in useful(row):
                    if np.isfinite(arc).all() and goal.contains(arc):
                        labels["live", i, j] = arc
        self.known = labels

    def new_point(self):
        self.pending = None
        self.lines = []
        self.qualified = 0
        self.proposals = 0
        self.goal_labels = []
        self.confirmed = 0

    def priors(self, x, q, reference, limit):
        stock = super().priors(x, q, reference, limit)
        if limit < 5 or len(stock) < 3 or x[1] < -1.38:
            return stock
        key = mpc.query(x, q, reference)
        opts = []
        if self.good_tree is not None:
            d, ix = self.good_tree.query(key, min(60, len(self.good)))
            opts.extend((float(v), self.good[j][1], self.good[j][2], False)
                        for v,j in zip(np.atleast_1d(d), np.atleast_1d(ix), strict=True))
        opts.extend((max(0., np.linalg.norm(key - s) - .10), gs, tag, True)
                    for s,gs,tag in self.online if tag in self.known or tag[0] == "goal-confirmed")
        if len(opts) < 2:
            return stock
        opts.sort(key=lambda v:v[0])
        ordinary_nearest, _ = self.tree.query(key, 1)
        allowed = min(1.65, max(.77, float(ordinary_nearest) + .43))
        winners, used = [], set()
        stock_key = {g.tobytes() for g,_ in stock}
        for d, gs, tag, live in opts:
            if d > allowed or len(winners) >= min(MAX_RESERVED, limit-3):
                break
            if tag in used or gs.tobytes() in stock_key:
                continue
            winners.append((gs.copy(), live))
            used.add(tag)
        if not winners:
            return stock
        self.proposals += len(winners)
        self.qualified += 1
        return (stock[:3] + winners + stock[3:])[:limit]

    def keep(self, archive, leg):
        arc = optical_departure(archive)
        tag = ("goal-pending", *leg[1:], arc) if arc is not None and self.goal.contains(arc) else leg
        n = super().keep(archive, tag)
        if n and tag[0] == "goal-pending":
            self.pending = tag
            self.lines.clear()
        else:
            self.pending = None
        return n

    def settle(self, tick, fit, x):
        if (self.pending is None or x is None or fit is None or not fit.bounced
                or abs(tick*mpc.DT-fit.time)>1e-8 or not -1.28 < x[1] < -.94 or x[4] > -1.15):
            return
        a = physical_projection(x, True)
        if a is None or not .02 < a[2] < .24:
            return
        self.lines.append((tick, float(a[1])))
        if len(self.lines) < 2:
            return
        (before, old), (now, h) = self.lines[-2:]
        if now-before > 4 or abs(old-h) > .025:
            return
        depth = .5 * (old+h)
        qualified = depth >= SAFETY_MIN
        choice = ("goal-confirmed" if qualified else "goal-observed", *self.pending[1:], depth)
        self.online = deque(((x,g,choice if tag == self.pending else tag) for x,g,tag in self.online),
                            maxlen=self.online.maxlen)
        self.goal_labels.append(dict(at_tick=tick, score=self.pending[2], depth_m=depth,
                                     confirmed=qualified))
        self.confirmed += bool(qualified)
        self.pending = None
        self.lines.clear()


class Player(season.Player):
    def begin(self):
        super().begin()
        if isinstance(self.priors, GoalFragments):
            self.priors.new_point()

    def step(self, tick, eye, x, q, reference, base):
        u = super().step(tick, eye, x, q, reference, base)
        if isinstance(self.priors, GoalFragments):
            self.priors.settle(tick, eye.latest, x)
        return u


def run(a):
    d = _read(PLAN)
    if any(_hash(Path(p)) != h for p,h in d["files"].items()):
        raise ValueError("Goal learning input changed")
    desc = dict(tactic_plan=_hash(PLAN), actor_sha256=_hash(Path(__file__)),
                goal_sha256=_hash(GOAL))
    if a.out.exists() and any(_read(a.out).get(k)!=v for k,v in desc.items()):
        raise ValueError("Checkpoint belongs to different actor")
    goal = tactic.Envelope(_read(GOAL))
    ranker = combined.shot.Coverage(_read(combined.RISK))
    old_play = mpc.play

    def frags(*args, **kw):
        return GoalFragments(*args, goal=goal, excluded=set(d["known"]), **kw)

    def player(*args, **kw):
        return Player(*args, coverage=ranker, **kw)

    def play(*args, **kw):
        row = old_play(*args, **kw)
        p = args[10].priors
        row["goal_retrieval"] = dict(archived_labelled_rows=len(p.good), retained_labels=len(p.known),
                                     rescue_decisions=p.qualified, extra_goal_seeds=p.proposals,
                                     earned_online=p.goal_labels)
        return row

    def writer(path, data):
        _write(path, data | desc)

    a.n, a.cap, a.deck, a.mode, a.carry = len(d["cards"]), 240, "practice", "live", season.CARRY
    a.resume = a.out.exists()
    with (patch.object(course, "PLAN", season.PLAN), patch.object(course, "Carried", frags),
          patch.object(course, "_write", writer), patch.object(mpc, "MPC", player),
          patch.object(mpc, "play", play)):
        course.run(a)


def report(a):
    new, stock = _read(a.out), _read(REF)
    if new["cards"] != stock["cards"][:len(new["cards"])]:
        raise ValueError("Wrong base course")
    rows = []
    for c, r, s in zip(new["cards"], new["rows"], stock["rows"]):
        for z in (r, s):
            if z["shot"] != s["shot"] or any(e["target"] != list(course.draw(c,j,"random",None)[0])
                        for j,e in enumerate(z["placements"]["requested"])):
                raise ValueError("Official setting changed")
        if not rows and r["memory"]["before"] != s["memory"]["before"]:
            raise ValueError("Initial physical motor checkpoint differs")
        rows.append(dict(id=c["id"], control=s["score"]["learner"], own=r["score"]["learner"],
                         tally=r["goal_retrieval"]))
    _write(ROOT / "official-report.json", dict(complete=len(rows)==len(new["cards"]),
            control=sum(r["control"] for r in rows), own=sum(r["own"] for r in rows),
            control_peak=max(r["control"] for r in rows), peak=max(r["own"] for r in rows),
            rows=rows, source_hash=_hash(a.out), exact_draws=True))
    print([(r["id"],r["control"],r["own"],r["tally"]["extra_goal_seeds"]) for r in rows], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan", "run", "report"))
    p.add_argument("--out", type=Path, default=ROOT / "official-live.json.gz")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    {"plan": lambda a: receipt(), "run": run, "report": report}[a.phase](a)
