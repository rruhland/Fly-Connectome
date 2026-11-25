"""Prospective continuous points with the previously frozen TT59 repair readout."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt59_fewshot_joints as repair
from . import tt48_contact_patch_report as check
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt40_long_low import same_trace

school = repair.school
mpc = school.mpc
course = school.course
season = school.season
PLAN = repair.ROOT / "games.protocol.json"
OUT = repair.ROOT / "games.json.gz"
MODES = ("stock", *repair.MODES)
COUNT = 4


class Player(season.Player):
    def __init__(self, *args, readout, mode, limits, **kwargs):
        super().__init__(*args, **kwargs)
        self.readout = readout
        self.mode = mode
        self.limits = limits

    def begin(self):
        super().begin()
        self.examined = False
        self.pending = None
        self.pending_tick = -1000
        self.repairs = []

    def search(self, tick, x, q, ref, base):
        super().search(tick, x, q, ref, base)
        feature = school.context(x, q, ref, self.engine.rate, self.engine.bounced)
        if self.examined or not repair.visible(x, feature) or not self.notes or self.notes[-1]["tick"] != tick:
            return
        self.examined = True
        if self.notes[-1].get("robust", 0):
            return
        if not self.engine.sync():
            return
        anchors = repair.anchor_paths(self.engine, self.raw, self.priors.home, x, q, ref, base)
        paths, details = self.readout.proposals(x, feature, anchors, self.mode)
        options, diag = repair.rank_paths(self.engine, paths, ref, details)
        # The fork can spend several real attempts on a replayed miss.
        # Continuous games get a single action, require proxy robustness,
        # and never use an archived future command or previously failed tape.
        robust = [(u, d) for u, d in options if d["legal"] and d["robust"] > 0]
        row = dict(tick=tick, p=x[:3].tolist(), issued=bool(robust), **diag)
        if robust:
            rates, chosen = min(robust, key=lambda r: r[1]["score"])
            self.pending = school.atlas.reference_path(ref, rates, *self.limits)[1:]
            self.pending_tick = tick
            self.best, self.tick, self.lease = self.pending, tick, repair.H
            row["selection"] = chosen
        self.repairs.append(row)

    def step(self, tick, eye, x, q, ref, base):
        if x is not None and x[4] > .9:
            self.examined = False
            self.pending = None
        if self.pending is not None and 0 <= tick-self.pending_tick < repair.H:
            self.engine.observe(tick, eye, q, ref.copy())
            return self.pending[tick-self.pending_tick]
        return super().step(tick, eye, x, q, ref, base)


def freeze(feeds):
    p = repair.validate_plan()
    old = _read(course.PLAN)
    banned = set(p["excluded"]) | {r["id"] for r in p["cases"]}
    banned.update(c["id"] for group in ("practice", "assessment") for c in old[group])
    for path in (mpc.BANK, course.PRACTICE):
        banned.update(r["card"]["id"] for r in _read(path.with_suffix(".json"))["practice"])
    with np.load(school.atlas.MODEL) as f:
        banned.update(f["release"].tolist())
    candidates = [c for c in _read(course.SOURCE)["practice"]
                  if c["id"] not in banned and c["opening"] == "random" and c["profile"] == "reach"]
    if len(candidates) < COUNT:
        raise ValueError(f"Only {len(candidates)} distinct fresh measured releases")
    rng = np.random.default_rng(590739)
    shots = mpc.measured_launches(feeds)
    cases = []
    for index in rng.permutation(len(candidates))[:COUNT]:
        card = dict(candidates[index], seed=int(rng.integers(1, 2**30)),
                    reply_seed=int(rng.integers(1, 2**30)))
        cases.append(dict(card=card, shot=shots[card["id"]]))
    fixed = (repair.PLAN, repair.MODEL, course.OPTIC, season.CARRY,
             season.CARRY.with_suffix("").with_suffix(".npz"), season.fresh.RISK, mpc.BANK, course.PRACTICE)
    _write(PLAN, dict(schema="tt59-prospective-continuous-points-1", cases=cases,
                      sources={str(f): _hash(f) for f in fixed}, measured_launch_source=_hash(feeds),
                      modes=MODES, cap=10000,
                      ready=_read(mpc.BANK.with_suffix(".json"))["ready"],
                      guard="Frozen standard player, except an early unsupported nominal stock search may be replaced once per offer by a correction rated nominal legal with ≥1 of two public model perturbation votes. One measured launch per point and no resets. Ordinary independent random opponent draws throughout. No post hoc choice of trial or replay-based action selection."))
    print("continuous roster", [r["card"]["id"] for r in cases], flush=True)


def compare(before, after):
    compact = []
    for a, b in zip(before["placements"]["delivered"], after["placements"]["delivered"]):
        if same_trace(a["far_contact"], b["far_contact"]) and same_trace(a["near_bounce"], b["near_bounce"]):
            if a["near_landed"] != b["near_landed"]:
                compact.append(dict(leg=a["leg"], stock=a["near_landed"], adjusted=b["near_landed"]))
    return compact


def games():
    plan = _read(PLAN)
    if any(_hash(Path(p)) != h for p, h in plan["sources"].items()):
        raise ValueError("Changed prospective whole-game plan")
    result = dict(protocol=_hash(PLAN), operator=_hash(Path(__file__)), rows=[], complete=False)
    if OUT.exists():
        old = _read(OUT)
        if any(old[k] != result[k] for k in ("protocol", "operator")):
            raise ValueError("Mismatched game receipt")
        result = old
    ready = np.asarray(plan["ready"])
    releases = {r["card"]["id"] for r in plan["cases"]}
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    carry = _read(season.CARRY)
    if releases & {c["id"] for c in carry["cards"][:len(carry["rows"])]}:
        raise ValueError("Model checkpoint was exposed to a prospective release")
    with np.load(mpc.BANK) as f:
        raw = mpc.DeepField(f, "local")
    readout = repair.Corrections()
    with mpc.centered(ready), school.enabled():
        a = mpc.arena()
        cam, rig, stride, lo, hi = mpc.resources(a)
        try:
            for c in plan["cases"][len(result["rows"]):]:
                card, shot = c["card"], c["shot"]
                rows = {}
                for mode in MODES:
                    with patch.object(mpc, "Fragments", lambda f,e: Carried(f,e,practice=[course.PRACTICE],held=releases)):
                        kw = dict(coverage=rank, variant="elastic", budget=28)
                        if mode == "stock":
                            actor = season.Player(mpc.BANK, raw, **kw)
                        else:
                            actor = Player(mpc.BANK, raw, readout=readout, mode=mode,
                                           limits=(stride, lo, hi), **kw)
                    try:
                        course.resume(actor, season.CARRY.with_suffix("").with_suffix(".npz"), carry)
                        r = mpc.play(a, cam, rig, stride, lo, hi, ready, card, shot, raw,
                                     actor, course.OPTIC, "frozen", plan["cap"])
                        if r["memory"]["before"] != r["memory"]["after"]:
                            raise ValueError("Frozen stock bank changed")
                        for leg, step in enumerate(r["placements"]["requested"]):
                            if step["target"] != list(draw(card, leg, "random", None)[0]):
                                raise ValueError("Opponent law changed")
                        r["repairs"] = getattr(actor, "repairs", [])
                        r["complete_point"] = r["events"][-1]["kind"] == "point"
                        if mode != "stock":
                            r["same_offer_changes"] = compare(rows["stock"], r)
                        rows[mode] = r
                    finally:
                        actor.close()
                    print(card["id"], mode, r["score"]["learner"],
                          len(r["actions"]), sum(v["issued"] for v in r["repairs"]), flush=True)
                result["rows"].append(dict(c, arms=rows))
                result["complete"] = len(result["rows"]) == len(plan["cases"])
                _write(OUT, result)
        finally:
            cam.close()
            a.close()
    # Independent renderer-free full-tape replay, including all CAD samples.
    with school.enabled(cameras=False):
        for row in result["rows"]:
            for r in row["arms"].values():
                if "replay" not in r:
                    r["replay"] = check.replay(ready, row["card"], r)
        _write(OUT, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "run"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    freeze(a.feeds) if a.mode == "plan" else games()


if __name__ == "__main__":
    main()
