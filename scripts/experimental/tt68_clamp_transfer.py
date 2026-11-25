"""Full unreset continuous-random points on a 12 cm lateral re-clamp with carried motor memory."""

import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt52_practice_dose as dose
from . import tt63_lateral_clamp as fixture
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt39_opportunity_map import projected_height
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt68_clamp_transfer")
PLAN = ROOT / "protocol.json"
course = dose.course
season = dose.season
mpc = course.mpc
DX = 0.12


def plan():
    d = _read(fixture.GEOMETRY)
    ready = next(r["ready"] for r in d["rows"] if r["dx"] == DX)
    stock = dose.ROOT / "live-from-48.json.gz"
    cards = _read(stock)["cards"][:32]
    if (
        cards != _read(dose.PLAN)["practice"][:32]
        or len({c["id"] for c in cards}) != 32
    ):
        raise ValueError("Baseline roster changed")
    ROOT.mkdir(parents=True, exist_ok=True)
    _write(
        PLAN,
        dict(
            dx=DX,
            ready=ready,
            cards=cards,
            source_geometry=_hash(fixture.GEOMETRY),
            carry=_hash(season.CARRY),
            baseline=str(stock),
            baseline_hash=_hash(stock),
            rule="32 different measured openings. Both shifted-body arms start with the same 48-game central-clamp memory and learn on every real return. World mode retains original world queries. Registered mode shifts prior motion queries into the calibrated base frame and stores newly paid motion in that frame. Both use the same new known mechanical body in their private model, calibrated home, physical opponent, original full continuous law and serving distribution. All points count.",
        ),
    )


def shifted(x):
    if x is None:
        return None
    x = x.copy()
    x[0] -= DX
    return x


class Registered(season.Player):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        lookup = self.priors.priors
        self.priors.priors = lambda x, q, r, limit: lookup(shifted(x), q, r, limit)
        keep = self.priors.keep

        def remember(tape, tag):
            return keep(
                [(t, shifted(x), q, ref, sent) for t, x, q, ref, sent in tape], tag
            )

        self.priors.keep = remember

    def begin(self):
        super().begin()
        self._temporal_predict = self.motor.predict

        def predict(f):
            g = f.copy()
            # Time differences survive a rigid base displacement.
            g[0] -= DX / season.fresh.servo.SCALES["local"][0]
            return self._temporal_predict(g)

        self.motor.predict = predict


def run(kind, feeds):
    design = _read(PLAN)
    if (
        _hash(season.CARRY) != design["carry"]
        or _hash(fixture.GEOMETRY) != design["source_geometry"]
    ):
        raise ValueError("Fixture or prior changed")
    cards = design["cards"]
    ready = design["ready"]
    shots = mpc.measured_launches(feeds)
    root = ROOT / (kind + ".json.gz")
    bank = root.with_suffix("").with_suffix(".npz")
    prior = np.asarray(_read(mpc.BANK.with_suffix(".json"))["ready"])
    with np.load(mpc.BANK) as f:
        raw = mpc.DeepField(f, "local")
    if kind == "registered":
        raw = fixture.BodyRelativeRaw(raw, DX)
    doc = dict(
        cards=cards,
        ready=ready,
        original_ready=prior.tolist(),
        fixture_dx_m=DX,
        protocol=_hash(PLAN),
        mode="live",
        actor=kind,
        cap=240,
        rows=[],
    )
    scorer = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    ids = {c["id"] for c in cards}
    base = _read(season.CARRY)
    if ids & {c["id"] for c in base["cards"][: len(base["rows"])]}:
        raise ValueError("Carry leakage")
    with enabled(), fixture.slid(DX), mpc.centered(ready):
        a = mpc.arena()
        cam, rig, stride, low, high = mpc.resources(a)
        try:
            with patch.object(
                mpc,
                "Fragments",
                lambda file, engine: course.Carried(
                    file, engine, practice=[course.PRACTICE], held=ids
                ),
            ):
                cls = Registered if kind == "registered" else season.Player
                actor = cls(
                    mpc.BANK, raw, variant="elastic", budget=28, coverage=scorer
                )
            try:
                actor.priors.home = np.array(ready)
                course.resume(
                    actor, season.CARRY.with_suffix("").with_suffix(".npz"), base
                )
                if root.exists():
                    saved = _read(root)
                    if saved["protocol"] != doc["protocol"] or saved["actor"] != kind:
                        raise ValueError("Incompatible resume")
                    doc = saved
                    course.resume(
                        actor, bank, dict(doc, rows=[None] * 48 + doc["rows"])
                    )
                for i in range(len(doc["rows"]), len(cards)):
                    c = cards[i]
                    r = mpc.play(
                        a,
                        cam,
                        rig,
                        stride,
                        low,
                        high,
                        ready,
                        c,
                        shots[c["id"]],
                        raw,
                        actor,
                        course.OPTIC,
                        "live",
                        240,
                        20,
                    )
                    for j, g in enumerate(r["placements"]["requested"]):
                        if (
                            g["draw"] != "random"
                            or list(draw(c, j, "random", None)[0]) != g["target"]
                        ):
                            raise ValueError("Opponent distribution changed")
                    doc["rows"].append(r)
                    doc["checkpoint_sha256"] = course.save(actor, bank)
                    _write(root, doc)
                    print(i + 1, c["id"], r["score"]["learner"], flush=True)
            finally:
                actor.close()
        finally:
            cam.close()
            a.close()


def describe(rows):
    score = np.array([r["score"]["learner"] for r in rows])
    counts = Counter()
    for r in rows:
        for o in r["placements"]["delivered"]:
            l = o.get("receive_line")
            if not l:
                continue
            z = projected_height(l)
            x = abs(l["xyz"][0])
            name = ("center" if x < 0.10 else "wing") + (
                "_low" if z < 1.0 else "_other"
            )
            counts[name + "_offered"] += 1
            counts[name + "_returned"] += int(bool(o["near_landed"]))
    return dict(
        games=len(rows),
        own=int(score.sum()),
        median=float(np.median(score)),
        best=int(max(score)),
        zero=int(sum(score == 0)),
        at_least_15=int(sum(score >= 15)),
        coverage=dict(counts),
    )


def report():
    d = _read(PLAN)
    paths = {
        "stock": Path(d["baseline"]),
        "world": ROOT / "world.json.gz",
        "registered": ROOT / "registered.json.gz",
    }
    if _hash(paths["stock"]) != d["baseline_hash"]:
        raise ValueError("Changed original course")
    ans = {}
    for k, p in paths.items():
        doc = _read(p)
        if doc["cards"][:32] != d["cards"] or len(doc["rows"]) < 32:
            raise ValueError("Unfinished or altered course")
        ans[k] = describe(doc["rows"][:32])
    _write(
        ROOT / "comparison.json",
        dict(groups=ans, hashes={k: _hash(v) for k, v in paths.items()}),
    )


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("op", choices=("plan", "world", "registered", "report"))
    cmd.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = cmd.parse_args()
    if a.op == "plan":
        plan()
    elif a.op == "report":
        report()
    else:
        run(a.op, a.feeds)
