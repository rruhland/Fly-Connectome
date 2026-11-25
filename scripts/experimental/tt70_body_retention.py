"""Measure what physically re-clamping and practicing changes on both bodies."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt68_clamp_transfer as transfer
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt70_body_retention")
PLAN = ROOT / "protocol.json"
course, season, mpc = transfer.course, transfer.season, transfer.mpc


def plan():
    parent = _read(transfer.PLAN)
    exclude = {c["id"] for c in parent["cards"]} | {
        c["id"] for c in _read(season.CARRY)["cards"][:48]
    }
    reserve = {c["id"]: c for c in _read(season.PLAN)["practice"]}
    if len(reserve) != 16 or exclude & set(reserve):
        raise ValueError("Changed independent release pool")
    rng = np.random.default_rng(7068139)
    cards = [
        dict(
            reserve[i],
            seed=int(rng.integers(1, 2**30)),
            reply_seed=int(rng.integers(1, 2**30)),
        )
        for i in rng.permutation(sorted(reserve))
    ]
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        PLAN,
        dict(
            cards=cards,
            parent=_hash(transfer.PLAN),
            source=_hash(season.CARRY),
            body="registered",
            comparison="Both frozen memories play both fixtures on the exact same 16 fresh measured independent openings and fresh opponent draws. The original memory had 48 ordinary original-fixture games. After transfer adds exactly 32 unreset, learning-enabled points on a 12 cm translated end clamp. Body-relative storage enables assessment after returning to center.",
        ),
    )


def run(body, period, feeds, *, acquired=None, destination=None):
    p = _read(PLAN)
    prior = season.CARRY if period == "before" else transfer.ROOT / "registered.json.gz"
    if acquired is not None:
        prior = acquired
    docs = _read(prior)
    expected = (
        len(docs["rows"]) if acquired is not None else 48 if period == "before" else 32
    )
    if len(docs["rows"]) != expected or (
        period == "after" and acquired is None and docs["actor"] != "registered"
    ):
        raise ValueError("Required learned boundary is not complete")
    if p["parent"] != _hash(transfer.PLAN) or p["source"] != _hash(season.CARRY):
        raise ValueError("Parent changed")
    cards = p["cards"]
    ids = {c["id"] for c in cards}
    if {c["id"] for c in docs["cards"][:expected]} & ids:
        raise ValueError("Repeated assessment releases")
    with np.load(mpc.BANK) as f:
        raw = mpc.DeepField(f, "local")
    if body == "lateral":
        raw = transfer.fixture.BodyRelativeRaw(raw, transfer.DX)
    rig_ready = (
        _read(transfer.PLAN)["ready"]
        if body == "lateral"
        else _read(mpc.BANK.with_suffix(".json"))["ready"]
    )
    doc = dict(
        protocol=_hash(PLAN),
        source=_hash(prior),
        cards=cards,
        ready=rig_ready,
        fixture=body,
        period=period,
        rows=[],
    )
    output = (
        destination if destination is not None else ROOT / f"{body}-{period}.json.gz"
    )
    if output.exists():
        saved = _read(output)
        if any(saved[k] != doc[k] for k in doc if k != "rows"):
            raise ValueError("Wrong prior assessment")
        doc = saved
    shots = mpc.measured_launches(feeds)
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with (
        transfer.enabled(),
        transfer.fixture.slid(transfer.DX) if body == "lateral" else nullcontext(),
        mpc.centered(rig_ready),
    ):
        a = mpc.arena()
        camera, rig, stride, low, high = mpc.resources(a)
        try:
            with patch.object(
                mpc,
                "Fragments",
                lambda file, engine: course.Carried(
                    file, engine, practice=[course.PRACTICE], held=ids
                ),
            ):
                cls = transfer.Registered if body == "lateral" else season.Player
                actor = cls(mpc.BANK, raw, variant="elastic", budget=28, coverage=rank)
            try:
                actor.priors.home = np.asarray(rig_ready)
                origin = (
                    dict(docs, rows=[None] * 48 + docs["rows"])
                    if period == "after"
                    else docs
                )
                course.resume(actor, prior.with_suffix("").with_suffix(".npz"), origin)
                before = actor.priors.digest()
                actor.serial += len(doc["rows"])
                for i in range(len(doc["rows"]), len(cards)):
                    c = cards[i]
                    r = mpc.play(
                        a,
                        camera,
                        rig,
                        stride,
                        low,
                        high,
                        rig_ready,
                        c,
                        shots[c["id"]],
                        raw,
                        actor,
                        course.OPTIC,
                        "frozen",
                        240,
                        20,
                    )
                    if r["memory"]["after"] != before or any(
                        v["target"] != list(transfer.draw(c, j, "random", None)[0])
                        or v["draw"] != "random"
                        for j, v in enumerate(r["placements"]["requested"])
                    ):
                        raise ValueError("Drift in frozen actor or opponent")
                    doc["rows"].append(r)
                    _write(output, doc)
                    print(i + 1, c["id"], r["score"]["learner"], flush=True)
            finally:
                actor.close()
        finally:
            camera.close()
            a.close()


def report():
    from . import tt53_responsive_report as measurements

    ans = {}
    for body in ("original", "lateral"):
        paths = {
            k: ROOT / f"{body}-{label}.json.gz"
            for k, label in (("before", "before"), ("responsive", "after"))
        }
        v = measurements.compare(paths, 16)
        v["groups"]["after"] = v["groups"].pop("responsive")
        ans[body] = v
    _write(ROOT / "assessment.json", ans)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("body", choices=("plan", "original", "lateral", "report"))
    p.add_argument("period", nargs="?", choices=("before", "after"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    if a.body == "plan":
        plan()
    elif a.body == "report":
        report()
    elif a.period:
        run(a.body, a.period, a.feeds)
    else:
        p.error("Each assessment needs a checkpoint period")
