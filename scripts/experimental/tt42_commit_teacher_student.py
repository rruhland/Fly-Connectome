"""Offer physically earned teacher fragments to the unchanged optical direct motor search."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt41_bounce_joint import OPTIC, PRACTICE
from .tt42_commit_teacher import OUT, PLAN

FRAGMENTS = mpc.Fragments


class WithTeacher(Carried):
    """Replace at most two proposal slots using actual scored optical practice."""

    def __init__(self, bank, engine, teacher, *, held, use):
        super().__init__(bank, engine, practice=[PRACTICE], held=held)
        self.teacher = teacher
        self.use = use
        self.admissions = []
        if not np.array_equal(self.home, teacher.home):
            raise ValueError("Unmatched teacher reference coordinates")

    def priors(self, x, q, reference, limit):
        stock = super().priors(x, q, reference, limit)
        if not self.use or x[4] > -0.9 or len(self.teacher.rows) == 0 or len(stock) < 3:
            return stock
        key = mpc.query(x, q, reference)
        ds, ids = self.teacher.tree.query(key, min(len(self.teacher.rows), 140))
        proposals = []
        used = set()
        for d, i in zip(np.atleast_1d(ds), np.atleast_1d(ids), strict=True):
            if d > 2.65:
                break
            context, seq, label = self.teacher.rows[i]
            if label in used:
                continue
            # Move only a quarter of the measured reference lag, decaying
            # through the received strike. Do not copy body pose blindly.
            physical = (context[6:11] * mpc.OPEN_NORM[6:11]) + (
                context[11:] * mpc.OPEN_NORM[11:]
            )
            offset = (reference[:5] - physical).clip(-0.24, 0.24)
            adjusted = seq.copy()
            adjusted[:, :5] += (
                0.28 * (1.0 - np.arange(mpc.HORIZON)[:, None] / mpc.HORIZON) * offset
            )
            proposals.append((adjusted, True))
            used.add(label)
            self.admissions.append(float(d))
            if len(proposals) >= min(2, max(0, len(stock) - 2)):
                break
        return stock[: len(stock) - len(proposals)] + proposals


def run(a):
    practice = _read(a.teacher.with_suffix(".json"))
    plan = _read(a.plan)
    cards = plan["assessment"][: a.limit]
    held = {c["id"] for c in cards}
    if held & {r["card"]["id"] for r in practice["practice"]}:
        raise ValueError("Relevant teacher experience reused the held release")
    ready = _read(mpc.BANK.with_suffix(".json"))["ready"]
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    shots = mpc.measured_launches(a.feeds)
    doc = {
        "role": "held ordinary RGB/encoder control only",
        "teacher_camera_bank_sha256": _hash(a.teacher),
        "plan_sha256": _hash(a.plan),
        "cards": cards,
        "cap": a.cap,
        "rows": [],
    }
    with mpc.centered(ready):
        court = mpc.arena(protocol="random")
        res = mpc.resources(court)
        try:
            for c in cards:
                r = {"card": c, "modes": {}}
                for use in (False, True):

                    def make(bank, engine, treatment=use):
                        return WithTeacher(
                            bank,
                            engine,
                            FRAGMENTS(a.teacher, engine),
                            held=held,
                            use=treatment,
                        )

                    with patch.object(mpc, "Fragments", make):
                        model = mpc.MPC(mpc.BANK, raw, variant="elastic", budget=28)
                    tag = "earned" if use else "stock"
                    try:
                        game = mpc.play(
                            court,
                            *res,
                            ready,
                            c,
                            shots[c["id"]],
                            raw,
                            model,
                            OPTIC,
                            "frozen",
                            a.cap,
                        )
                        game["marker_free_teacher_candidates"] = model.priors.admissions
                    finally:
                        model.close()
                    r["modes"][tag] = game
                    print(
                        c["id"],
                        tag,
                        game["score"]["learner"],
                        len(game["marker_free_teacher_candidates"]),
                        flush=True,
                    )
                one, two = (r["modes"][m] for m in ("stock", "earned"))
                at = next(
                    (
                        i
                        for i, (x, y) in enumerate(zip(one["actions"], two["actions"]))
                        if x != y
                    ),
                    None,
                )
                tick = len(one["actions"]) if at is None else at
                p = lambda s, end=tick: [e for e in s["events"] if e["step"] <= end]
                if (
                    digest_json(p(one)) != digest_json(p(two))
                    or one["memory"]["before"] != two["memory"]["before"]
                ):
                    raise ValueError(
                        "Paired held controller changed before intervention"
                    )
                r["first_action_change"] = at
                doc["rows"].append(r)
                _write(a.out, doc)
        finally:
            res[0].close()
            court.close()
    doc["complete"] = True
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, nargs="?", default=OUT / "held-camera.json.gz")
    p.add_argument("--teacher", type=Path, default=OUT / "earned-camera.npz")
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--cap", type=int, default=200)
    a = p.parse_args()
    run(a)
