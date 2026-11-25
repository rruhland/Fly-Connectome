"""Learn which modeled trajectories survive real play and seek replies from uncertain future observations."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt45_composed_feedback as motor
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt46_reliability")
FIT = ROOT / "actual.json"
SOURCE = course.ROOT / "learning-48.json.gz"


def train(out):
    rows = []
    for i, r in enumerate(_read(SOURCE)["rows"]):
        for offer in r["placements"]["delivered"]:
            if not offer["near_bounce"]:
                continue
            start = offer["far_contact"]["step"]
            end = (offer.get("near_contact") or r["events"][-1])["step"]
            visits = [p for p in r["searches"] if start <= p["tick"] < end]
            for phase in range(3):
                group = [p for p in visits if stage(p["p"][1]) == phase]
                if group:
                    rows.append(
                        dict(
                            game=i,
                            phase=phase,
                            agree=group[-1].get("robust", 0),
                            scored=int(offer["near_landed"]),
                        )
                    )
    counts = np.zeros((3, 3, 2), int)
    for r in rows:
        counts[r["phase"], r["agree"]] += [r["scored"], 1]
    _write(out, dict(source=_hash(SOURCE), rows=rows, counts=counts.tolist()))


def stage(y):
    return int(y < -0.81) + int(y < -1.18)


class Reliable(motor.Composed):
    def begin(self):
        super().begin()
        data = np.asarray(_read(FIT)["counts"])
        self.chance = (data[:, :, 0] + 1.0) / (data[:, :, 1] + 2.0)

    def robust(self, goals, rank, nominal):
        agree, quality = self.uncertainty(goals, nominal)
        # Make the cost of a failed stress case depend on what physically
        # happened on earlier receiving attempts at this flight stage.
        reliability = self.chance[self.phase, agree]
        adjusted = quality - 4.5 * reliability
        picked = int(np.argmin(adjusted))
        original = int(np.argmin(quality))
        self.diag["calibrated_edits"] += int(picked != original)
        return int(rank[picked]), int(agree[picked]), float(quality[picked])

    def search(self, tick, x, q, reference, base):
        self.phase = stage(x[1])
        super().search(tick, x, q, reference, base)
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        note = self.notes[-1]
        if (
            note.get("robust", 0) != 1
            or not -1.28 < x[1] < -0.49
            or not self.tick == tick
            or self.best is None
        ):
            return
        steps = 2 if self.phase == 2 else 4
        priors = [self.best.copy(), np.tile(base, (course.mpc.HORIZON, 1))]
        priors.extend(g for g, _ in self.priors.priors(x, q, reference, 2))
        segs = np.stack(priors)[:, : steps + 1].copy()
        t = self.engine.twin
        loc, vel = t.ball.copy(), t.velocity.copy()
        spin = t.data.qvel[t._ball_dof + 3 : t._ball_dof + 6]
        w = spin.copy()
        proposed = {}
        try:
            for s in (-1.0, 1.0):
                t.ball[:] = loc + s * np.array([0.0018, -0.0023, 0.0028])
                t.velocity[:] = vel + s * np.array([0.023, 0.043, 0.073])
                spin[:] = w + [16.0 + 17.0 * s, 0.0, 0.0]
                trace = np.full((*segs.shape[:2], 17), np.nan)
                self.engine.trajectories(
                    segs, course.mpc.AIM, (steps + 1) * course.mpc.DT, traces=trace
                )
                for g in motor.compose(self.priors, trace, priors, steps):
                    proposed.setdefault(g.tobytes(), g)
        finally:
            t.ball[:], t.velocity[:], spin[:] = loc, vel, w
        moves = list(proposed.values())
        if not moves:
            return
        costs, legal = self.test(moves, landing=True)
        ok = np.flatnonzero(legal)
        note["conditional_searched"] = len(moves)
        if not len(ok):
            return
        ids = ok[np.argsort(costs[ok])[:6]]
        votes, val = self.confirm([moves[j] for j in ids], costs[ids])
        valid = np.flatnonzero(votes == 2)
        if len(valid):
            k = valid[np.argmin(val[valid])]
            if val[k] <= note["score"] + 4.5 * (
                self.chance[self.phase, 2] - self.chance[self.phase, 1]
            ):
                self.best = moves[ids[k]]
                self.tick, self.lease = tick, course.mpc.LEASE
                note["conditional_issued"] = dict(votes=2, risk=float(val[k]))
                self.diag["conditional_rescues"] += 1


def run(a):
    if _read(FIT)["source"] != _hash(SOURCE):
        raise ValueError("Calibration changed")
    if {c["id"] for c in _read(SOURCE)["cards"]} & {
        c["id"] for c in _read(course.PLAN)[a.deck]
    }:
        raise ValueError("Calibration overlaps comparison")
    with patch.object(course.mpc, "MPC", Reliable):
        course.run(a)
    d = _read(a.out)
    d.update(reliability=_hash(FIT), source=_hash(Path(__file__)))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "play"))
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="assessment",
        mode="frozen",
        cap=240,
        carry=course.ROOT / "learning-12.json.gz",
        resume=False,
    )
    a = p.parse_args()
    if a.phase == "fit":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        train(a.out)
    else:
        run(a)
