"""Construct direct joint schedules between earned alternatives whose modeled errors point in opposite directions."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt45_composed_feedback as compose
from .tt4_transfer_trial import _hash, _read, _write

mpc = course.mpc


def bracket(commands, errors, close, labels):
    candidates = []
    for i in range(len(commands)):
        for j in range(i):
            if labels[i] != labels[j] or not (
                np.isfinite(errors[i]).all() and np.isfinite(errors[j]).all()
            ):
                continue
            delta = errors[j] - errors[i]
            step = -(errors[i] @ delta) / (0.0001 + delta @ delta)
            if not 0.12 < step < 0.88:
                continue
            improved = np.linalg.norm(errors[i] + step * delta)
            advantage = (
                min(np.linalg.norm(errors[i]), np.linalg.norm(errors[j])) - improved
            )
            if advantage <= 0.035:
                continue
            # A broad interpolation between unrelated action modes circles
            # through an unfamiliar state. Prefer neighboring movements.
            diff = np.sqrt(np.mean((commands[i][::3, :5] - commands[j][::3, :5]) ** 2))
            priority = (
                improved
                + 0.75 * diff
                + 0.2 * min(close[i], close[j])
                - 0.25 * advantage
            )
            candidates.append((priority, i, j, step))
    result = []
    used = set()
    for _, i, j, step in sorted(candidates):
        if i in used and j in used:
            continue
        used.update((i, j))
        for w in (step, 0.5 * (step + 0.5)):
            result.append((1 - w) * commands[i] + w * commands[j])
        if len(result) >= 12:
            break
    return result


class Actor(compose.Composed):
    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        note = self.notes[-1]
        if note.get("robust", 0) == 2 or note.get("composed_issued") or x[1] < -1.37:
            return
        qs = self.priors.priors(x, q, reference, 10)
        if len(qs) < 2:
            return
        schedules = [s for s, _ in qs]
        if self.best is not None and 0 <= tick - self.tick < mpc.HORIZON - 2:
            schedules.insert(0, mpc.tail(self.best[tick - self.tick :], mpc.HORIZON))
        batch = np.ascontiguousarray(schedules)
        fb = np.zeros((len(batch), 9))
        close = np.zeros((len(batch), 19))
        self.engine.predict.trajectories(
            batch, mpc.AIM, mpc.HORIZON * mpc.DT, feedback=fb, approaches=close
        )
        hit = fb[:, 1] > 0.5
        # For missed movements, bracket tangential ball/bat displacements.
        # For contacts, bracket outgoing flight errors instead. Combining
        # across this discontinuity would give a false interpolation target.
        normal = close[:, 7:10]
        offset = close[:, 1:4] - close[:, 4:7]
        slide = offset - normal * (offset * normal).sum(axis=1)[:, None]
        departure = fb[:, 2:5]
        position = fb[:, 5:8]
        duration = 0.94 * np.maximum(
            0.1,
            (
                departure[:, 2]
                + np.sqrt(
                    np.maximum(
                        0, departure[:, 2] ** 2 + 19.62 * (position[:, 2] - 0.779)
                    )
                )
            )
            / 9.81,
        )
        predicted = position[:, :2] + departure[:, :2] * duration[:, None]
        net_t = -position[:, 1] / np.maximum(0.55, 0.94 * departure[:, 1])
        net = position[:, 2] + net_t * departure[:, 2] - 4.905 * net_t**2
        error = np.where(
            hit[:, None],
            np.c_[predicted[:, 0], predicted[:, 1] - 0.65, np.minimum(net - 1.00, 0.0)],
            slide / 0.10,
        )
        error[~hit & (close[:, 0] <= 0)] = np.nan
        candidates = bracket(batch, error, fb[:, 0], hit)
        if not candidates:
            return
        loss, legal, _, _, _ = self.trial(candidates, long=True)
        ids = np.flatnonzero(legal)
        note["corrective_mixes"] = len(candidates)
        note["legal_mixes"] = len(ids)
        if not len(ids):
            return
        ids = ids[np.argsort(loss[ids])[:4]]
        votes, val = self.uncertainty([candidates[i] for i in ids], loss[ids])
        allowed = np.flatnonzero(votes > note.get("robust", 0))
        if len(allowed):
            k = allowed[np.argmin(val[allowed] - 1.1 * votes[allowed])]
            self.best = candidates[ids[k]]
            self.tick, self.lease = tick, mpc.LEASE
            note["mix_issued"] = int(votes[k])
            self.diag["mix_issued"] += 1


def run(a):
    with patch.object(mpc, "MPC", Actor):
        course.run(a)
    d = _read(a.out)
    d.update(mix_source=_hash(Path(__file__)))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", default=8, type=int)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="frozen",
        deck="assessment",
        carry=course.ROOT / "learning-12.json.gz",
        cap=240,
        resume=False,
    )
    a = p.parse_args()
    a.out.parent.mkdir(exist_ok=True, parents=True)
    run(a)
