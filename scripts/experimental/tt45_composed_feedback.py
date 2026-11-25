"""Search early joint commands by what learned controls the arm can execute from their predicted result."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write

mpc = course.mpc


def compose(priors, traces, prefixes, steps):
    """Query new movements at future observed coordinates, separately for each motor prefix."""
    goals = []
    for trace, seq in zip(traces, prefixes, strict=True):
        future = trace[steps]
        if not np.isfinite(future).all() or future[4] >= -0.9:
            continue
        # Traces contain pre-command encoder values but end-of-tick references.
        ref = trace[steps - 1, 11:]
        for suffix, _ in priors.priors(future[:6], future[6:11], ref, 5):
            goals.append(np.vstack([seq[:steps], suffix]))
    return goals


class Composed(servo.Temporal):
    def test(self, goals, *, landing):
        frames = len(goals[0])
        size = frames + (mpc.TAIL if landing else 0)
        batch = np.array([mpc.tail(g, size) for g in goals])
        f = np.zeros((len(batch), 9))
        _, y = self.engine.trajectories(batch, mpc.AIM, frames * mpc.DT, feedback=f)
        self.diag["composed_rollouts"] += len(batch)
        return mpc.cost(y, f)

    def confirm(self, goals, losses):
        plant = self.engine.twin
        spin = plant.data.qvel[plant._ball_dof + 3 : plant._ball_dof + 6]
        p, v, w = plant.ball.copy(), plant.velocity.copy(), spin.copy()
        votes = np.zeros(len(goals), dtype=int)
        value = 0.43 * losses
        try:
            for sign in (-1.0, 1.0):
                plant.ball[:] = p + sign * np.array([0.0018, -0.0023, 0.0028])
                plant.velocity[:] = v + sign * np.array([0.023, 0.043, 0.073])
                spin[:] = w + [16.0 + 17.0 * sign, 0.0, 0.0]
                loss, legal = self.test(goals, landing=True)
                votes += legal
                value += 0.285 * np.where(
                    legal, loss, 2.75 + 0.12 * np.minimum(loss, 10.0)
                )
        finally:
            plant.ball[:], plant.velocity[:], spin[:] = p, v, w
        return votes, value

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        note = self.notes[-1]
        if note.get("robust", 0) > 0 or not -1.13 < x[1] < -0.49:
            return
        steps = 6 if x[1] > -0.78 else 3
        constant = np.tile(base, (mpc.HORIZON, 1))
        prefixes = [constant]
        if self.best is not None and 0 <= tick - self.tick < mpc.HORIZON - 2:
            prefixes.append(mpc.tail(self.best[tick - self.tick :], mpc.HORIZON))
        for seq, _ in self.priors.priors(x, q, reference, 4):
            if all(
                np.max(abs(seq[:steps, :5] - other[:steps, :5])) > 0.025
                for other in prefixes
            ):
                prefixes.append(seq)
        # Only carry the prefix until a new prior will be queried.
        segs = np.asarray(prefixes)[:, : steps + 1].copy()
        trace = np.full((*segs.shape[:2], 17), np.nan)
        self.engine.trajectories(segs, mpc.AIM, (steps + 1) * mpc.DT, traces=trace)
        candidates = compose(self.priors, trace, prefixes, steps)
        if not candidates:
            return
        note["composed_candidates"] = len(candidates)
        # Full landing trials are decisive. Cheap misses can still be forecast
        # contacts if the ball reaches the bat at the end of the longer action.
        costs, legal = self.test(candidates, landing=True)
        ok = np.flatnonzero(legal)
        note["composed_legal"] = len(ok)
        if not len(ok):
            return
        ids = ok[np.argsort(costs[ok])[:4]]
        votes, value = self.confirm([candidates[k] for k in ids], costs[ids])
        robust = np.flatnonzero(votes > 0)
        if len(robust):
            selected = robust[np.argmin(value[robust] + 2.0 * (2 - votes[robust]))]
            self.best = candidates[ids[selected]]
            self.tick, self.lease = tick, mpc.LEASE
            note["composed_issued"] = dict(
                prefix=steps, votes=int(votes[selected]), cost=float(value[selected])
            )
            self.diag["composed_issued"] += 1


def run(a):
    with patch.object(mpc, "MPC", Composed):
        course.run(a)
    result = _read(a.out)
    result.update(
        composition=_hash(Path(__file__)), temporal=_hash(servo.ROOT / "prior.npz")
    )
    _write(a.out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
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
    run(p.parse_args())
