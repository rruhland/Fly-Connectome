"""Reconsider already-discovered motor options after a new camera exposure during execution."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt39_carry_joints as carry
from .tt4_transfer_trial import _read, _write


class Feedback(mpc.MPC):
    def begin(self):
        super().begin()
        self.pool = []
        self.window = None
        self.collecting = False

    def trial(self, goals, *, long=False, traces=False):
        result = super().trial(goals, long=long, traces=traces)
        if self.collecting and not long and len(goals) >= 7:
            cost = result[0]
            for j in np.argsort(cost)[:12]:
                tape = np.asarray(goals[j])
                if all(
                    np.max(abs(tape[::4, :5] - s[::4, :5])) > 0.04 for s in self.pool
                ):
                    self.pool.append(tape.copy())
        return result

    def search(self, tick, x, q, reference, base):
        self.window = None
        self.pool = []
        self.collecting = True
        try:
            super().search(tick, x, q, reference, base)
        finally:
            self.collecting = False
        if self.engine.bounced and -1.48 < x[1] < -0.62 and self.pool:
            frame = self.engine.latest
            self.window = (tick, frame.time)

    def step(self, tick, eye, x, q, reference, base):
        desired = super().step(tick, eye, x, q, reference, base)
        if self.window is None or x is None or tick % 2 or x[4] >= -0.9 or x[1] < -1.56:
            return desired
        start, last = self.window
        age = tick - start
        if not 2 <= age <= 4 or self.engine.latest.time <= last + 0.004:
            return desired
        self.window = None
        if not self.engine.sync():
            return desired
        if self.best is not None and 0 <= tick - self.tick < self.lease:
            original = mpc.tail(self.best[tick - self.tick :], mpc.HORIZON)
        else:
            original = np.broadcast_to(desired, (mpc.HORIZON, 6)).copy()
        # Roll every motor suffix from the same visual/encoder state *after*
        # the commands actually issued. Unexecuted first controls are discarded.
        goals = [original, *[mpc.tail(g[age:], mpc.HORIZON) for g in self.pool[:8]]]
        costs, legal, *_ = self.trial(goals, long=True)
        self.diag["second_exposures"] += 1
        if legal[0]:
            return desired
        valid = np.flatnonzero(legal)
        if not len(valid):
            return desired
        shortlist = valid[np.argsort(costs[valid])[:3]]
        subset = [original, *[goals[j] for j in shortlist]]
        agree, risk = self.uncertainty(subset, costs[np.r_[0, shortlist]])
        eligible = np.flatnonzero((agree[1:] == 2) & (risk[1:] < risk[0] - 0.5))
        if not len(eligible):
            return desired
        j = shortlist[eligible[np.argmin(risk[1:][eligible])]]
        self.best = goals[j]
        self.tick = tick
        self.lease = mpc.LEASE
        self.diag["fresh_camera_reselections"] += 1
        return self.best[0]


def run(a):
    with patch.object(mpc, "MPC", Feedback):
        carry.run(a)
    d = _read(a.out)
    d["contingent_control"] = (
        "On the first new stereo exposure after 2 ticks, check a maximum of eight cached motor suffixes from actual encoder and optical state; switch only when the issued plan now fails and an alternative lands across both optical perturbations."
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        practice=[mpc.ROOT / "tt37/camera-time-school.npz"], late=False, capacity=768
    )
    run(p.parse_args())
