"""Move a rewarded joint trajectory toward a new optical approach using the nominal body."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory as memory
from .tt4_transfer_trial import _read, _write
from .tt31_openings import evaluate


class SpatialRecall:
    def __init__(self, bank, planner, sign):
        self.bank = bank
        self.planner = planner
        self.sign = sign
        self.offered = 0

    @property
    def examples(self):
        return self.bank.examples

    def add(self, *example):
        self.bank.add(*example)

    def nearest(self, key, count):
        original = self.bank.nearest(key, count)
        proposals = []
        dt = self.planner.arena.control_dt
        for e in original:
            t = e.time_to_contact
            # The original memory aligns tapes at this optical depth landmark.
            tick = int(np.clip(round(t / dt) - 1, 0, len(e.targets) - 1))
            displacement = np.array([key[0] - e.flight[0], 0.0, key[1] - e.flight[1]])
            distance = np.linalg.norm(displacement)
            if distance < 0.012:
                continue
            displacement *= self.sign * min(1.0, 0.20 / distance)
            q = e.targets[tick, :5]
            ik = self.planner.ik
            _, _, position, normal = ik.pose(q, jacobian=True)
            change = np.linalg.lstsq(
                np.vstack([position, 0.12 * normal, 0.025 * np.eye(5)]),
                np.r_[displacement, np.zeros(8)],
                rcond=None,
            )[0]
            change *= min(1.0, 0.70 / max(np.max(abs(change)), 1e-9))
            target = e.targets.copy()
            time = dt * (np.arange(len(target)) + 1)
            weight = np.minimum(
                np.clip(time / max(0.05, 0.65 * t), 0.0, 1.0),
                np.clip((t + 0.37 - time) / 0.18, 0.0, 1.0),
            )
            target[:, :5] = np.clip(
                target[:, :5] + weight[:, None] * change, ik.low, ik.high
            )
            proposals.append(memory.Example(e.flight, t, target))
        self.offered += len(proposals)
        return original + proposals


def trial(feeds, roster, checkpoint, output, sign):
    banks = []

    class AdaptedPlanner(memory.RecallPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memory = SpatialRecall(self.memory, self, sign)
            banks.append(self.memory)

    with patch.object(memory, "RecallPlanner", AdaptedPlanner):
        evaluate(feeds, roster, checkpoint, output)
    archive = _read(output)
    archive["spatial_recall_sign"] = sign
    for bank, row in zip(banks, archive["rows"], strict=True):
        row["spatial_proposals"] = bank.offered
    _write(output, archive)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("roster", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    args = p.parse_args()
    trial(args.feeds, args.roster, args.checkpoint, args.output, args.sign)
