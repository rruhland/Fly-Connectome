"""Associate actually issued joint references with visual reversals that earn a delayed score."""

import itertools
from collections import deque
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _write


def optical_state(tracker, now):
    tracker = getattr(tracker, "tracker", tracker)
    f = tracker.latest
    if f is None or now - f.time >= 0.105 or f.rms >= 2.4 or f.depth_sigma >= 0.16:
        return None
    return np.r_[*f.advance(now)]


class ScoredJoints:
    def __init__(self):
        self.states = []
        self.targets = []
        self.leads = []
        self.marks = []
        self.practice = []
        self.completed = 0
        self.begin_point()

    def begin_point(self):
        self.recent = deque(maxlen=110)
        self.last_credit = -1
        self.strikes = 0
        self.start = len(self.states)

    def step(self, tick, state, q, reference):
        self.recent.append((tick, state, q[:5].copy(), reference.copy()))

    def credit(self):
        rows = list(self.recent)
        visible = [
            r for r in rows if r[1] is not None and np.linalg.norm(r[1][3:5]) > 0.9
        ]
        turns = []
        for a, b in itertools.pairwise(visible):
            if b[0] - a[0] > 9 or a[0] <= self.last_credit:
                continue
            av, bv = a[1][3:5], b[1][3:5]
            if (
                av[1] < -0.9
                and bv[1] > 0.9
                and av @ bv < -0.48 * np.linalg.norm(av) * np.linalg.norm(bv)
            ):
                turns.append((a, b))
        if not turns:
            return
        before, after = turns[-1]
        impact = (before[0] + after[0]) * 0.5
        if not 5 <= rows[-1][0] - impact <= 73:
            return
        count = 0
        for i, (tick, opt, q, control) in enumerate(rows):
            age = (tick - impact) / 60.0
            if opt is None or not (-0.55 < age < -0.018 or 0.055 < age < 0.44):
                continue
            facing = before[1][3:5] if age < 0 else after[1][3:5]
            if facing @ opt[3:5] <= 0.28 * np.linalg.norm(facing) * np.linalg.norm(
                opt[3:5]
            ):
                continue
            self.states.append(np.r_[opt, q])
            self.targets.append(control)
            self.leads.append(rows[min(i + 2, len(rows) - 1)][3])
            self.marks.append((tick, age))
            count += 1
        if count:
            self.strikes += 1
            self.last_credit = after[0]

    def complete(self, card):
        if self.strikes:
            self.practice.append(
                {
                    "card": card,
                    "frames": len(self.states) - self.start,
                    "mentor_returns": self.strikes,
                    "source_row": self.completed,
                }
            )

        self.completed += 1

    def save(self, path, ready, depth, source):
        path = Path(path)
        tmp = path.with_suffix(".partial.npz")
        np.savez_compressed(
            tmp,
            states=np.array(self.states).reshape(-1, 11),
            targets_now=np.array(self.targets).reshape(-1, 6),
            targets_lead=np.array(self.leads).reshape(-1, 6),
            marks=np.array(self.marks).reshape(-1, 2),
        )
        tmp.replace(path)
        _write(
            path.with_suffix(".json"),
            {
                "ready": list(ready),
                "depth": depth,
                "sources": {source.name: _hash(source)},
                "practice": [dict(row, source=source.name) for row in self.practice],
                "legal": sum(r["mentor_returns"] for r in self.practice),
                "frames": len(self.states),
                "bank_sha256": _hash(path),
                "credit": "Scalar legal return and horizontal reversal measured by stereo. No simulator contact time or coordinates select frames.",
            },
        )
