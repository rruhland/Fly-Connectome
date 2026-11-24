"""Use each subsequent stereo frame as a training target for an earlier motion estimate."""

from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt37_visual_inertia import DT, SCALE, VisualInertia, feature, future_velocity


class BlankInertia(VisualInertia):
    """Matched camera wrapper with all learned velocity updates disabled."""

    def __init__(self):
        pass

    def update(self, now, point, state):
        return state


class StreamingInertia(VisualInertia):
    def __init__(self, checkpoint=None, period=DT):
        self.period = period
        self.features = deque(maxlen=8192)
        self.updates = deque(maxlen=8192)
        self.tree = None
        self.y = None
        self.gated = True
        self.windows = deque(maxlen=7)
        self.samples = deque(maxlen=4)
        self.last_time = None
        self.added = 0
        if checkpoint:
            with np.load(checkpoint) as z:
                if not np.isclose(float(z["period"]) if "period" in z else DT, period):
                    raise ValueError("Checkpoint and live camera cadence differ")
                self.features.extend(z["features"])
                self.updates.extend(z["updates"])
            self.rebuild()

    def rebuild(self):
        if len(self.features) >= 12:
            self.tree = cKDTree(np.asarray(self.features) / SCALE)
            self.y = np.asarray(self.updates)

    def update(self, now, point, state):
        if self.last_time is None or not np.isclose(
            now - self.last_time, self.period, atol=0.001
        ):
            self.windows.clear()
        self.windows.append((point, state))
        # At frame t the most recent eligible label is for t-3. All four label positions are now observed.
        if len(self.windows) == 7 and all(
            row[0] is not None and np.isfinite(row[0]).all() for row in self.windows
        ):
            past = np.stack([r[0] for r in self.windows])
            old = self.windows[3][1]
            if old is not None and abs(old[4]) > 0.9:
                v = future_velocity(past[3:], self.period)
                if v is not None and np.linalg.norm(v - old[3:6]) < 9:
                    self.features.append(feature(past[:4], old, self.period))
                    self.updates.append(v - old[3:])
                    self.added += 1
                    if self.added % 16 == 0:
                        self.rebuild()
        if self.tree is None:
            self.last_time = now
            return state
        return super().update(now, point, state)

    def save(self, path):
        tmp = Path(path).with_suffix(".partial.npz")
        np.savez_compressed(
            tmp,
            features=np.asarray(self.features).reshape(-1, len(SCALE)),
            updates=np.asarray(self.updates).reshape(-1, 3),
            gated=True,
            period=self.period,
        )
        tmp.replace(path)
