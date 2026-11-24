"""Record physical candidate frames as training-only direct joint examples."""

import numpy as np


class NativeFrames:
    def __init__(self, path):
        self.path = path
        self.positive = []
        self.negative = []
        self.group = None

    def add(self, tracks, outcomes, feedback, eta, dt, round_number):
        valid = np.flatnonzero(np.isfinite(outcomes[:, 0]))
        # Keep distinct motor solutions without counting every nearly identical
        # candidate at a single ball as an independent point of experience.
        best = valid[np.argsort(outcomes[valid, 0])][:8]
        failed = np.flatnonzero(~np.isfinite(outcomes[:, 0]) & (feedback[:, 1] > 0))
        weak = failed[:2]
        for indices, collection, legal in (
            (best, self.positive, True),
            (weak, self.negative, False),
        ):
            for i in indices:
                track = tracks[i]
                end = outcomes[i, 1] if legal else eta
                ticks = np.arange(len(track) - 2)
                use = (
                    (ticks * dt < end)
                    & (track[ticks, 4] < -0.9)
                    & np.isfinite(track[ticks + 2, -1])
                )
                ticks = ticks[use]
                if len(ticks):
                    marks = np.c_[self.group[-1] + ticks, ticks * dt - end]
                    metadata = np.tile([*self.group, round_number, i], (len(ticks), 1))
                    collection.append(
                        np.c_[
                            track[ticks, :17], track[ticks + 2, 11:17], marks, metadata
                        ].astype(np.float64)
                    )

    def save(self):
        arrays = {}
        for prefix, records in (("", self.positive), ("weak_", self.negative)):
            block = np.concatenate(records) if records else np.empty((0, 31))
            for field, cols in [
                ("states", slice(0, 11)),
                ("targets_now", slice(11, 17)),
                ("targets_lead", slice(17, 23)),
                ("marks", slice(23, 25)),
                ("group", slice(25, None)),
            ]:
                # Seed metadata must retain all integer digits.
                a = block[:, cols]
                arrays[prefix + field] = a.astype(
                    np.int64 if field == "group" else np.float32
                )
        tmp = self.path.with_suffix(".partial.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(self.path)
