"""Refit a five-axis motor response at short boundaries of a single control stream."""
from collections import deque
from pathlib import Path

import numpy as np

from .tt24_motor_bank import I
from .tt24_pong_arm_local import ALPHAS, fit, windows


class RollingMotorLearner:
    """Keep bounded proprioceptive history. Call observe after each actual motor request."""
    def __init__(self, capacity=12, block_ticks=480):
        if capacity < 4 or block_ticks < 16:
            raise ValueError('need four or more blocks and at least sixteen ticks per block')
        self.capacity = capacity
        self.block_ticks = block_ticks
        self.blocks = deque(maxlen=capacity)
        self.current = None
        self.count = 0
        self.completed = 0
        self.map = I.copy()

    def begin(self, joints):
        if self.current is not None:
            self._close_block()
        self.current = np.empty((self.block_ticks, 10), dtype=np.float64)
        self.start = np.asarray(joints, dtype=np.float64).copy()
        self.count = 0

    def observe(self, requested, measured):
        if self.current is None:
            raise RuntimeError('start the motor stream with its first encoder reading')
        if self.count == self.block_ticks:
            last = self.current[-1, 5:].copy()
            self.begin(last)
        self.current[self.count, :5] = requested
        self.current[self.count, 5:] = measured
        self.count += 1

    def _close_block(self):
        if self.current is not None and self.count >= 16:
            self.blocks.append((self.start, self.current[:self.count].copy()))
            self.completed += 1
        self.current = None
        self.count = 0

    def end(self):
        """Fit at a caller-chosen break. The per-tick method only collects measurements."""
        self._close_block()
        if len(self.blocks) < 4:
            return False
        clips = [dict(commands=data[:, :5], encoders=np.vstack([first, data[:, 5:]]))
                 for first, data in self.blocks]
        # Each fold must contain useful movement for each output. Identity is
        # retained until there is evidence for fitting, including on a parked arm.
        for alpha in ALPHAS:
            groups = windows(clips, alpha)
            for fold in (groups[::2], groups[1::2]):
                active = np.concatenate([group[2] for group in fold])
                if np.min(active.sum(axis=0)) < 5:
                    return False
        fitted = fit(clips)
        self.map = np.asarray([r['matrix_row'] for r in fitted])
        return True

    def save(self, path):
        spans = [len(points) for _, points in self.blocks]
        flat = np.concatenate([points for _, points in self.blocks]) if spans else np.empty((0, 10))
        archive = dict(schema='tt24-rolling-motor-1', capacity=self.capacity, block_ticks=self.block_ticks,
                       completed=self.completed, spans=spans, starts=np.array([p for p, _ in self.blocks]),
                       points=flat, fitted_map=self.map, active=self.current is not None)
        if self.current is not None:
            archive.update(start=self.start, current=self.current[:self.count])
        np.savez_compressed(Path(path), **archive)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if archive['schema'].item() != 'tt24-rolling-motor-1':
                raise ValueError('unknown motor checkpoint')
            obj = cls(int(archive['capacity']), int(archive['block_ticks']))
            lengths = archive['spans'].astype(int)
            if len(lengths) > obj.capacity or len(archive['points']) != sum(lengths):
                raise ValueError('inconsistent motor history')
            offset = 0
            points = archive['points']
            for start, n in zip(archive['starts'], lengths, strict=True):
                obj.blocks.append((start.copy(), points[offset:offset+n].copy()))
                offset += n
            obj.map = archive['fitted_map'].copy()
            obj.completed = int(archive['completed'])
            if bool(archive['active']):
                obj.begin(archive['start'])
                live = archive['current']
                obj.count = len(live)
                obj.current[:obj.count] = live
        return obj
