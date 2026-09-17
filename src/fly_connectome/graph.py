"""Immutable, canonical directed contact graph (offline preprocessing)."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


def _integers(values, name):
    raw = np.asarray(values)
    if raw.size and (not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError(f"{name} must contain integers")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return raw.astype(np.int64, copy=True)


def _immutable(array):
    # A bytes-backed array cannot be made writable by a caller.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class Graph:
    body_ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    contacts: np.ndarray
    signs: np.ndarray
    gain: float

    def __post_init__(self):
        for name in ("body_ids", "pre", "post", "contacts", "signs"):
            object.__setattr__(self, name, _immutable(_integers(getattr(self, name), name)))
        n = len(self.body_ids)
        if n == 0 or np.any(self.body_ids <= 0) or np.any(np.diff(self.body_ids) <= 0):
            raise ValueError("body IDs must be positive, unique and sorted")
        if self.signs.shape != (n,) or not np.isin(self.signs, [-1, 1]).all():
            raise ValueError("every neuron requires an explicitly resolved fixed sign")
        if not (len(self.pre) == len(self.post) == len(self.contacts)):
            raise ValueError("edge arrays must have equal lengths")
        if np.any(self.pre < 0) or np.any(self.pre >= n) or np.any(self.post < 0) or np.any(self.post >= n):
            raise ValueError("edge endpoint outside roster")
        if np.any(self.contacts <= 0) or not np.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("contacts and gain must be positive and finite")
        keys = self.pre * n + self.post
        if np.any(np.diff(keys) <= 0):
            raise ValueError("edges must be unique and sorted by ordered pair")

    @classmethod
    def from_contacts(cls, body_ids, pre, post, counts, signs, gain):
        ids = _integers(body_ids, "body_ids")
        source, target, counts = (_integers(x, k) for x, k in
                                  ((pre, "pre"), (post, "post"), (counts, "counts")))
        if not (len(source) == len(target) == len(counts)) or np.any(counts <= 0):
            raise ValueError("contact rows must have equal lengths and positive counts")
        if not np.isin(source, ids).all() or not np.isin(target, ids).all():
            raise ValueError("contact endpoint outside roster")
        if np.any(np.diff(ids) <= 0):
            raise ValueError("body IDs must be unique and sorted")
        pairs, inverse = np.unique(np.column_stack((source, target)), axis=0, return_inverse=True)
        totals = np.zeros(len(pairs), dtype=np.int64)
        np.add.at(totals, inverse, counts)
        return cls(ids, np.searchsorted(ids, pairs[:, 0]),
                   np.searchsorted(ids, pairs[:, 1]), totals, signs, gain)

    def threshold(self, minimum):
        if minimum not in (1, 3, 5):
            raise ValueError("milestone thresholds are 1, 3 and 5")
        keep = self.contacts >= minimum
        return Graph(self.body_ids, self.pre[keep], self.post[keep],
                     self.contacts[keep], self.signs, self.gain)

    def initial_weights(self):
        return self.gain * self.contacts * self.signs[self.pre]

    def identity(self):
        payload = {name: getattr(self, name).tolist() for name in
                   ("body_ids", "pre", "post", "contacts", "signs")}
        payload["gain"] = self.gain
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
