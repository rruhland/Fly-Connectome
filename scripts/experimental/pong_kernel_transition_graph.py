"""Anonymous visual kernel transition model; convex backups, no gradients.

This module never reads or runs Pong. Start identifiers establish independent
support only; they are not query inputs. Unknown boundaries are model boundaries,
not natural terminal observations.
"""
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = "anonymous-kernel-transition-graph-v1"
CAPACITY = 4
NEIGHBORS = 8
NORMAL_PERIOD_SECONDS = .02
HALF_LIFE_SECONDS = 160 * NORMAL_PERIOD_SECONDS


def scheduled_ticks(ordinal):
    ordinal = np.asarray(ordinal, dtype=np.int64)
    return (ordinal + 1) * 12 // 5 - ordinal * 12 // 5


def reconstruct_timing(rows, physics_seconds, dt=1 / 120):
    """Verify report total and derive final duration without clipping it."""
    if rows < 1 or not np.isfinite(physics_seconds) or physics_seconds <= 0:
        raise ValueError("nonempty trace and positive reported elapsed time required")
    total_ticks = round(physics_seconds / dt)
    if abs(total_ticks * dt - physics_seconds) > 1e-9:
        raise ValueError("reported duration is not an integral number of sensor ticks")
    ticks = scheduled_ticks(np.arange(rows))
    final_ticks = total_ticks - int(ticks[:-1].sum())
    if not 1 <= final_ticks <= ticks[-1]:
        raise ValueError("report cannot match full interior camera intervals")
    ticks[-1] = final_ticks
    if abs(float(ticks.sum()) * dt - physics_seconds) > 1e-9:
        raise AssertionError("timing reconstruction failed")
    return ticks * dt, ticks


def elapsed_discount(elapsed_seconds):
    elapsed = np.asarray(elapsed_seconds, dtype=np.float64)
    if not np.isfinite(elapsed).all() or np.any(elapsed <= 0):
        raise ValueError("positive finite elapsed durations required")
    return np.exp2(-elapsed / HALF_LIFE_SECONDS)


class SceneMetric:
    def __init__(self, scales, radius):
        self.scales = np.asarray(scales, dtype=np.float64).copy()
        self.radius = float(radius)
        if (self.scales.shape != (4,) or not np.isfinite(self.scales).all()
                or np.any(self.scales <= 0) or not np.isfinite(radius) or radius <= 0):
            raise ValueError("four positive fixed scales and positive radius required")

    def encode(self, scene, phase, past_hit):
        rows = np.asarray(scene, dtype=np.float64).reshape(-1, 6)
        if (len(rows) > CAPACITY or not np.isfinite(rows).all()
                or not np.isin(rows[:, 4:], [0, 1]).all()):
            return np.zeros(CAPACITY * 4), -1
        if not 0 <= int(phase) < 5:
            raise ValueError("public camera phase must be in [0,5)")
        # Visual positions are (y,x). Last lexsort key is primary.
        rows = rows[np.lexsort(rows[:, [5, 4, 3, 2, 0, 1]].T)]
        vector = np.zeros((CAPACITY, 4))
        vector[:len(rows)] = rows[:, :4] / self.scales
        mask = len(rows)
        for i, row in enumerate(rows):
            mask |= int(row[4]) << (3 + i)
            mask |= int(row[5]) << (7 + i)
        key = ((mask * 2 + int(bool(past_hit))) * 5 + int(phase))
        return vector.ravel() / np.sqrt(CAPACITY * 4), key


@dataclass
class Neighbors:
    indices: np.ndarray
    weights: np.ndarray
    supported: np.ndarray

    def q_values(self, values):
        return np.einsum("nak,nak->na", values[self.indices], self.weights)


class TransitionIndex:
    def __init__(self, vectors, keys, actions, start_ids, radius):
        self.vectors = np.asarray(vectors, dtype=np.float64)
        self.keys = np.asarray(keys, dtype=np.int64)
        self.actions = np.asarray(actions, dtype=np.int8)
        self.start_ids = np.asarray(start_ids, dtype=np.int64)
        self.radius = float(radius)
        n = len(self.vectors)
        if (self.vectors.shape != (n, 16) or not np.isfinite(self.vectors).all()
                or self.keys.shape != (n,) or self.actions.shape != (n,)
                or self.start_ids.shape != (n,) or not np.isin(self.actions, [0, 1, 2]).all()
                or np.any(self.keys < -1) or self.radius <= 0):
            raise ValueError("consistent anonymous source states and actions required")
        self.groups = {}
        for key in np.unique(self.keys):
            if key < 0:
                continue  # Retained overflow transitions provide no local support.
            for action in range(3):
                indices = np.flatnonzero((self.keys == key) & (self.actions == action))
                if len(indices):
                    self.groups[(int(key), action)] = indices, cKDTree(self.vectors[indices])

    def query(self, vectors, keys, deadline=None):
        vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 16)
        keys = np.asarray(keys, dtype=np.int64)
        n = len(vectors)
        if keys.shape != (n,) or not np.isfinite(vectors).all():
            raise ValueError("finite query vectors and matching keys required")
        indices = np.zeros((n, 3, NEIGHBORS), dtype=np.int32)
        weights = np.zeros((n, 3, NEIGHBORS), dtype=np.float64)
        supported = np.zeros((n, 3), dtype=bool)
        for key in np.unique(keys):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("neighbor construction exceeded offline fit allocation")
            positions = np.flatnonzero(keys == key)
            for action in range(3):
                group = self.groups.get((int(key), action))
                if group is None:
                    continue
                candidates, tree = group
                distances, local = tree.query(vectors[positions], k=NEIGHBORS,
                    distance_upper_bound=np.nextafter(self.radius, np.inf), workers=1)
                valid = np.isfinite(distances)
                safe_local = np.where(valid, local, 0)
                selected = candidates[safe_local]
                ids = self.start_ids[selected]
                minimum = np.min(np.where(valid, ids, np.iinfo(np.int64).max), axis=1)
                maximum = np.max(np.where(valid, ids, -1), axis=1)
                eligible = (valid.sum(axis=1) >= 2) & (minimum != maximum)
                kernel = np.exp(-.5 * (np.where(valid, distances, 0) / self.radius) ** 2)
                kernel *= valid & eligible[:, None]
                mass = kernel.sum(axis=1)
                normalized = np.divide(kernel, mass[:, None], out=np.zeros_like(kernel),
                    where=mass[:, None] > 0)
                indices[positions, action] = selected
                weights[positions, action] = normalized
                supported[positions, action] = eligible
        assert np.allclose(weights.sum(axis=2), supported.astype(float), atol=1e-14)
        return Neighbors(indices, weights, supported)


def backup(values, successor_neighbors, hit, terminal, discounts):
    continuation = successor_neighbors.q_values(values).max(axis=1)
    # Unsupported successors have exactly zero interpolation mass.
    return np.where(hit, 1., np.where(terminal, 0., discounts * continuation))


def solve(successor_neighbors, hit, terminal, discounts, deadline,
          tolerance=1e-8, max_iterations=4096):
    hit = np.asarray(hit, dtype=bool)
    terminal = np.asarray(terminal, dtype=bool)
    discounts = np.asarray(discounts, dtype=np.float64)
    if (hit.shape != terminal.shape or discounts.shape != hit.shape
            or not np.isfinite(discounts).all() or np.any(discounts <= 0)
            or np.any(discounts >= 1)):
        raise ValueError("finite strictly discounted transition targets required")
    values = hit.astype(np.float64)
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        if time.monotonic() >= deadline:
            return values, dict(converged=False, iterations=iteration - 1,
                residual=None if not np.isfinite(residual) else residual, stop="deadline")
        updated = backup(values, successor_neighbors, hit, terminal, discounts)
        residual = float(np.max(np.abs(updated - values)))
        assert np.isfinite(updated).all() and np.all((updated >= 0) & (updated <= 1))
        values = updated
        if residual <= tolerance:
            return values, dict(converged=True, iterations=iteration, residual=residual,
                value_error_bound=residual / (1 - float(discounts.max())), stop="tolerance")
    return values, dict(converged=False, iterations=max_iterations, residual=residual,
        stop="iteration_limit")


class KernelGraph:
    def __init__(self, metric, index, values):
        self.metric, self.index = metric, index
        self.values = np.asarray(values, dtype=np.float64).copy()
        if (self.values.shape != (len(index.vectors),) or not np.isfinite(self.values).all()
                or np.any(self.values < 0) or np.any(self.values > 1)):
            raise ValueError("one probability-valued target per transition required")

    def inspect(self, scene, phase, past_hit, baseline_action):
        vector, key = self.metric.encode(scene, phase, past_hit)
        neighbors = self.index.query(vector[None], np.array([key]))
        q = neighbors.q_values(self.values)[0]
        support = neighbors.supported[0]
        action = int(baseline_action)
        if past_hit and key >= 0 and support[action]:
            best = int(np.argmax(np.where(support, q, -np.inf)))
            if q[best] > q[action]:
                action = best
        return dict(action=action, baseline_action=int(baseline_action),
            q=q.tolist(), supported=support.tolist(), key=int(key))

    def save(self, path):
        np.savez_compressed(Path(path), kind=SCHEMA, scales=self.metric.scales,
            radius=self.metric.radius, vectors=self.index.vectors, keys=self.index.keys,
            actions=self.index.actions, start_ids=self.index.start_ids, values=self.values)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["kind"]) != SCHEMA:
                raise ValueError("unknown graph schema")
            metric = SceneMetric(data["scales"], float(data["radius"]))
            index = TransitionIndex(data["vectors"], data["keys"], data["actions"],
                data["start_ids"], metric.radius)
            return cls(metric, index, data["values"])
