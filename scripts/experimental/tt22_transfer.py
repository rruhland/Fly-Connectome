"""Learn a role-free optical continuation prior from native image tracklets."""

from collections import deque

import numpy as np
from scipy.ndimage import center_of_mass, label
from scipy.optimize import linear_sum_assignment


WINDOW = 4
VELOCITY = np.array((0., 0., 1.))


def components(image, background):
    """Find all compact changes from the first image, with no color or role labels."""
    changed = abs(image - background) >= .035
    groups, count = label(changed)
    sizes = np.bincount(groups.ravel(), minlength=count + 1)
    eligible = np.flatnonzero((sizes >= 1) & (sizes <= .045 * image.size))
    eligible = eligible[eligible != 0]
    if not len(eligible):
        return np.empty((0, 2))
    # Keep computationally bounded matching when a scene changes globally.
    ids = eligible[np.argsort(sizes[eligible])[-32:]]
    yx = center_of_mass(np.ones_like(image), groups, ids)
    return np.asarray(yx)[:, ::-1]


class Tracklets:
    """Link consecutive image components by bounded generic motion."""

    def __init__(self):
        self.background = None
        self.tracks = []

    def push(self, image):
        if self.background is None:
            self.background = image.copy()
            return []
        points = components(image, self.background)
        max_step = .088 * min(image.shape)
        paths = [[pos] for pos in points]
        windows = []
        if len(points) and self.tracks:
            predictions = np.array([p[-1] if len(p) == 1 else 2*p[-1] - p[-2] for p in self.tracks])
            prior = np.array([p[-1] for p in self.tracks])
            costs = np.linalg.norm(predictions[:, None] - points, axis=2)
            costs += .25 * np.linalg.norm(prior[:, None] - points, axis=2)
            old, new = linear_sum_assignment(costs)
            for i, j in zip(old, new, strict=True):
                if costs[i, j] > max_step:
                    continue
                path = self.tracks[i] + [points[j]]
                if len(path) == WINDOW + 1:
                    movement = np.linalg.norm(np.diff(path, axis=0), axis=1)
                    if .30 <= np.median(movement) and max(movement) <= max_step:
                        windows.append(path)
                    path = path[1:]
                paths[j] = path
        self.tracks = paths
        return windows


def patches(windows):
    movement = np.diff(np.asarray(windows, dtype=float), axis=1)
    scale = np.maximum(.4, np.sqrt(np.mean(movement[:, :-1]**2, axis=(1, 2))))
    motion = movement / scale[:, None, None]
    return motion[:, :-1].transpose(0, 2, 1).reshape((-1, WINDOW - 1)), motion[:, -1].ravel()


class MotionWeights:
    """Fit the same three time weights for either coordinate in either game."""

    def __init__(self, initial=VELOCITY, strength=28.):
        self.prior = np.asarray(initial).copy()
        self.gram = np.eye(WINDOW - 1) * strength
        self.cross = self.gram @ self.prior
        self.samples = 0

    @property
    def weights(self):
        return np.linalg.solve(self.gram, self.cross)

    def update(self, x, y):
        self.gram += x.T @ x
        self.cross += x.T @ y
        self.samples += len(x)

    def error(self, x, y):
        return float(np.sqrt(np.mean((x @ self.weights - y)**2)))

    def state(self):
        return dict(prior=self.prior.tolist(), weights=self.weights.tolist(), gram=self.gram.tolist(),
                    cross=self.cross.tolist(), samples=self.samples)

    @classmethod
    def restore(cls, state):
        model = cls(state['prior'])
        model.gram = np.asarray(state['gram'])
        model.cross = np.asarray(state['cross'])
        model.samples = state['samples']
        return model


class ViewMotion:
    """Apply a learned pixel forecast without any world position or object template."""

    def __init__(self, weights):
        self.weights = np.asarray(weights)
        self.path = deque(maxlen=WINDOW)

    def step(self, tick, center):
        prior = None
        if len(self.path) == WINDOW and self.path[-1][0] == tick - 1:
            trail = np.array([point for _, point in self.path])
            moves = np.diff(trail, axis=0)
            if max(np.linalg.norm(moves, axis=1)) <= 24.:
                prior = trail[-1] + self.weights @ moves
        if center is None:
            return prior, prior is not None
        point = np.asarray(center, dtype=float)
        if self.path and self.path[-1][0] != tick - 1:
            self.path.clear()
        self.path.append((tick, point))
        if prior is None or np.linalg.norm(point-prior) > 6.:
            return point, False
        return .30*prior + .70*point, False
