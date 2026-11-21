"""Pool anonymous image-track relationships and learn from public point scores."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit

from .tt3_pixels import regions


FEATURE_HORIZON = 12
PAIR_SIZE = 16
TRACK_SIZE = 6


def _scale(value, width):
    return np.tanh(value / width)


def _pool(values, columns):
    if not len(values):
        return np.zeros(3 * columns)
    return np.stack((values.mean(axis=0), values.min(axis=0), values.max(axis=0))).ravel()


@dataclass
class _Track:
    appearance: int
    radius: float
    history: deque

    @property
    def position(self):
        return self.history[-1][1]

    def velocity(self):
        first, last = self.history[0], self.history[-1]
        return (last[1] - first[1]) / max(1, last[0] - first[0])


class Tracks:
    """Quantize appearance locally, then match the same generic regions in both games."""

    def __init__(self, library):
        self.library = library
        self.tracks = []
        self.images = deque(maxlen=4)
        self.tick = -1

    def update(self, rgb):
        self.tick += 1
        height, width = rgb.shape[:2]
        scale = max(width, height)
        quantized = np.ascontiguousarray(rgb & 0xe0)
        records = regions(quantized, self.library)
        pixel = records[:, :2] / 2.
        appearance = records[:, 2]
        size = np.sqrt(records[:, 3]) / scale
        position = (pixel - (width / 2., height / 2.)) / scale
        if len(records) > 26:
            # Preserve small moving regions and a few persistent scene references.
            idx = np.minimum(np.rint(pixel).astype(int), (width - 1, height - 1))
            if self.images:
                current = quantized[idx[:, 1], idx[:, 0]]
                past = self.images[0][idx[:, 1], idx[:, 0]]
                change = np.count_nonzero(current != past, axis=1)
            else:
                change = np.zeros(len(idx))
            if self.tracks:
                nearby = np.linalg.norm(position[:, None, :] -
                                        np.stack([track.position for track in self.tracks]), axis=-1)
                kept = np.any((nearby < .045) & (appearance[:, None] ==
                                                [track.appearance for track in self.tracks]), axis=1)
            else:
                kept = np.zeros(len(idx), dtype=bool)
            good = np.argsort(-(change + .75 * kept + .16 * _scale(size, .04)), kind='stable')[:20]
            anchors = np.argsort(-size, kind='stable')[:6]
            pick = np.unique(np.r_[good, anchors])
            position, appearance, size = position[pick], appearance[pick], size[pick]
        self.images.append(quantized)

        used = set()
        if self.tracks and len(position):
            old = np.stack([track.position for track in self.tracks])
            velocity = np.stack([track.velocity() for track in self.tracks])
            gap = np.array([self.tick - track.history[-1][0] for track in self.tracks])
            prediction = old + velocity * gap[:, None]
            distance = np.linalg.norm(prediction[:, None] - position[None], axis=-1)
            same = np.array([track.appearance for track in self.tracks])[:, None] == appearance
            widths = np.asarray([track.radius for track in self.tracks])[:, None]
            cost = distance + .060 * ~same + .010 * np.abs(np.log(size / widths))
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols, strict=True):
                if cost[row, col] > .055:
                    continue
                track = self.tracks[row]
                track.history.append((self.tick, position[col]))
                track.radius = float(.75 * track.radius + .25 * size[col])
                used.add(col)
        for index in range(len(position)):
            if index not in used:
                self.tracks.append(_Track(int(appearance[index]), float(size[index]),
                                          deque([(self.tick, position[index])], maxlen=6)))
        self.tracks = [track for track in self.tracks if self.tick - track.history[-1][0] <= 2]
        eligible = [track for track in self.tracks if track.history[-1][0] == self.tick and len(track.history) >= 2]
        if len(eligible) > 12:
            ordered = sorted(eligible, key=lambda track: np.linalg.norm(track.velocity()), reverse=True)
            anchors = sorted(eligible, key=lambda track: track.radius, reverse=True)
            eligible = ordered[:8]
            included = {id(track) for track in eligible}
            eligible.extend([track for track in anchors if id(track) not in included][:4])
        return eligible, np.array((width, height)) / (2. * scale)


def descriptors(tracks, bounds):
    if not tracks:
        return _pool([], TRACK_SIZE), _pool([], PAIR_SIZE)
    p = np.stack([track.position for track in tracks])
    v = np.stack([track.velocity() for track in tracks])
    size = np.array([track.radius for track in tracks])
    speed = np.linalg.norm(v, axis=1)
    age = np.array([len(track.history) for track in tracks])
    edge = np.min(bounds - np.abs(p), axis=1)
    ahead = np.min(bounds - np.abs(p + 4. * v), axis=1)
    unary = np.column_stack((_scale(speed, .020), _scale(size, .05), _scale(edge, .14),
                             _scale(ahead - edge, .07), _scale(age, 4.),
                             _scale(np.linalg.norm(p, axis=1), .25)))
    left, right = np.triu_indices(len(p), 1)
    delta = p[right] - p[left]
    drift = v[right] - v[left]
    d = np.linalg.norm(delta, axis=1)
    change = np.sum(delta * drift, axis=1) / np.maximum(d, 1e-6)
    relative = np.linalg.norm(drift, axis=1)
    later = np.linalg.norm(delta + 4. * drift, axis=1)
    earlier = np.linalg.norm(delta - 4. * drift, axis=1)
    closeness = np.exp(-d / .16)
    angular = np.sum(v[left] * v[right], axis=1) / np.maximum(speed[left] * speed[right], 1e-6)
    span = size[left] + size[right]
    quicker = np.maximum(speed[left], speed[right])
    slower = np.minimum(speed[left], speed[right])
    pair = np.column_stack((
        closeness, _scale(change, .024), _scale(relative, .024), _scale(later - d, .09),
        _scale(earlier - d, .09), angular, _scale(d - .5 * span, .16),
        _scale(quicker, .024), _scale(slower, .024),
        _scale(np.abs(size[left] - size[right]), .07), _scale(span, .09),
        closeness * _scale(change, .024), closeness * _scale(relative, .024),
        closeness * angular, _scale(later - earlier, .12),
        _scale(np.linalg.norm(p[left] + p[right], axis=1), .55),
    ))
    return _pool(unary, TRACK_SIZE), _pool(pair, PAIR_SIZE)


BASE = 3 * TRACK_SIZE * 2 + 3
PAIRS = 3 * PAIR_SIZE * 2


class Observation:
    """One causal frame sample with no object names, legal-return flags or scene roles."""

    def __init__(self, library):
        self.tracker = Tracks(library)
        self.unary = deque(maxlen=4)
        self.paired = deque(maxlen=4)
        self.actions = deque(maxlen=4)

    def update(self, rgb, command):
        tracks, bounds = self.tracker.update(rgb)
        unary, pair = descriptors(tracks, bounds)
        self.unary.append(unary)
        self.paired.append(pair)
        self.actions.append(float(np.max(np.abs(command))) if command is not None else 0.)
        action = (self.actions[-1], np.mean(self.actions),
                  np.mean(np.abs(np.diff(self.actions))) if len(self.actions) > 1 else 0.)
        return np.r_[unary, np.mean(self.unary, axis=0), action, pair, np.mean(self.paired, axis=0)]


def targets(episode, *, terminal_image=True):
    features, reward = episode
    index = np.arange(6, len(features) - 1, 3)
    remaining = len(features) - index - int(terminal_image)
    end = (remaining <= FEATURE_HORIZON) & (reward != 0)
    return features[index], end.astype(float), np.full(len(index), float(reward > 0)), index


def learn(x, y, *, prior=None, penalty=.035):
    """Regularized logistic update with equal weight per observed prediction."""
    initial = np.zeros(x.shape[1] + 1)
    target = np.zeros_like(initial) if prior is None else np.r_[0., prior]
    initial[0] = np.log((.5 + y.sum()) / (.5 + len(y) - y.sum()))
    design = np.column_stack((np.ones(len(x)), x))
    regularization = np.full(len(target), penalty)
    regularization[0] = .003

    def loss(weights):
        logits = design @ weights
        offset = weights - target
        cost = np.mean(np.logaddexp(0., logits) - y * logits) + .5 * np.sum(regularization * offset**2)
        gradient = design.T @ (expit(logits) - y) / len(y) + regularization * offset
        return cost, gradient

    solution = minimize(loss, initial, method='L-BFGS-B', jac=True, options=dict(maxiter=180))
    return solution.x


def predict(weights, x):
    return expit(weights[0] + x @ weights[1:])
