"""A causal RGB/proprioceptive policy issuing fresh joint rates every frame."""

from collections import deque

import numpy as np
from scipy.ndimage import find_objects, label
from scipy.spatial import cKDTree


EYES = ('left', 'right')
WINDOW = 7


def orange_center(rgb, previous):
    """Measure the ordinary ball silhouette in an unmarked camera image."""
    r, g, b = rgb.transpose(2, 0, 1)
    pixel_scale = 480 / rgb.shape[1]
    if previous is not None and pixel_scale != 1:
        previous = (previous + .5) / pixel_scale - .5
    mask = (r > 90) & (g > .26*r) & (g < .64*r) & (b < .26*r)
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    # Most images have only a single orange object. Search local components
    # when highlights, furniture or a second orange object enlarge the mask.
    if x.max() - x.min() <= 20 and y.max() - y.min() <= 20:
        center = np.array((x.mean(), y.mean()))
        return center if pixel_scale == 1 else (center + .5)*pixel_scale - .5
    x0, y0 = x.min(), y.min()
    components, _ = label(mask[y0:y.max()+1, x0:x.max()+1])
    best = None
    for index, (ys, xs) in enumerate(find_objects(components), 1):
        if max(ys.stop - ys.start, xs.stop - xs.start) > 20:
            continue
        yy, xx = np.nonzero(components[ys, xs] == index)
        if len(xx) < 2:
            continue
        center = np.array((x0 + xs.start + xx.mean(), y0 + ys.start + yy.mean()))
        cost = -np.sqrt(len(xx)) if previous is None else np.linalg.norm(center - previous) - np.sqrt(len(xx))
        if best is None or cost < best[0]:
            best = cost, center
    if best is None:
        return None
    return best[1] if pixel_scale == 1 else (best[1] + .5)*pixel_scale - .5


class ViewHistory:
    """Encode current apparent position and recent motion without a flight model."""

    def __init__(self):
        self.views = [deque(maxlen=WINDOW) for _ in EYES]
        self.latest = [None] * len(EYES)
        self.joints = None
        self.last_rate = None

    def observe(self, frames, joints, issued_position):
        image = [orange_center(frames[eye], self.latest[i]) for i, eye in enumerate(EYES)]
        return self.observe_pixels(image, joints, issued_position)

    def observe_pixels(self, image, joints, issued_position):
        visual = []
        for eye, (position, history) in enumerate(zip(image, self.views, strict=True)):
            if position is not None:
                self.latest[eye] = position
            history.append(position)
            reads = [(i, p) for i, p in enumerate(history) if p is not None]
            last = np.array((240., 180.)) if not reads else reads[-1][1]
            motion = np.zeros(2)
            older = np.zeros(2)
            age = WINDOW if not reads else len(history) - reads[-1][0] - 1
            if len(reads) >= 2:
                a, first = reads[-2]
                b, second = reads[-1]
                motion = (second - first) / (b - a)
                c, earlier = reads[max(0, len(reads) - 5)]
                older = (second - earlier) / (b - c)
            visual.extend(np.r_[last + min(age, 2)*motion, motion, older, age])
        movement = np.zeros_like(joints) if self.joints is None else joints - self.joints
        self.joints = joints
        previous = np.zeros_like(joints) if self.last_rate is None else self.last_rate
        return np.r_[visual, issued_position, joints, movement, previous]

    def issued(self, rates):
        self.last_rate = rates


class JointPolicy:
    """Local least squares from demonstrations of sensor/command transitions.

    Each prediction uses a neighborhood of individual observations. The saved
    fit has no episode order, stroke identifier, action chunk or time index.
    """

    def __init__(self, state, postures, rates, step_size, *, neighbors=20, body=.22, mix=.0):
        self.step_size = np.asarray(step_size)
        self.mix = mix
        self.neighbors = min(neighbors, len(state))
        if not self.neighbors:
            return
        self.scale = np.r_[np.tile([14., 14., 7., 7., 5., 5., 4.], 2),
                           np.tile([.30, .45, .35, .35, .26, .60], 2)/max(body, 1e-6),
                           np.full(6, .075)/max(body, 1e-6), np.full(6, .55)/max(body, 1e-6)]
        self.state = state / self.scale
        self.tree = cKDTree(self.state)
        self.values = np.column_stack((postures, rates))

    def command(self, state, issued_position):
        if not self.neighbors:
            return np.zeros_like(issued_position)
        query = state / self.scale
        distance, idx = self.tree.query(query, k=self.neighbors)
        distance, idx = np.atleast_1d(distance), np.atleast_1d(idx)
        weights = np.exp(-.5 * ((distance - distance[0])/max(.11, .48*distance[-1]))**2)
        delta = self.state[idx] - query
        # Weighted local linear correction supplies feedback inside each visual region.
        weight_root = np.sqrt(weights)
        design = np.column_stack((np.ones(len(idx)), delta))
        weighted = design * weight_root[:, None]
        ridge = np.eye(design.shape[1]) * .22
        ridge[0, 0] = 0.
        coefficients = np.linalg.solve(weighted.T @ weighted + ridge, weighted.T @ (self.values[idx] * weight_root[:, None]))
        posture, rate = np.split(coefficients[0], 2)
        servo = (posture-issued_position)/self.step_size
        return np.clip((1-self.mix)*servo + self.mix*rate, -1., 1.)
