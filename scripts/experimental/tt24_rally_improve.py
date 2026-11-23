"""Learn stereo motion and relative joint feedback from single-frame teacher labels."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .tt4_camera import PerspectiveCamera
from .tt19_view_trial import CAMERAS
from .tt24_rally_policy import EYES


RIG = tuple(PerspectiveCamera(CAMERAS[key][1], CAMERAS[key][2], 480, 360, CAMERAS[key][3]) for key in EYES)
BODY_SCALE = np.r_[np.tile([.30, .45, .35, .35, .26, .60], 2), np.full(6, .075), np.full(6, .55)]


def triangulate(left, right):
    first, second = RIG
    a, b = first.ray(left), second.ray(right)
    cross = np.cross(a, b)
    gap = np.subtract(second.eye, first.eye)
    wedge = np.maximum(np.sum(cross**2, axis=-1), 1e-6)
    t = np.sum(np.cross(gap, b) * cross, axis=-1) / wedge
    u = np.sum(np.cross(gap, a) * cross, axis=-1) / wedge
    points = (np.add(first.eye, t[..., None] * a) + np.add(second.eye, u[..., None] * b)) / 2.
    return np.clip(points, [-1.0, -2.2, -.7], [1.0, 2.3, 1.3])


def stereo_motion(state):
    """Register two measured views and compare current and past apparent positions."""
    left, right = state[..., :7], state[..., 7:14]
    present = triangulate(left[..., :2], right[..., :2])
    past = triangulate(left[..., :2] - left[..., 2:4], right[..., :2] - right[..., 2:4])
    older = triangulate(left[..., :2] - 3*left[..., 4:6], right[..., :2] - 3*right[..., 4:6])
    fast_velocity = np.clip(present-past, -.22, .22)
    slow_velocity = np.clip((present-older)/3, -.18, .18)
    quality = state[..., [6, 13]]
    return np.concatenate((present, fast_velocity, slow_velocity, quality), axis=-1)


RETINA_SCALE = np.tile([14., 14., 7., 7., 5., 5., 4.], 2)
OPTIC_SCALE = np.array([.10, .23, .075, .039, .061, .036, .033, .054, .032, 3., 3.])
POSE_SCALE = np.array([.22, .22, .25, .28, .26, .65])


class MotionPolicy:
    """Fit bounded local motor feedback with both position and rate labels."""

    def __init__(self, state, postures, command, step, *, mode):
        self.mode = mode
        self.stereo = mode in ('stereo', 'relative', 'relative_contrast', 'moving')
        self.relative = dict(stereo=0., relative=.67, retina_rate=.15, relative_contrast=.52, moving=.38)[mode]
        self.rate_mix = dict(stereo=.03, relative=.18, retina_rate=.72, relative_contrast=.23, moving=.38)[mode]
        self.neighbors = dict(stereo=44, relative=44, retina_rate=36, relative_contrast=34, moving=46)[mode]
        self.step = np.asarray(step)
        self.scales = np.r_[OPTIC_SCALE if self.stereo else RETINA_SCALE, BODY_SCALE/.68]
        self.state = self.encode(state) / self.scales
        self.tree = cKDTree(self.state)
        self.postures = postures
        self.values = np.column_stack((postures, postures-state[:, 20:26], command))
        self.rate_energy = np.linalg.norm(command[:, :5], axis=1)
        self.ridge = np.eye(self.state.shape[1] + 1) * .24
        self.ridge[0, 0] = 0.

    def encode(self, state):
        return np.concatenate((stereo_motion(state), state[..., 14:]), axis=-1) if self.stereo else state

    def command(self, state, issued_position):
        query = self.encode(state) / self.scales
        distance, ids = self.tree.query(query, k=self.neighbors)
        width = max(.13, .44*distance[-1])
        weights = np.exp(-.5*((distance-distance[0])/width)**2)
        if self.mode == 'relative_contrast':
            # Favor locally consistent demonstrated joint efforts in ambiguous regions.
            spread = (self.postures[ids] - self.postures[ids[0]]) / POSE_SCALE
            weights *= np.exp(-.25*np.sum(spread**2, axis=1))
        if self.mode == 'moving':
            weights *= .28 + np.minimum(self.rate_energy[ids], 1.1)
        delta = self.state[ids] - query
        root = np.sqrt(weights)
        design = np.column_stack((np.ones(len(ids)), delta)) * root[:, None]
        outputs = np.linalg.solve(design.T @ design + self.ridge, design.T @ (self.values[ids]*root[:, None]))[0]
        absolute, relative, rate = np.split(outputs, 3)
        correction = (1-self.relative)*absolute + self.relative*(state[20:26]+relative)
        servo = (correction - issued_position)/self.step
        return np.clip((1-self.rate_mix)*servo + self.rate_mix*rate, -1., 1.)
