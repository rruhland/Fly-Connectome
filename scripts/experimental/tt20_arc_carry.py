"""Transfer an incoming-serve arc prior between one-camera RGB observers."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .tt4_arena import BALL_RADIUS
from .tt19_flight import FlightLaw


ANCHOR = 12
SOURCE_FRAMES = range(7, 16)
TARGET_FRAMES = range(12, 20)
# Velocity components use metres per control frame. Floors cover residual
# aerodynamic and camera-fit error in a small collection of practice arcs.
NOISE_FLOOR = np.array((.019, .034, .015, .003, .005, .003))
GRAVITY = FlightLaw().gravity
DRAG = FlightLaw().drag * 2.8 / 30.


def _terms(time):
    decay = np.exp(-DRAG * time)
    move = -np.expm1(-DRAG * time) / DRAG
    drop = (time - move) * GRAVITY / DRAG
    return decay, move, drop


def _position_basis(ticks):
    time = np.asarray(ticks) - ANCHOR
    _, move, drop = _terms(time)
    eye = np.broadcast_to(np.eye(3), (*time.shape, 3, 3))
    basis = np.concatenate((eye, eye * move[..., None, None]), axis=-1)
    offset = np.zeros((*time.shape, 3))
    offset[..., 2] = -drop
    return basis, offset


def _speed(state, tick):
    decay, move, _ = _terms(tick - ANCHOR)
    return state[3:] * decay - np.array((0., 0., GRAVITY * move))


def _recover_arc(history):
    points = [(tick, np.asarray(point), np.asarray(sigma)) for tick, point, sigma in history
              if tick in SOURCE_FRAMES and point is not None and sigma is not None]
    if len(points) < 5:
        return None
    ticks, positions, sigmas = map(np.asarray, zip(*points, strict=True))
    if ticks[-1] - ticks[0] < 5:
        return None
    basis, offset = _position_basis(ticks)
    weights = 1. / np.clip(sigmas, .012, .23)
    target = (positions - offset) * weights
    design = (basis * weights[..., None]).reshape(-1, 6)
    state = np.linalg.lstsq(design, target.ravel(), rcond=None)[0]
    residual = positions - (basis @ state + offset)
    if (not -.40 < state[1] < .32 or not .09 < state[2] < .72 or not -.17 < state[4] < -.025
            or np.median(np.linalg.norm(residual, axis=-1)) > .075):
        return None
    return state


@dataclass(frozen=True)
class ArcPrior:
    mean: np.ndarray
    covariance: np.ndarray
    arcs: int

    @classmethod
    def fit(cls, tracks, *, timing_shuffled=False):
        fitted = [arc for track in tracks if (arc := _recover_arc(track)) is not None]
        if len(fitted) < 4:
            return None
        if timing_shuffled:
            # Break alignment to the point clock without changing the visual
            # source, sample budget or available target sightings.
            shifted = []
            for index, arc in enumerate(fitted):
                tick = ANCHOR + (-4, 3, 4, -3)[index % 4]
                basis, offset = _position_basis(tick)
                shifted.append(np.r_[basis @ arc + offset, _speed(arc, tick)])
            fitted = shifted
        states = np.asarray(fitted)
        mean = np.median(states, axis=0)
        covariance = .70 * np.cov(states.T) + np.diag(NOISE_FLOOR**2)
        return cls(mean, covariance, len(fitted))

    def state(self):
        return dict(mean=self.mean.tolist(), covariance=self.covariance.tolist(), arcs=self.arcs)

    @classmethod
    def restore(cls, state):
        return cls(np.asarray(state['mean']), np.asarray(state['covariance']), state['arcs'])


class CarryArc:
    """Condition a table-coordinate source prior on a registered target stream."""

    def __init__(self, prior):
        self.prior = prior
        self.precision = np.linalg.inv(prior.covariance)
        self.history = deque(maxlen=9)

    def observe(self, tick, camera, blob, size_position, size_sigma):
        if tick < SOURCE_FRAMES.start or tick >= TARGET_FRAMES.stop or camera is None or blob is None:
            return None
        if self.history and tick - self.history[-1][0] > 4:
            self.history.clear()
        self.history.append((tick, camera, np.asarray(blob.center), size_position, size_sigma))
        if tick not in TARGET_FRAMES or len(self.history) < 4:
            return None
        if tick - self.history[0][0] < 4:
            return None

        ticks, cameras, pixels, sizes, sigmas = zip(*self.history, strict=True)
        basis, offset = _position_basis(ticks)
        eyes = np.stack([cam.eye for cam in cameras])
        axes = np.stack([cam.rotation for cam in cameras])
        focal = np.array([cam.focal for cam in cameras])
        centers = np.stack([cam.center for cam in cameras])
        rays = (np.asarray(pixels) - centers) / focal[:, None]
        direction = axes[:, :2] - rays[..., None] * axes[:, 2:3]
        nominal = basis @ self.prior.mean + offset
        nominal_depth = np.einsum('ni,ni->n', axes[:, 2], nominal - eyes)
        precision = focal / np.maximum(.08, nominal_depth)
        precision /= [.72 + .35 * cam.rms for cam in cameras]

        rows = direction * precision[:, None, None]
        design = np.einsum('nij,njk->nik', rows, basis).reshape(-1, 6)
        target = np.einsum('nij,nj->ni', rows, eyes - offset).ravel()
        valid = [index for index, (size, sigma) in enumerate(zip(sizes, sigmas, strict=True))
                 if size is not None and sigma is not None]
        if valid:
            weight = 1. / np.clip([np.max(sigmas[i]) * 1.35 for i in valid], .016, .30)
            depth_rows = axes[valid, 2] * weight[:, None]
            design = np.vstack((design, np.einsum('ni,nik->nk', depth_rows, basis[valid])))
            target = np.r_[target, np.einsum('ni,ni->n', depth_rows, np.stack([sizes[i] for i in valid]) - offset[valid])]

        system = design.T @ design + self.precision
        output = np.linalg.solve(system, design.T @ target + self.precision @ self.prior.mean)
        path = basis @ output + offset
        position, speed = path[-1], _speed(output, tick)
        residuals = np.linalg.norm([cam.project(point)[:2] - pixel
                                    for cam, point, pixel in zip(cameras, path, pixels, strict=True)], axis=1)
        if (residuals[-1] > 2.3 or np.median(residuals) > 1.8 or abs(position[0]) > .52
                or not -.82 < position[1] < .10 or not BALL_RADIUS + .003 < position[2] < .74
                or not -.19 < speed[1] < -.028):
            return None
        return position, speed
