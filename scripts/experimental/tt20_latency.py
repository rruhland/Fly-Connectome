"""Learn command-transport latency from a single view, then plan with a clock lead."""

from collections import deque

import numpy as np

from .tt18_view import ImageColors
from .tt19_view import _blob
from .tt19_model_rollout import RolloutPlayer
from .tt19_shoot import COURT


class CommandPipe:
    """The evaluator's FIFO, upstream of the real simulated motor actuation."""

    def __init__(self, delay, width=6):
        if type(delay) is not int or delay < 0:
            raise ValueError('expected a non-negative control-frame delay')
        self.width = width
        self.queue = deque(np.zeros(width) for _ in range(delay))

    def apply(self, requested):
        command = np.asarray(requested, dtype=float)
        if command.shape != (self.width,) or not np.isfinite(command).all():
            raise ValueError('expected finite normalized port rates')
        self.queue.append(np.clip(command, -1., 1.))
        return self.queue.popleft()


def blade_motion(frames):
    samples = []
    for rgb in frames:
        colors = ImageColors(rgb)
        blade = _blob(colors, .993, tolerance=.027, minimum=57., area=4)
        samples.append(None if blade is None else blade.center)
    return np.array([0. if a is None or b is None else float(np.linalg.norm(b-a))
                     for a, b in zip(samples, samples[1:])])


def scene_motion(frames):
    def background(rgb):
        h, w = rgb.shape[:2]
        # Keep the background above and alongside the held blade. The wrist
        # itself need not be visible for the static hall to move in the image.
        frames = (rgb[h//10:h//2, w//9:8*w//9], rgb[h//2:3*h//4, :w//5], rgb[h//2:3*h//4, 4*w//5:])
        return np.concatenate([crop[::2, ::2].mean(axis=2).ravel() for crop in frames])
    samples = [background(image) for image in frames]
    return np.array([float(np.mean(abs(b-a))) for a, b in zip(samples, samples[1:])])


def identify_delay(template, measured, maximum=6):
    reference, response = map(np.asarray, (template, measured))
    if (reference.ndim != 1 or response.shape != reference.shape or not len(reference) > maximum
            or min(np.linalg.norm(reference), np.linalg.norm(response)) < .1):
        raise ValueError('the RGB probe had insufficient visible motion')
    normalized = response / np.linalg.norm(response)
    loss = []
    for delay in range(maximum + 1):
        shifted = np.r_[np.zeros(delay), reference[:len(reference)-delay]]
        guess = shifted / max(.0001, np.linalg.norm(shifted))
        loss.append(float(np.linalg.norm(guess - normalized)))
    ordered = np.argsort(loss)
    return dict(delay_frames=int(ordered[0]), losses=loss, margin=float(loss[ordered[1]] - loss[ordered[0]]))


class LatencyPlayer(RolloutPlayer):
    """Forecast the visible ball at the first physical servo command."""

    def __init__(self, catalog, law, response, delay, *, count=32):
        super().__init__(catalog, law, response, count=count)
        self.delay = delay

    def choose(self, estimate):
        if not self.delay:
            return super().choose(estimate)
        points, velocities = self.law.trajectory(*estimate, self.delay, bounds=COURT)
        detail = super().choose((points[-1], velocities[-1]))
        if detail is not None:
            detail['delay_lead_frames'] = self.delay
        return detail
