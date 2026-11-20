"""Estimate a striker's rebound response from consecutive public hockey frames."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from air_hockey import HZ, PIXELS_PER_UNIT
from breakout_visual_rebound import _line


@dataclass(frozen=True)
class ShotContact:
    seconds: float
    offset: float
    separation: float
    incoming_angle: float
    outgoing_angle: float
    striker_velocity: float
    speed: float


class HockeyShotObserver:
    """Resolve a longitudinal reversal near the visible mobile striker."""

    def __init__(self):
        self.history = deque(maxlen=30)
        self.last_contact = -math.inf
        self.seconds = -1 / HZ

    def observe(self, image):
        self.seconds += 1 / HZ
        if image.puck is None or image.player is None:
            return None
        self.history.append((self.seconds, *image.puck, *image.player))
        if len(self.history) < 16 or self.seconds - self.last_contact < .22:
            return None
        history = np.asarray(self.history)
        if (history[-1, 1] - history[-7, 1] < .035
                or history[-1, 1] - history[-1, 3] > .28
                or history[-1, 0] - history[-8, 0] > .15):
            return None
        after = history[-8:]
        start, outgoing, miss = _line(after, self.seconds)
        if outgoing[0] < .28 or miss > 1 / PIXELS_PER_UNIT:
            return None
        best = None
        for end in range(len(history) - 8, 6, -1):
            before = history[end - 7:end]
            if self.seconds - before[-1, 0] > .27:
                break
            stop, incoming, past_miss = _line(before, self.seconds)
            if incoming[0] > -.24 or past_miss > 1 / PIXELS_PER_UNIT:
                continue
            delay = (stop[0] - start[0]) / (outgoing[0] - incoming[0])
            when = self.seconds + delay
            frame_time = 1 / HZ
            if not before[-1, 0] - frame_time <= when <= after[0, 0] + frame_time:
                continue
            pre, post = stop + delay * incoming, start + delay * outgoing
            if abs(pre[1] - post[1]) > .015:
                continue
            center = [np.interp(when, history[:, 0], history[:, i]) for i in (3, 4)]
            offset = (pre + post) * .5 - center
            separation = float(np.linalg.norm(offset))
            if not .044 <= separation <= .088 or offset[0] <= .02:
                continue
            near = history[abs(history[:, 0] - when) <= .052]
            if len(near) < 3:
                continue
            _, striker, _ = _line(near[:, (0, 3, 4)], when)
            cost = past_miss + abs(pre[1] - post[1])
            if best is None or cost < best[0]:
                best = cost, ShotContact(float(when), float(offset[1]), separation,
                    math.atan2(incoming[1], -incoming[0]), math.atan2(outgoing[1], outgoing[0]),
                    float(striker[1]), float(np.linalg.norm(outgoing)))
        if best is None:
            return None
        shot = best[1]
        self.last_contact = shot.seconds
        return shot


def fit_shot_response(shots, reach):
    """Fit local angular gain and keep incoming-direction and movement terms visible."""
    if len(shots) < 8 or not math.isfinite(reach) or reach <= 0.:
        raise ValueError('eight observed contacts and positive reach are required')
    features = np.array([(shot.offset / reach, shot.incoming_angle,
                          shot.striker_velocity / shot.speed) for shot in shots])
    if np.ptp(features[:, 0]) < .40:
        raise ValueError('observed contacts need more lateral variation')
    angles = np.array([shot.outgoing_angle for shot in shots])
    coefficients, _, rank, _ = np.linalg.lstsq(features, angles, rcond=None)
    if rank != features.shape[1]:
        raise ValueError('the observed motions do not identify the response')
    return tuple(float(value) for value in coefficients)


def predict_shot_angle(shot, reach, coefficients):
    offset, incoming, motion = coefficients
    return offset * shot.offset / reach + incoming * shot.incoming_angle + motion * shot.striker_velocity / shot.speed
