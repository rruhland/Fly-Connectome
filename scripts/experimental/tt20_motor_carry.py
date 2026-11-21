"""Identify a signed public motor assignment from natural monocular motion."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment


WIDTH = 5
READ_TICKS = (2, 4, 7)
COMMANDS = (.45, .45, .45, .45, 0., 0., 0.)


@dataclass(frozen=True)
class Wiring:
    """Public index to canonical axis, including the public sign convention."""

    order: tuple[int, ...]
    signs: tuple[int, ...]

    def __post_init__(self):
        if sorted(self.order) != list(range(WIDTH)) or len(self.signs) != WIDTH or set(self.signs) - {-1, 1}:
            raise ValueError('expected a signed permutation of the five arm commands')

    def physical(self, public):
        value = np.asarray(public, dtype=float)
        actual = value.copy()
        actual[list(self.order)] = np.asarray(self.signs) * value[:WIDTH]
        return actual

    def inverse(self, canonical):
        value = np.asarray(canonical, dtype=float)
        public = value.copy()
        public[:WIDTH] = np.asarray(self.signs) * value[list(self.order)]
        return public

    def state(self):
        return dict(order=list(self.order), signs=list(self.signs))

    @classmethod
    def restore(cls, state):
        return cls(tuple(state['order']), tuple(state['signs']))


IDENTITY = Wiring(tuple(range(WIDTH)), (1,) * WIDTH)


def image_tiles(rgb):
    """Retain motion of the arm and ordinary scene; blank independently seeded balls."""
    r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
    orange = (r > 70.) & (.26 * r < g) & (g < .73 * r) & (b < .31 * r)
    orange = binary_dilation(orange, iterations=3)
    height, width = orange.shape
    block = 4
    if height % block or width % block:
        raise ValueError('native image dimensions must allow four-pixel tiles')
    values = rgb.astype(np.float32).reshape((height // block, block, width // block, block, 3))
    masks = orange.reshape((height // block, block, width // block, block))
    return values.mean(axis=(1, 3)) / 255., ~masks.any(axis=(1, 3))


def descriptor(images):
    """Subtract each camera's own reset image using only its native pixels."""
    start, mask = image_tiles(images[0])
    frames = [image_tiles(rgb) for rgb in images[1:]]
    return np.stack([color - start for color, _ in frames]), np.stack([visible & mask for _, visible in frames])


def optical_loss(observation, model, frames):
    measured, first = observation
    known, second = model
    usable = (first & second)[:frames]
    a, b = measured[:frames][usable], known[:frames][usable]
    return float(np.sum((a - b)**2) / max(.000001, float(np.sum(a**2) + np.sum(b**2))))


def identify(observations, atlas):
    """Fit a global assignment over every canonical port and both pulse signs."""
    if len(observations) != WIDTH or len(atlas) != WIDTH or any(len(pair) != 2 for pair in atlas):
        raise ValueError('need a complete five-port observation and simulation bank')
    fitted = []
    for used, tick in enumerate(READ_TICKS, 1):
        costs = np.array([[[optical_loss(seen, candidate, used) for candidate in pair] for pair in atlas]
                          for seen in observations])
        cheapest = costs.min(axis=2)
        rows, axes = linear_sum_assignment(cheapest)
        signs = 2 * np.argmin(costs[rows, axes], axis=1) - 1
        selected = float(cheapest[rows, axes].sum())
        alternatives = []
        for row, axis, sign in zip(rows, axes, signs, strict=True):
            blocked = cheapest.copy()
            blocked[row, axis] = np.inf
            x, y = linear_sum_assignment(blocked)
            alternatives.append(float(blocked[x, y].sum()))
            other_index = int(sign < 0)
            alternatives.append(selected - cheapest[row, axis] + float(costs[row, axis, other_index]))
        estimate = Wiring(tuple(map(int, axes)), tuple(map(int, signs)))
        fitted.append(dict(read_tick=tick, target_frames=WIDTH * tick, map=estimate.state(), loss=selected,
                           next_assignment_margin=min(alternatives) - selected, cost=costs.tolist()))
    if fitted[-1]['next_assignment_margin'] <= .01 or fitted[-1]['loss'] > .35 * WIDTH:
        raise ValueError('the optical responses do not support a unique signed assignment')
    return estimate, fitted


def capture_pulse(arena, eye, seed, channel, polarity, wiring):
    """Return images and executed commands; the estimator receives descriptors only."""
    arena.reset(seed=seed, server=-1, render=False)
    first = arena.render_cameras()
    if tuple(first) != (eye,):
        raise ValueError('a probe must use exactly its assigned native camera')
    images = [first[eye]]
    commands = []
    for tick, amplitude in enumerate(COMMANDS, 1):
        public = np.zeros(WIDTH + 1)
        public[channel] = polarity * amplitude
        _, _, done, expired, _ = arena.step(wiring.physical(public), render=False)
        if done or expired:
            raise ValueError('probe ended before its final optical observation')
        commands.append(public.tolist())
        if tick in READ_TICKS:
            images.append(arena.render_cameras()[eye])
    energy, visible = descriptor(images)
    sample = dict(seed=seed, channel=channel, polarity=polarity, commands=commands,
                  image_sha256=[sha256(rgb).hexdigest() for rgb in images], visibility=visible.mean(axis=(1, 2)).tolist(),
                  motion_energy=[float(np.sum(frame[mask]**2)) for frame, mask in zip(energy, visible, strict=True)])
    return (energy, visible), sample
