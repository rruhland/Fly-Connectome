"""Local policy learning with actual observed planar movements and scalar credit."""

from collections import deque
from dataclasses import dataclass

import numpy as np


FEATURE_COUNT = 25
RATE_VALUES = np.array((0., -1., 1.))


def silhouettes(frame):
    """Identify the elongated and compact foreground regions in fixed RGB."""
    panel = frame[..., 0]
    backdrop = min(panel[0, 0], panel[0, -1], panel[-1, 0], panel[-1, -1])
    mask = panel != backdrop
    occupancy = np.count_nonzero(mask, axis=1)
    body_rows = occupancy > max(6, int(.63 * occupancy.max()))
    by, bx = np.nonzero(mask & body_rows[:, None])
    yy, xx = np.nonzero(mask & ~body_rows[:, None])
    if len(bx) < 7 or len(xx) < 3:
        return None
    scale = .5 * (bx.max() - bx.min() + 1)
    return np.array(((bx.min() + bx.max()) / 2, (by.min() + by.max()) / 2)), np.array((xx.mean(), yy.mean())), scale


@dataclass(frozen=True)
class Reading:
    features: np.ndarray
    potential: float
    supported: float
    visible: bool
    appearance: tuple | None = None


class Retina:
    """Read positional, motion and temporal signals directly from rendered RGB."""

    def __init__(self, source, dt):
        self.source = source
        self.dt = dt
        self.history = deque(maxlen=5)
        self.previous = np.zeros(FEATURE_COUNT)
        self.previous[0] = 1.
        self.potential = 0.
        self.camera_center = None

    def update(self, frame, *, credit=False):
        return self.measure(silhouettes(frame), credit=credit)

    def measure(self, appearance, *, credit=False):
        if appearance is None:
            self.history.clear()
            return Reading(self.previous, self.potential, 0., False)
        actor, mover, scale = appearance
        if self.camera_center is None:
            self.camera_center = actor
        self.history.append((actor, mover))
        if len(self.history) < 4:
            return Reading(self.previous, self.potential, 0., True, appearance)

        actor_path, mover_path = (np.asarray(items) for items in zip(*self.history, strict=True))
        actor_flow = actor_path[-1] - actor_path[0]
        mover_flow = mover_path[-1] - mover_path[0]
        relative = (mover - actor) / scale
        body_offset = np.tanh((actor - self.camera_center) / (3. * scale))
        features = sensory_basis(relative, actor_flow / scale, mover_flow / scale, body_offset)
        self.previous = features
        support = 0.
        if credit:
            value = self.source.score_history(actor_path, mover_path, dt=self.dt, scale=scale, details=True)
            self.potential = float(value['score'] - self.source.prior)
            support = float(value['support'])
        return Reading(features, self.potential, support, True, appearance)


def sensory_basis(relative, agency_flow, object_flow, body_offset):
    """A shared smooth expansion of direct relative positions and optical differences."""
    near = np.exp(-.12 * np.sum(relative ** 2))
    offset = np.tanh(relative / 2.4)
    own = np.tanh(2.4 * agency_flow)
    other = np.tanh(2.4 * object_flow)
    return np.r_[1., offset, own, other, body_offset, near, near * offset, near * own, near * other,
                 offset * own, offset * other, offset[::-1] * other, offset[0] * offset[1],
                 other[0] * other[1], near ** 2]


class LocalCredit:
    """A linear actor and critic with local synaptic eligibility updates."""

    def __init__(self, axes, rng, *, actor_rate=.075, critic_rate=.09, source_gain=1., weights=None):
        self.rng = rng
        self.actor = np.zeros((axes, len(RATE_VALUES), FEATURE_COUNT))
        if weights is not None:
            self.actor[:len(weights)] = weights
        self.value = np.zeros(FEATURE_COUNT)
        self.actor_rate = actor_rate
        self.critic_rate = critic_rate
        self.source_gain = source_gain
        self.reset()

    def reset(self):
        self.actor_trace = np.zeros_like(self.actor)
        self.value_trace = np.zeros_like(self.value)

    def choose(self, reading, *, practice, bias=None):
        logits = self.actor @ reading.features
        if bias is not None:
            logits += bias
        if practice:
            policy = np.exp(logits - logits.max(axis=-1, keepdims=True))
            policy /= policy.sum(axis=-1, keepdims=True)
            choices = (np.cumsum(policy, axis=1) < self.rng.random((len(policy), 1))).sum(axis=1)
            choices = np.minimum(choices, len(RATE_VALUES) - 1)
            activity = np.eye(len(RATE_VALUES))[choices] - policy
        else:
            choices = np.argmax(logits, axis=-1)
            activity = None
        return RATE_VALUES[choices], activity

    def update(self, old, new, activity, reward, done):
        feature = old.features
        gamma = 0. if done else .965
        delta = reward + gamma * (self.value @ new.features) - self.value @ feature
        optical = self.source_gain * (gamma * new.potential - old.potential)
        self.actor_trace = .82 * self.actor_trace + activity[..., None] * feature
        self.value_trace = .74 * self.value_trace + feature
        self.actor += self.actor_rate * np.clip(delta + optical, -1.8, 1.8) * self.actor_trace
        self.value += self.critic_rate * np.clip(delta, -1.8, 1.8) * self.value_trace / (1. + feature @ feature)
        np.clip(self.actor, -4.5, 4.5, out=self.actor)
