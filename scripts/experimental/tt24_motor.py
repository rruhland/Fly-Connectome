"""Direct optical and proprioceptive learning for continuous actuator channels."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


FEATURES = ('bias', 'visible', 'x', 'y', 'size', 'dx', 'dy', 'dsize',
            'slow_x', 'slow_y', 'slow_size')
MEMORY_UNITS = 14


def motor_feature_count(readout):
    return MEMORY_UNITS if readout == 'recurrent' else len(FEATURES)


class SensorimotorMemory:
    """A recurrent response driven by optical changes and executed movement."""

    def __init__(self, axes):
        rng = np.random.default_rng(231704861)
        self.recurrent = np.linalg.qr(rng.normal(size=(MEMORY_UNITS, MEMORY_UNITS)))[0]
        inputs = 6 + axes * 3
        self.afferent = rng.normal(size=(MEMORY_UNITS, inputs)) / np.sqrt(inputs)
        self.state = np.zeros(MEMORY_UNITS)
        self.previous_action = np.zeros(axes)

    def advance(self, feature, position, velocity):
        optical = np.r_[feature[2:5] - feature[8:11], feature[5:8]]
        sensory = np.r_[2.8 * optical, 4. * position, .24 * velocity, self.previous_action]
        proposal = 1.58 * self.recurrent @ self.state + self.afferent @ sensory
        self.state = .70 * self.state + .30 * np.tanh(proposal)
        return self.state


@dataclass(frozen=True)
class Sighting:
    center: np.ndarray
    radius: float
    half_span: float


def _sighting(mask):
    # Measuring a colored component in the image provides angles and angular
    # size. No depth, pose, or camera projection matrix enters the policy.
    components, count = label(mask)
    if not count:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    chosen = sizes.argmax()
    yy, xx = np.nonzero(components == chosen)
    if len(xx) < max(5, mask.shape[0] // 10):
        return None
    height, width = mask.shape
    return Sighting(np.array((xx.mean() - (width - 1) / 2, yy.mean() - (height - 1) / 2)) / height,
                    float(np.sqrt(len(xx) / np.pi) / height),
                    float(.5 * (max(np.ptp(xx), np.ptp(yy)) + 1) / height))


def optical_sightings(rgb):
    """Return conventional hue-channel moments for ball and paddle."""
    if rgb is None:
        return None, None
    red, green, blue = np.moveaxis(rgb, -1, 0)
    ball = ((red > 68) & (green > .20 * red) & (green < .68 * red) & (blue < .34 * red))
    paddle = ((red > 54) & (green < .45 * red) & (blue < .56 * red))
    return _sighting(ball), _sighting(paddle)


class OpticalMemory:
    """Keep recurrent retinal features and aligned tracks for reward credit."""

    def __init__(self, *, retinal='absolute', detector=None):
        if retinal not in ('absolute', 'relative'):
            raise ValueError('expected absolute or release-relative retinal coordinates')
        self.retinal = retinal
        self.detector = detector
        self.current = np.zeros(3)
        self.slow = np.zeros(3)
        self.first = None
        self.missing = 0
        self.tracks = deque(maxlen=5)
        self.length = self.visible = self.paired = 0

    def observe(self, rgb, previous_action=None):
        if rgb is None or self.detector is None:
            ball, paddle = optical_sightings(rgb)
        else:
            ball, paddle = self.detector.observe(rgb, previous_action=previous_action)
        self.length += 1
        if ball is None:
            self.missing += 1
            self.tracks.clear()
            if self.missing > 1:
                self.current *= .72
                self.slow *= .9
            return np.r_[1., 0., self.current, np.zeros(3), self.slow]

        self.visible += 1
        anchor = paddle.center if self.retinal == 'relative' and paddle is not None else np.zeros(2)
        instant = ball.center - anchor
        if self.first is None:
            first_center = instant if self.retinal == 'relative' else np.zeros(2)
            self.first = first_center, max(ball.radius, .015)
        center, radius = self.first
        value = np.r_[np.tanh(4. * (instant - center)), np.tanh(np.log(max(ball.radius, .005) / radius))]
        change = np.clip(4. * (value - self.current), -1., 1.) if self.visible > 1 and not self.missing else np.zeros(3)
        self.current = value
        self.slow = .72 * self.slow + .28 * value
        self.missing = 0
        if paddle is None:
            self.tracks.clear()
        else:
            self.tracks.append((paddle.center, ball.center, paddle.half_span))
            self.paired += 1
        return np.r_[1., 1., value, change, self.slow]

    def consequence(self, prior, dt):
        if prior is None or len(self.tracks) != self.tracks.maxlen:
            return None
        actor, mover, scales = zip(*self.tracks, strict=True)
        result = prior.score_history(np.asarray(actor), np.asarray(mover), dt=dt,
                                     scale=float(np.median(scales)), details=True)
        return float(result['score'] - prior.prior), float(result['support'])


class MotorEffects:
    """Fit a local action-to-joint-velocity response by recursive least squares.

    Two available traces allow the fit to separate previous movement from
    the effect of an issued command. The fit can continue after a body change.
    """

    def __init__(self, axes):
        self.axes = axes
        self.readout = np.r_[np.eye(axes) * .16, np.eye(axes) * .76, np.zeros((1, axes))]
        self.precision = np.eye(2 * axes + 1) * 3.
        self.samples = 0
        self.error = 0.
        self.gain_reference = None
        self.gain_limits = None
        self.reset()

    def reset(self):
        self.previous = None
        self.drive = np.zeros(self.axes)

    def observe(self, velocity, *, plastic):
        velocity = np.asarray(velocity)
        if self.previous is not None and plastic:
            feature = np.r_[self.drive, self.previous / 4., 1.]
            error = velocity / 4. - feature @ self.readout
            gain = self.precision @ feature
            gain /= .998 + feature @ gain
            self.readout += np.outer(gain, error)
            self.precision = (self.precision - np.outer(gain, feature @ self.precision)) / .998
            self.error += float(error @ error)
            self.samples += 1
        self.previous = velocity

    def issued(self, action):
        self.drive = .55 * self.drive + .45 * action

    def gain(self):
        # Apply a bounded correction only to direct channel sensitivity.
        direct = np.maximum(.07, self.readout[:self.axes].diagonal())
        if self.gain_reference is None:
            return np.clip(.16 / direct, .55, 1.45)
        return np.clip(self.gain_reference / direct, *self.gain_limits)

    def track_change(self):
        """Preserve the present gains until measured channel sensitivity changes."""
        gains = self.gain()
        self.gain_reference = gains * np.maximum(.07, self.readout[:self.axes].diagonal())
        self.gain_limits = (.65 * gains, 1.5 * gains)


class MotorPolicy:
    """Actuator readout over visual memory, with measured joint stabilization."""

    def __init__(self, weights, effects, span=.25, *, readout='position', retinal='absolute', detector=None):
        self.weights = weights
        self.effects = effects
        self.span = span
        if readout not in ('position', 'rate', 'flow', 'recurrent'):
            raise ValueError('unknown motor readout')
        self.readout = readout
        self.optics = OpticalMemory(retinal=retinal, detector=detector)
        self.memory = SensorimotorMemory(weights.shape[0]) if readout == 'recurrent' else None
        self.home = None
        self.feature = np.r_[1., np.zeros(len(FEATURES) - 1)]
        self.last_action = None

    def command(self, observation, *, plastic=False, blind=False, sampled=True):
        if sampled:
            self.feature = self.optics.observe(None if blind else observation.get('rgb'), self.last_action)
        joints = observation['joints']
        velocity = observation['joint_velocity']
        if self.home is None:
            self.home = joints.copy()
            self.effects.reset()
        self.effects.observe(velocity, plastic=plastic)
        displacement = joints - self.home
        drive = self._drive(displacement, velocity)
        retained = -2.8 * displacement - .055 * velocity
        action = np.clip(self.effects.gain() * (retained + drive), -1., 1.)
        if self.memory is not None:
            self.memory.previous_action = action
        self.last_action = action
        self.effects.issued(action)
        return action

    def _drive(self, displacement, velocity):
        if self.memory is not None:
            state = self.memory.advance(self.feature, displacement, velocity)
            return 2.8 * self.span * np.tanh(self.weights @ state)
        if self.readout == 'position':
            return 2.8 * self.span * np.tanh(self.weights @ self.feature)
        feature = self.feature.copy()
        feature[0:2] *= .1
        feature[8:11] = 2. * (feature[2:5] - feature[8:11])
        feature[2:5] *= .3
        if self.readout == 'flow':
            feature[:5] = 0.
        return self.span * np.tanh(self.weights @ feature)


class RewardSearch:
    """Reward-gated perturbations of the recurrent sensorimotor readout.

    Noise is held across an episode to create an eligibility trace extending
    across delayed outcomes. A small population of past readouts preserves
    useful discoveries while exploring directions across all the actuators.
    """

    def __init__(self, axes, seed, *, weights=None, radius=.45, capacity=4, motor_modes=None,
                 features=len(FEATURES)):
        self.rng = np.random.default_rng(seed)
        self.shape = (axes, features)
        self.anchor = np.zeros(self.shape) if weights is None else weights.copy()
        self.radius = radius
        self.capacity = capacity
        self.motor_modes = np.eye(axes) if motor_modes is None else motor_modes
        self.elite = []
        self.trials = 0
        self._direction = None
        self._parent = None
        self._scale = None

    @property
    def best(self):
        return self.anchor if not self.elite else self.elite[0][1]

    def propose(self):
        if self.trials == 0:
            return self.anchor.copy()
        if self.trials % 2 == 0 and self._direction is not None:
            return self._parent - self._scale * self._direction
        rank = self.rng.choice(len(self.elite), p=self._rank_probability()) if self.elite else 0
        parent = self.elite[rank][1] if self.elite else self.anchor
        # Alternate a localized synapse-group perturbation and a low-rank
        # cross-actuator perturbation. Neither uses a task-specific axis.
        spatial = self.motor_modes @ self.rng.normal(size=self.shape[0])
        channel = self.rng.normal(size=self.shape[1])
        if self.trials % 4 == 1:
            channel *= self.rng.random(size=self.shape[1]) < .32
        self._direction = np.outer(spatial, channel) / np.sqrt(self.shape[1])
        self._parent = parent.copy()
        self._scale = self.radius * self.rng.choice((.55, 1., 1.7))
        return self._parent + self._scale * self._direction

    def _rank_probability(self):
        weights = .5 ** np.arange(len(self.elite))
        return weights / weights.sum()

    def update(self, weights, reward, credit=0.):
        self.trials += 1
        self.elite.append((reward + credit, weights.copy()))
        self.elite.sort(key=lambda item: item[0], reverse=True)
        del self.elite[self.capacity:]
