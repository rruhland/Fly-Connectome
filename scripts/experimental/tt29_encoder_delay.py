"""Identify command delay and local joint response using issued targets and encoders."""

from collections import deque

import numpy as np


class EncoderDelay:
    def __init__(self, maximum=4):
        self.maximum = maximum
        self.targets = deque(maxlen=96)
        self.encoders = deque(maxlen=96)
        self.coefficients = None
        self.lag = 0
        self.samples = 0
        self.losses = []
        self.votes = deque(maxlen=3)

    def observe(self, angles, commanded, fixed=None):
        # Both arrive before issuing this tick's next action. commanded includes
        # the most recent action requested on the previous tick.
        self.encoders.append(angles.copy())
        self.targets.append(commanded.copy())
        self.samples += 1
        if self.samples % 8 or len(self.targets) < 26:
            return
        q = np.stack(self.encoders)
        target = np.stack(self.targets)
        base = np.arange(self.maximum + 1, len(q) - 1)
        evaluate = base[-8:]
        train = base[:-8]
        active = np.ptp(target, axis=0) > 0.025
        if not active.any():
            return
        fits = []
        for lag in range(self.maximum + 1):
            velocity = q[train] - q[train - 1]
            error = target[train + 1 - lag] - q[train]
            design = np.stack([velocity, error], axis=-1)
            response = q[train + 1] - q[train]
            gram = np.einsum("tji,tjk->jik", design, design) + np.eye(2)[None] * 0.0001
            vec = np.einsum("tji,tj->ji", design, response)
            beta = np.linalg.solve(gram, vec[..., None])[..., 0]
            # Accelerating and braking obey the same signed local servo relation.
            pred = (q[evaluate] - q[evaluate - 1]) * beta[:, 0] + (
                target[evaluate + 1 - lag] - q[evaluate]
            ) * beta[:, 1]
            loss = np.mean(np.square((q[evaluate + 1] - q[evaluate] - pred)[:, active]))
            fits.append((float(loss), beta))
        self.losses = [f[0] for f in fits]
        best = int(np.argmin(self.losses))
        self.votes.append(best)
        self.lag = int(np.median(self.votes)) if fixed is None else fixed
        self.coefficients = fits[self.lag][1]

    def predict(self, position, delta):
        if self.lag == 0:
            return position, delta
        if self.coefficients is None:
            return position + self.lag * delta, delta
        q = position.copy()
        v = delta.copy()
        for target in list(self.targets)[-self.lag :]:
            v = np.clip(
                self.coefficients[:, 0] * v + self.coefficients[:, 1] * (target - q),
                -0.095,
                0.095,
            )
            q += v
        return q, v


def excitation(action_size):
    """A half-second, zero-jaw probe with visible acceleration and braking."""
    rng = np.random.default_rng(290705)
    commands = np.repeat(rng.choice([-0.3, 0.3], size=(8, action_size)), 3, axis=0)
    commands[:, -1] = 0.0
    return np.vstack((commands, np.zeros((8, action_size))))


def encoder_trace(station, commands, deliver):
    station.reset(seed=290705, server=-1, render=False)
    measured = []
    for command in commands:
        _, _, done, truncated, _ = station.exchange(deliver(command), np.zeros(5))
        measured.append(station.joints.copy())
        if done or truncated:
            break
    return np.asarray(measured)


def predicted_trace(nominal, commands, lag):
    queue = deque(np.zeros((lag, nominal.action_size)))

    def deliver(command):
        queue.append(command)
        return queue.popleft()

    return encoder_trace(nominal, commands, deliver)


def calibrate_delay(plant, nominal, deliver, maximum=4, *, body=False):
    """Fit transport lag from a paired physical and nominal encoder-motion probe."""
    commands = excitation(plant.action_size)
    observed = encoder_trace(plant, commands, deliver)
    errors = []
    for length in range(maximum + 1):
        prediction = predicted_trace(nominal, commands, length)
        n = min(len(prediction), len(observed))
        residual = np.diff(prediction[:n], axis=0) - np.diff(observed[:n], axis=0)
        errors.append(np.mean(residual[:, :-1] ** 2))
    lag = int(np.argmin(errors))
    result = dict(ticks=len(observed), mse=list(map(float, errors)))
    if body:
        from .tt29_body_fit import fit

        result["body"] = fit(nominal, commands, lag, observed)
    return lag, result
