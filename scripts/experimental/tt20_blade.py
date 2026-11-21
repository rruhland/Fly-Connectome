"""Learn a small port-rate and timing correction from public blade pixels."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


WIDTH = 5


@dataclass(frozen=True)
class TimingFit:
    gains: np.ndarray
    delays: np.ndarray

    @classmethod
    def nominal(cls):
        return cls(np.ones(WIDTH), np.zeros(WIDTH))

    def model_commands(self, commands):
        data = np.asarray(commands)
        previous = np.vstack((np.zeros(WIDTH), data[:-1]))
        return np.clip(self.gains * (data - self.delays * (data - previous)), -1., 1.)

    def state(self):
        return dict(gains=self.gains.tolist(), delays_frames=self.delays.tolist())


def fit_timing(pixels, original, observed, commands, project, *, fit_frames=48):
    """Fit only measured RGB displacement relative to a nominal rendered tape."""
    finite = np.isfinite(pixels).all(axis=-1) & np.isfinite(observed).all(axis=-1)
    changed = observed - pixels
    used = finite.copy()
    used[fit_frames + 1:] = False
    if used.sum() < 65:
        raise ValueError('the single calibration did not retain enough paired blade pixels')

    def measurement(fit):
        forecast = project(fit.model_commands(commands))
        return forecast - original - changed

    def residual(params):
        estimate = TimingFit(1. + params[:WIDTH], params[WIDTH:])
        error = measurement(estimate)[used].ravel()
        # Keep a weak prior on nominal actuation, especially where a marker is occluded.
        return np.r_[error, 3.0 * params[:WIDTH], 1.4 * params[WIDTH:]]

    limit = np.r_[np.full(WIDTH, .16), np.full(WIDTH, .48)]
    solver = least_squares(residual, np.zeros(2 * WIDTH), bounds=(-limit, limit), max_nfev=22,
                           loss='soft_l1', f_scale=.8, diff_step=.001)
    learned = TimingFit(1. + solver.x[:WIDTH], solver.x[WIDTH:])
    no_correction = measurement(TimingFit.nominal())
    corrected = measurement(learned)

    def score(model, mask):
        delta = model[mask]
        return dict(sightings=int(len(delta)), rms_px=float(np.sqrt(np.mean(np.sum(delta ** 2, axis=-1)))))

    validation = finite & ~used
    if validation.sum() < 24:
        raise ValueError('too few independent frames remained to check the timing fit')
    metrics = {key: {'nominal': score(no_correction, mask), 'corrected': score(corrected, mask)}
               for key, mask in (('fit', used), ('check', validation))}
    return learned, metrics


class PulseCompensator:
    """Send bounded commands that compensate fitted rate and phase error."""

    def __init__(self, learned):
        self.fit = learned
        self.reset()

    def reset(self):
        self.expected = np.zeros(WIDTH)
        self.sent = np.zeros(WIDTH)

    def command(self, command):
        rates = np.asarray(command[:WIDTH])
        self.expected += rates
        target = (self.expected + self.fit.delays * rates) / self.fit.gains
        adjusted = np.clip(target - self.sent, -1., 1.)
        self.sent += adjusted
        return np.r_[adjusted, command[WIDTH:]]


class CorrectedPlayer:
    def __init__(self, base, fit):
        self.base = base
        self.correction = PulseCompensator(fit)

    @property
    def plans(self):
        return self.base.plans

    def reset(self):
        self.base.reset()
        self.correction.reset()

    def command(self, tick, point, server):
        return self.correction.command(self.base.command(tick, point, server))
