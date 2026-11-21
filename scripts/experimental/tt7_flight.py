"""Learn systematic errors in the outgoing flight model from stereo trajectories."""
from collections import deque

import numpy as np

from .tt5_player import BALL_RADIUS, GRAVITY, DRAG, NET_CLEARANCE, TABLE_END, _launch_pitch, _landing_cost
from .tt6_continuous import ContinualPlayer


def outgoing_speed(trace, contact):
    """Extrapolate four coherent, departing observations to estimated contact time."""
    seen = [(f['tick'], f['ball']) for f in trace if f['ball'] is not None and
            contact - 3 <= f['tick'] <= contact + 23]
    for i in range(len(seen) - 3):
        times, points = zip(*seen[i:i + 4])
        dt = np.array(times, dtype=float) - contact
        pos = np.asarray(points)
        if dt[0] > 16 or not -1.85 < pos[0, 1] < -.05 or max(np.diff(times)) > 5:
            continue
        if np.any(np.diff(pos[:, 1]) < .012) or np.any(pos[:, 2] < .055):
            continue
        centered = dt - dt.mean()
        v = centered @ pos / (centered @ centered)
        if not .018 < v[1] < .25:
            continue
        acc = -DRAG * np.linalg.norm(v) * v - [0., 0., GRAVITY]
        corrected = pos - .5 * dt[:, None]**2 * acc
        v = centered @ corrected / (centered @ centered)
        intercept = corrected.mean(axis=0) - dt.mean() * v
        residual = np.linalg.norm(corrected - intercept - dt[:, None] * v, axis=1)
        if residual.max() <= .055 and -1.95 < intercept[1] < -.90:
            return v[1:], intercept, times
    return None


def nominal_exit(incoming, angle, forward, restitution):
    n = np.array([np.cos(angle), np.sin(angle)])
    t = np.array([-n[1], n[0]])
    relative = incoming - [forward, 0.]
    return incoming - (1. + restitution) * (relative @ n) * n - .23 * (relative @ t) * t


class FlightMemory:
    def __init__(self, mode, history=(), impulses=(), *, frozen=False):
        self.mode = mode
        self.frozen = frozen
        self.history = deque(history, maxlen=16)
        self.impulses = deque(impulses, maxlen=32)
        self.response_matrix = self._fit_impulse()

    def _fit_impulse(self):
        # Units are metres per video frame. A damped two-axis impulse prior
        # gives a usable prediction before enough distinct strokes have occurred.
        regularizer = np.diag([.07**2, .06**2]) * 2.
        prior = np.diag([-1.80, -.23])
        if not self.impulses:
            return prior
        states, outputs = np.moveaxis(np.asarray(self.impulses), 1, 0)
        prediction = states @ prior
        error = np.linalg.norm(outputs - prediction, axis=1)
        weight = np.minimum(1., .10 / np.maximum(error, .001))
        return np.linalg.solve(states.T @ (weight[:, None] * states) + regularizer,
                               states.T @ (weight[:, None] * outputs) + regularizer @ prior)

    def sample(self, inputs, observed, nominal):
        if self.frozen:
            return
        self.history.append((observed - nominal).tolist())
        incoming, angle, forward = inputs
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        relative = (incoming - [forward, 0.]) @ rotation
        impulse = (observed - incoming) @ rotation
        self.impulses.append([relative.tolist(), impulse.tolist()])
        self.response_matrix = self._fit_impulse()

    def choose_angle(self, position, incoming, forward, restitution):
        if self.mode == 'risk':
            return self._robust_angle(position, incoming[1:], forward, restitution)
        if self.mode != 'impulse':
            return _launch_pitch(position, incoming, forward, restitution, self.correction())
        angles = np.linspace(-.02, .72, 51)
        sine, cosine = np.sin(angles), np.cos(angles)
        normals = np.stack([cosine, sine], axis=1)
        tangents = np.stack([-sine, cosine], axis=1)
        relative = incoming[1:] - [forward, 0.]
        local = np.column_stack([normals @ relative, tangents @ relative])
        impulse = local @ self.response_matrix
        outgoing = incoming[1:] + normals * impulse[:, :1] + tangents * impulse[:, 1:]
        return float(angles[np.argmin(_landing_cost(position, np.maximum(.020, outgoing[:, 0]), outgoing[:, 1]))])

    def _robust_angle(self, position, incoming, forward, restitution):
        angles = np.linspace(-.02, .72, 51)
        cosine, sine = np.cos(angles), np.sin(angles)
        normal = (incoming[0] - forward) * cosine + incoming[1] * sine
        tangent = -(incoming[0] - forward) * sine + incoming[1] * cosine
        outgoing = incoming[0] - (1. + restitution) * normal * cosine + .23 * tangent * sine
        rising = incoming[1] - (1. + restitution) * normal * sine - .23 * tangent * cosine
        # A symmetric starting envelope is replaced by the joint distribution of
        # observed errors, retaining correlations between forward speed and rise.
        samples = np.array([[-.028, 0.], [.028, 0.], [0., -.014], [0., .014], *self.history])
        out = np.maximum(.02, outgoing[:, None] + samples[:, 0])
        rise = rising[:, None] + samples[:, 1]
        loss = DRAG * out

        def height(at):
            intervals = np.expm1(DRAG * (at - position[1])) / loss
            decay = -np.expm1(-loss * intervals) / loss
            return position[2] + (rise + GRAVITY / loss) * decay - GRAVITY / loss * intervals

        def clearance(margin):
            return 1. / (1. + np.exp(-np.clip(margin / .043, -45., 45.)))

        expected_return = np.mean(clearance(height(0.) - NET_CLEARANCE) *
                                  clearance(BALL_RADIUS - height(TABLE_END)), axis=1)
        tie_cost = .025 * np.mean(np.minimum(1., np.abs(height(.76) - BALL_RADIUS)), axis=1)
        return float(angles[np.argmax(expected_return - tie_cost)])

    def correction(self):
        if self.mode == 'fixed':
            return np.array([-.040, -.018])
        if self.mode == 'observe' or not self.history:
            return np.zeros(2)
        samples = np.clip(self.history, [-.115, -.074], [.115, .074])
        return np.median(samples, axis=0) * min(1., len(samples) / 3.)


class FlightPlayer(ContinualPlayer):
    def __init__(self, *args, flight_memory, **kwargs):
        self.memory = flight_memory
        self.prediction = None
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.pending = None
        self.last_strike = -100
        self.measurements = []
        self.prediction = None

    def _strike_pitch(self, position, incoming):
        angle = self.memory.choose_angle(position, incoming, self.stroke_speed, self.impact_restitution)
        self.prediction = incoming[1:].copy(), angle, self.stroke_speed
        return angle

    def _measure(self):
        if self.pending is None:
            return
        contact, inputs = self.pending
        fit = outgoing_speed(self.trace, contact)
        if fit is None:
            return
        observed, position, ticks = fit
        nominal = nominal_exit(*inputs, self.impact_restitution)
        self.memory.sample(inputs, observed, nominal)
        self.measurements.append(dict(contact=contact, observed=observed.tolist(), nominal=nominal.tolist(),
                                      position=position.tolist(), sightings=list(ticks), approach=inputs[0].tolist(), pitch=inputs[1], speed=inputs[2]))
        self.pending = None

    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        last = self.trace[-1]
        tick, horizon = last['tick'], last['horizon']
        if self.pending is not None and tick - self.pending[0] >= 16 and tick % 4 == 0:
            self._measure()
        if (horizon is not None and 0. <= horizon < 3. and self.prediction is not None
                and tick - self.last_strike > 22):
            self._measure()
            self.last_strike = tick
            self.pending = tick + horizon, self.prediction
        return command

    def finish(self, reward):
        self._measure()
        details = super().finish(reward)
        details['outgoing_model'] = dict(measurements=self.measurements,
                                         correction=self.memory.correction().tolist())
        return details
