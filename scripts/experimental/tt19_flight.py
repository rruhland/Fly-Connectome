"""Fit a ball's flight and table response from sequential camera tracks."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FlightLaw:
    """Distances are metres above the tabletop; time is in 30 Hz frames."""

    gravity: float = .0109
    drag: float = .17
    restitution: float = .86
    tangent: float = .945

    @classmethod
    def from_state(cls, state):
        return cls(**state)

    def state(self):
        return asdict(self)

    def trajectory(self, position, velocity, horizon, *, bounds=(.7625, 1.37)):
        p, v = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
        points, speeds = [p], [v]
        for _ in range(horizon):
            a = -self.drag * np.linalg.norm(v, axis=-1, keepdims=True) * v
            a[..., 2] -= self.gravity
            end = p + v + a * .5
            hit = (p[..., 2] >= .020) & (end[..., 2] < .020)
            frac = np.clip((p[..., 2] - .020) / np.maximum(.000001, p[..., 2] - end[..., 2]), 0., 1.)
            frac = np.asarray(frac)[..., None]
            contact = p + frac * (v + a * frac * .5)
            hit &= (abs(contact[..., 0]) <= bounds[0]) & (abs(contact[..., 1]) <= bounds[1])
            bounce = (v + frac * a) * [self.tangent, self.tangent, -self.restitution]
            remaining = 1. - frac
            contact[..., 2] = .020
            p = np.where(hit[..., None], contact + remaining * (bounce + remaining * a * .5), end)
            v = np.where(hit[..., None], bounce + remaining * a, v + a)
            points.append(p)
            speeds.append(v)
        return np.array(points), np.array(speeds)

    def predict(self, ball_history, horizons, role=None, *, bounds=(.7625, 1.37)):
        """Forecast using only optical history through the most recent frame."""
        start = optical_start(ball_history, self.gravity)
        if start is None:
            return None
        horizons = np.asarray(horizons, dtype=int)
        positions, _ = self.trajectory(*start, int(horizons.max()), bounds=bounds)
        return positions[horizons]


def optical_start(history, gravity):
    """Estimate current speed from up to four fresh camera sightings."""
    if len(history) < 2:
        return None
    tick = history[-1][0]
    recent = [(t, xyz) for t, xyz in history[-6:] if xyz is not None and tick - t <= 4]
    if len(recent) < 2 or recent[-1][0] != tick:
        return None
    ts, points = zip(*recent[-4:])
    ts, points = np.asarray(ts, float), np.asarray(points, float)
    # A change of direction at the surface or a blade invalidates the old arc.
    diff = np.diff(points, axis=0) / np.diff(ts)[:,None]
    for i in range(len(diff) - 1, 0, -1):
        if ((diff[i-1,2] < -.018 and diff[i,2] > .018 and points[i,2] < .095)
                or (diff[i,1] * diff[i-1,1] < -.0006)):
            ts, points = ts[i:], points[i:]
            break
    t = ts - tick
    centered = t - t.mean()
    corrected = points.copy()
    corrected[:,2] += .5 * gravity * t**2
    velocity = centered @ corrected / (centered @ centered)
    if not np.all(np.isfinite(velocity)) or np.linalg.norm(velocity) > .36:
        return None
    return points[-1], velocity


HORIZONS = (3, 6, 10)


def windows(traces):
    """Collect optical past/future pairs without crossing a point boundary."""
    rows = []
    for trace in traces:
        for tick in range(3, len(trace) - max(HORIZONS), 2):
            recent = list(enumerate(trace[tick-3:tick+1]))
            if any(xyz is None for _, xyz in recent):
                continue
            targets = [trace[tick+h] for h in HORIZONS]
            if any(xyz is None for xyz in targets):
                continue
            init = optical_start(recent, 0.)
            if init is None:
                continue
            # The only explicit gravity dependency in the velocity estimate is
            # linear. Recover it from a small finite difference on the same arc.
            perturbed = optical_start(recent, .001)
            if perturbed is None:
                continue
            rows.append((init[0], init[1], (perturbed[1] - init[1]) / .001, targets))
    if not rows:
        return None
    return tuple(np.asarray(items) for items in zip(*rows))


def errors(law, data, bounds=(.7625, 1.37)):
    position, speed, correction, target = data
    forecast, _ = law.trajectory(position, speed + correction * law.gravity, max(HORIZONS), bounds=bounds)
    return np.moveaxis(forecast[list(HORIZONS)], 0, 1) - target


def fit_flight(data, bounds=(.7625, 1.37)):
    nominal = FlightLaw()
    def residual(parameters):
        return errors(FlightLaw(*parameters), data, bounds).ravel()
    # Robust residuals allow unlabelled strikes, net contacts and noisy pixels.
    result = least_squares(residual, list(nominal.state().values()), loss='soft_l1', f_scale=.018,
                           bounds=([.002, .005, .35, .55], [.025, .50, 1.12, 1.10]),
                           x_scale=[.01, .15, .85, .95], max_nfev=70)
    return FlightLaw(*result.x)
