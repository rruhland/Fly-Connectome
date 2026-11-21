"""Use camera arc measurements and a rebound likelihood to steer the catcher."""

import numpy as np

from .tt8_transfer_rebound import DT, LIMIT, MOUTH, PLATE, RAIL, RATE, pixels


GRAVITY = 9.81
DRAG = .17
TANGENTIAL_RETENTION = .96
RESTITUTION = np.linspace(.42, 1.06, 65)
FEEDBACK_SIGMA = .075


def optical_prior(receipt, material):
    """Read only RGB fit results from points played on the named source plate."""
    groups = []
    observations = 0
    for visit in receipt['visits']:
        for point in visit['points']:
            if point['phase'] == material:
                samples = [sample['ratio'] for sample in point['table_bounce']['samples']]
                if samples:
                    groups.append(float(np.median(samples)))
                    observations += len(samples)
    if len(groups) < 3:
        raise ValueError(f'{material} has fewer than three independently observed points')
    center = float(np.median(groups))
    noise = float(max(.055, 1.4826 * np.median(np.abs(np.array(groups) - center))))
    return dict(center=center, spread=noise, points=len(groups), bounces=observations, medians=groups)


def fit_arc(samples):
    """Fit a local ballistic state in metric space with a shared drag approximation."""
    rows = np.asarray(samples)
    times = (rows[:, 0] - rows[-1, 0]) * DT
    position = rows[:, 1:]
    centered = times - times.mean()
    slope = centered / (centered @ centered)
    velocity = slope @ position
    for _ in range(2):
        acceleration = -DRAG * np.linalg.norm(velocity) * velocity - [0., GRAVITY]
        corrected = position - .5 * times[:, None]**2 * acceleration
        velocity = slope @ corrected
    return corrected.mean(axis=0) - velocity * times.mean(), velocity


def flow(z, vz, rate, elapsed):
    travel = -np.expm1(-rate * elapsed) / rate
    height = z + vz * travel - GRAVITY / rate * (elapsed - travel)
    vertical_velocity = vz * np.exp(-rate * elapsed) - GRAVITY * travel
    return travel, height, vertical_velocity


def crossing(z, vx, vz, height):
    """Find the later crossing of a fixed height under linearized drag."""
    # Vertical speed decays to zero across an ascending arc. Its initial speed
    # would overstate drag for the entire climb and the following descent.
    rate = DRAG * np.maximum(.4, np.hypot(vx, np.where(vz > 0., .58 * vz, vz)))
    apex = np.maximum(0., np.log1p(np.maximum(0., vz) * rate / GRAVITY) / rate)
    _, peak, _ = flow(z, vz, rate, apex)
    drop = np.maximum(0., peak - height)
    falling = np.maximum(0., -vz)
    # Start with the drag-free descent, then refine using the analytic curve.
    elapsed = apex + 2. * drop / np.maximum(.001, np.sqrt(falling**2 + 2. * GRAVITY * drop) + falling)
    for _ in range(4):
        _, projected, downward = flow(z, vz, rate, elapsed)
        correction = (height - projected) / np.minimum(-.001, downward)
        elapsed += np.where(drop > 0., correction, 0.)
    travel, _, vertical = flow(z, vz, rate, elapsed)
    return vx * travel, vx * np.exp(-rate * elapsed), vertical


def bank_predictions(position, velocity):
    x, z = position
    vx, vz = velocity
    span, horizontal, incoming = crossing(z, vx, vz, PLATE)
    outgoing = TANGENTIAL_RETENTION * horizontal
    onward, _, _ = crossing(PLATE, outgoing, -incoming * RESTITUTION, RAIL)
    return x + span + onward


def soft_window(error):
    """Soften the catch boundary to account for projection and flight-fit error."""
    return 1. / (1. + np.exp(np.clip((np.abs(error) - MOUTH) / .023, -45., 45.)))


class Catcher:
    def __init__(self, center, spread, *, learn, diffuse=0.):
        self.log_probability = -.5 * ((RESTITUTION - center) / spread)**2
        if diffuse:
            mass = self.probability
            self.log_probability = np.log((1. - diffuse) * mass + diffuse / len(mass))
        self.learn = learn

    @property
    def probability(self):
        mass = np.exp(self.log_probability - np.max(self.log_probability))
        return mass / mass.sum()

    @property
    def restitution(self):
        return float(self.probability @ RESTITUTION)

    def reset(self):
        self.samples = []
        self.frame = -1
        self.pre_bounce = None
        self.goal = None
        self.bank_policy = self.probability

    def act(self, rgb):
        self.frame += 1
        ball, cart = pixels(rgb)
        if ball is not None:
            self.samples.append((self.frame, *ball))
            self._aim()
        if self.goal is None:
            return 0.
        return float(np.clip((self.goal - cart) / (RATE * DT), -1., 1.))

    def _aim(self):
        if len(self.samples) < 4:
            return
        samples = self.samples
        low = min(range(len(samples)), key=lambda i: samples[i][2])
        after_bounce = samples[low][2] < .105 and low < len(samples) - 2
        if after_bounce:
            outgoing = samples[max(low + 1, len(samples) - 7):]
            if len(outgoing) >= 4:
                (x, z), (vx, vz) = fit_arc(outgoing)
                offset, _, _ = crossing(z, vx, vz, RAIL)
                self.goal = float(np.clip(x + offset, -LIMIT, LIMIT))
            return
        if samples[-1][2] < .095 or low < len(samples) - 3:
            return
        position, velocity = fit_arc(samples[-7:])
        predictions = bank_predictions(position, velocity)
        self.pre_bounce = predictions
        values = self.bank_policy @ soft_window(predictions[:, None] - predictions)
        self.goal = float(np.clip(predictions[np.argmax(values)], -LIMIT, LIMIT))

    def finish(self, rgb):
        """Use the publicly shown strike location as feedback after scoring."""
        if not self.learn or self.pre_bounce is None:
            return
        ball, _ = pixels(rgb)
        if ball is not None:
            residual = (ball[0] - self.pre_bounce) / FEEDBACK_SIGMA
            self.log_probability -= .5 * np.minimum(9., residual**2)
