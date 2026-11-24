"""Causal RGB trajectory fit with optional latent spin and a visible rebound impulse."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .tt4_arena import BALL_INERTIA, BALL_MASS, BALL_RADIUS, SPIN_DECAY
from .tt29_rally_physics import DRAG, LIFT


VERTICAL = np.array([0., 0., -9.81])
TORQUE = BALL_MASS * BALL_RADIUS / BALL_INERTIA * np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]])
SLIP = BALL_RADIUS * np.array([[0., -1., 0.], [1., 0., 0.]])
SIGMA = np.array([.0010, .0025, .0012])
MAX_ARC = .44
MAX_BOUNCE = .19
MODES = ('zero', 'impulse', 'arc', 'contact0', 'coupled')
NODE, WEIGHT = np.polynomial.legendre.leggauss(5)
NODE = (NODE + 1.) / 2.
WEIGHT /= 2.


def damped(t, damping):
    take = -np.expm1(-damping * t) / damping
    return take, (t - take) / damping


def velocity_fit(t, p):
    """A rough drag and gravity-corrected linearization point at t = 0."""
    delta = t - t.mean()
    slope = delta @ p / (delta @ delta)
    a = VERTICAL - DRAG * slope * np.linalg.norm(slope)
    v = slope - a * (delta @ (t*t)) / (2. * (delta @ delta))
    return v.clip(-12., 12.)


def kernels(t, ref):
    """First Picard drag correction plus integration of the linearized Magnus term."""
    t = np.atleast_1d(t)
    damp = DRAG * np.clip(np.linalg.norm(ref), 2., 12.)
    s = t[:, None] * NODE
    travel, _ = damped(s, damp)
    vel = np.exp(-damp * s)[..., None] * ref + travel[..., None] * VERTICAL
    force = vel * (damp - DRAG * np.linalg.norm(vel, axis=2))[..., None]
    spin = LIFT * np.exp(-SPIN_DECAY * s)[..., None, None] * np.cross(
        np.eye(3)[None, None, :, :], vel[..., None, :]).swapaxes(2, 3)
    late = t[:, None] - s
    h, _ = damped(late, damp)
    wp = WEIGHT * t[:, None] * h
    wv = WEIGHT * t[:, None] * np.exp(-damp * late)
    length, fall = damped(t, damp)
    fixed_p = fall[:, None] * VERTICAL + np.einsum('nu,nui->ni', wp, force)
    fixed_v = length[:, None] * VERTICAL + np.einsum('nu,nui->ni', wv, force)
    spin_p = np.einsum('nu,nuij->nij', wp, spin)
    spin_v = np.einsum('nu,nuij->nij', wv, spin)
    return length, np.exp(-damp * t), fixed_p, fixed_v, spin_p, spin_v


def model(t, p, split=None):
    """Return one shared linearization for all spin ablations."""
    before = np.ones(len(t), dtype=bool) if split is None else t < -split
    after = np.zeros(len(t), dtype=bool) if split is None else t > split
    use = before | after
    if before.sum() < 3 or (split is not None and after.sum() < 2):
        return None
    shape = (len(t), 3, 9 if split is not None else 6)
    base = np.zeros(shape)
    impulse = np.zeros(shape)
    spin = np.zeros((len(t), 3, 3))
    fixed = np.zeros_like(p)
    base[:, :, :3] = np.eye(3)
    refs = []
    for phase, mask in enumerate((before, after)):
        if not mask.any():
            break
        ref = velocity_fit(t[mask], p[mask])
        refs.append(ref)
        length, _, f, _, spin_p, _ = kernels(t[mask], ref)
        col = (slice(3, 6), slice(6, 9))[phase]
        base[mask, :, col] = length[:, None, None] * np.eye(3)
        spin[mask] = spin_p
        if phase:
            delta = spin_p @ TORQUE
            impulse[mask, :, 3:6] = -delta
            impulse[mask, :, 6:9] = delta
        fixed[mask] = f
    return base[use], impulse[use], spin[use], fixed[use], refs, use


@dataclass(frozen=True)
class Estimate:
    time: float
    bounce: float | None
    position: np.ndarray
    pre: np.ndarray
    post: np.ndarray | None
    omega: np.ndarray
    impulse: bool
    refs: tuple
    residual: float
    posterior: np.ndarray

    def state(self, now):
        t = now - self.time
        after = self.post is not None and t >= 0.
        v0 = self.post if after else self.pre
        w0 = self.omega.copy()
        if after and self.impulse:
            w0 += TORQUE @ (self.post - self.pre)
        a, d, p, v, j, h = kernels(t, self.refs[int(after)])
        xyz = self.position + a[0] * v0 + p[0] + j[0] @ w0
        vel = d[0] * v0 + v[0] + h[0] @ w0
        return np.r_[xyz, vel], w0 * np.exp(-SPIN_DECAY * t)


def solve(t, p, packet, settings, mode, instant, bounce):
    base, impulse, spin, fixed, ref, use = packet
    latent = mode in ('arc', 'coupled')
    transfer = mode != 'zero'
    x = base + impulse if transfer else base
    if latent:
        x = np.concatenate((x, spin), axis=2)
    columns = x.shape[-1]
    q = (x / SIGMA[:, None]).reshape(-1, columns)
    target = ((p[use] - fixed) / SIGMA).ravel()
    rows, values = [], []
    if bounce is not None:
        height = np.zeros(columns)
        height[2] = 1. / .0045
        rows.append(height)
        values.append(settings['height'] / .0045)
        normal = np.zeros(columns)
        normal[5] = settings['restitution'] / .19
        normal[8] = 1. / .19
        rows.append(normal)
        values.append(0.)
        if mode in ('coupled', 'contact0') and settings['friction'] > 0.:
            # Soft effective-slip constraint, not an assumed no-slip collision.
            k = settings['friction']
            eq = np.zeros((2, columns))
            eq[:, 3:5] = (k - 1.) * np.eye(2)
            eq[:, 6:8] = np.eye(2)
            if latent:
                eq[:, -3:] = k * SLIP
            rows.extend(eq / .16)
            values.extend((0., 0.))
    if latent:
        rows.extend(np.eye(columns)[-3:] / settings['spin_prior'])
        values.extend((0., 0., 0.))
    if rows:
        q = np.vstack((q, rows))
        target = np.r_[target, values]
    normal = q.T @ q + 1e-9 * np.eye(columns)
    coeff = np.linalg.solve(normal, q.T @ target)
    misfit = np.linalg.norm(np.einsum('nij,j->ni', x, coeff) + fixed - p[use], axis=1)
    post = None if bounce is None else coeff[6:9]
    omega = coeff[-3:] if latent else np.zeros(3)
    cov = np.sqrt(np.diag(np.linalg.inv(normal))[-3:]) if latent else np.zeros(3)
    return Estimate(instant, bounce, coeff[:3], coeff[3:6], post, omega, transfer,
                    tuple(ref), float(np.sqrt(np.mean(misfit**2))), cov)


class Observer:
    """Opt-in observer with only exposure timestamps and triangulated RGB centers."""

    def __init__(self, period, settings):
        self.period = period
        self.settings = settings
        self.samples = deque(maxlen=44)

    def reset(self):
        self.samples.clear()

    def ingest(self, now, point):
        if (not self.samples or now - self.samples[-1][0] > self.period * 2.1
                or now <= self.samples[-1][0]):
            self.samples.clear()
        if point is not None and np.isfinite(point).all():
            self.samples.append((float(now), np.asarray(point)))

    def evidence(self):
        if len(self.samples) < 4:
            return None
        t, p = map(np.array, zip(*self.samples, strict=True))
        now = t[-1]
        if np.median(np.diff(p[-4:, 1]) / np.diff(t[-4:])) > -1.:
            return None
        current = self.settings
        # The detector uses only earlier observations. Share the clock across all ablations.
        visible = (t >= now - MAX_BOUNCE) & (p[:, 1] > -1.42) & (p[:, 1] < -.03) & (abs(p[:, 0]) < .75)
        good = []
        for k in range(max(3, len(t)-13), len(t)-1):
            if not visible[k] or not p[k, 2] < min(.846, p[k-1, 2], p[k+1, 2]):
                continue
            early = (p[k-1, 2]-p[max(0, k-3), 2]) / (t[k-1]-t[max(0, k-3)])
            late = (p[-1, 2]-p[k, 2]) / (t[-1]-t[k])
            if early < -.55 and late > .28:
                good.append(k)
        if good:
            k = good[-1]
            first = np.searchsorted(t, t[k] - MAX_ARC)
            selected = None
            for offset in np.linspace(-.85, .85, 7) * self.period:
                when = t[k] + offset
                z, y = t[first:]-when, p[first:]
                pack = model(z, y, split=.006)
                if pack is None:
                    continue
                s = solve(z, y, pack, current, 'zero', when, when)
                cost = s.residual + .22 * abs(s.position[2]-current['height'])
                if (s.pre[2] < -.60 and s.post[2] > .36 and s.pre[1] < -.8
                        and s.post[1] < -.8 and .754 < s.position[2] < .795
                        and (selected is None or cost < selected[0])):
                    selected = cost, z, y, pack, when, when
            if selected is not None and selected[0] < .007:
                return selected[1:]
        # Retain only a coherent flight window before the reversal is visibly established.
        k = np.searchsorted(t, now - MAX_ARC)
        for start in np.unique(np.clip([k, max(k, len(t)-12), max(k, len(t)-7), len(t)-4], k, len(t)-4)):
            z, y = t[start:]-now, p[start:]
            if z[-1]-z[0] < .048:
                continue
            pack = model(z, y)
            s = solve(z, y, pack, current, 'zero', now, None)
            if s.residual < .0040 and s.pre[1] < -.85:
                return z, y, pack, now, None
        return None

    def fit(self, mode):
        result = self.evidence()
        if result is None:
            return None
        t, p, packet, stamp, bounce = result
        return solve(t, p, packet, self.settings, mode, stamp, bounce)
