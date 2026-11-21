"""Identify a hidden change in racket response from stereo ball tracks."""

from dataclasses import dataclass
from statistics import median

import numpy as np

from .tt5_player import DRAG, FRAME_DT, RallyPlayer


RACKET_DAMPING = {
    'stock': 25.,
    'medium': 135.,
    'soft': 160.,
    'very_soft': 175.,
}


def set_racket_response(arena, response):
    """Change the learner's native MuJoCo contact damping without changing its appearance."""
    if response not in RACKET_DAMPING:
        raise ValueError(f'unknown racket response {response!r}')
    # The stock blade uses direct solref (negative stiffness and damping).
    # Its contact priority gives these values precedence over the ball's.
    arena.model.geom('learner_racket').solref[1] = -RACKET_DAMPING[response]


@dataclass(frozen=True)
class OpticalImpact:
    tick: int
    incoming: float
    outgoing: float


def _depth_line(path, start, stop, tick, dt, *, far_limit=-.80, correct_drag=False):
    indices = np.arange(start, stop)
    points = path[indices]
    seen = np.isfinite(points).all(axis=1)
    # Only fit free flight close to the player and above the table.
    seen &= (points[:, 1] < far_limit) & (points[:, 2] > .06)
    if seen.sum() < 3:
        return None
    times = (indices[seen] - tick) * dt
    depths = points[seen, 1]
    values = np.exp(DRAG * depths) if correct_drag else depths
    slope = np.dot(times - times.mean(), values - values.mean()) / np.sum((times - times.mean())**2)
    center = values.mean() - times.mean() * slope
    predicted = center + times * slope
    if correct_drag:
        if min(center, predicted.min()) <= 0.:
            return None
        # Exponentiating depth linearizes longitudinal quadratic drag.
        slope /= DRAG * center
        center = np.log(center) / DRAG
        predicted = np.log(predicted) / DRAG
    error = np.mean(np.abs(depths - predicted))
    return center, slope, error


def _scan_impact(path, dt, span, *, delayed=False):
    outgoing_span = max(span, round(.34 / dt)) if delayed else span
    best = None
    for tick in range(3, len(path) - 3):
        if best is not None and tick > best[1].tick + span:
            break
        before = _depth_line(path, max(0, tick - span), tick, tick, dt)
        after = _depth_line(path, tick + 1, min(len(path), tick + outgoing_span + 1), tick, dt,
                            far_limit=.12 if delayed else -.80, correct_drag=delayed)
        if before is None or after is None:
            continue
        near, incoming, inbound_error = before
        departure, outgoing, outbound_error = after
        gap = abs(near - departure)
        if not (-1.85 < near < -1.05 and -8. < incoming < -1. and .15 < outgoing < 10.):
            continue
        if gap > (.20 if delayed else .16) or max(inbound_error, outbound_error) > .055:
            continue
        quality = gap + 2. * (inbound_error + outbound_error)
        if best is None or quality < best[0]:
            best = quality, OpticalImpact(tick, -float(incoming), float(outgoing))
    return None if best is None else best[1]


def optical_impact(ball, dt=FRAME_DT, *, reacquisition=False):
    """Fit the first near-end optical reversal, even if the blade occludes contact."""
    path = np.asarray([p if p is not None else (np.nan,) * 3 for p in ball], dtype=float)
    span = max(4, round(.15 / dt))
    impact = _scan_impact(path, dt, span)
    if reacquisition:
        delayed = _scan_impact(path, dt, span, delayed=True)
        if delayed is not None and (impact is None or delayed.tick < impact.tick - span):
            return delayed
    return impact


def practiced_gain(pairs):
    """Fit responsiveness from paired optical trials at the same mini-table launch."""
    shifts = [abs(a.outgoing / a.incoming - b.outgoing / b.incoming)
              for a, b in pairs if a is not None and b is not None]
    if not shifts:
        raise ValueError('practice requires at least one paired optical impact')
    change = median(shifts)
    return dict(pairs=len(shifts), median_response_change=change,
                gain=float(np.clip(.20 + 1.8 * change, .20, .88)))


class RacketAdaptation:
    """Correct swing speed from the visible bounce and the most recent point score."""

    def __init__(self, *, gain=.20, enabled=True, default_speed=.049, dt=FRAME_DT, reacquisition=False):
        if not 0. <= gain <= 1. or default_speed <= 0. or dt <= 0.:
            raise ValueError('expected a valid adaptation gain, swing speed and frame interval')
        self.gain = float(gain)
        self.enabled = enabled
        self.default_speed = float(default_speed)
        self.dt = float(dt)
        self.reacquisition = reacquisition
        self.speed = self.default_speed
        self.reference = []

    def finish(self, ball, reward):
        used_speed = self.speed
        impact = optical_impact(ball, self.dt, reacquisition=self.reacquisition)
        response = None
        if impact is not None:
            nominal_blade_speed = used_speed / self.dt
            # Depth-only effective response. The same strike and its visible
            # incoming and outgoing speeds supply the scale across bodies.
            response = (impact.outgoing - nominal_blade_speed) / (impact.incoming + nominal_blade_speed)
            # Retain the reference through shifts that arrive before the
            # second clean observation of the starting response.
            consistent = len(self.reference) == 1 and abs(response - self.reference[0]) < .12
            if not self.reference or consistent:
                self.reference.append(response)
            elif self.enabled:
                deficit = median(self.reference) - response
                increase = deficit * (impact.incoming * self.dt + self.default_speed) / max(.70, 1. + response)
                desired = float(np.clip(self.default_speed + increase, self.default_speed, .102))
                # Downweight failures of an optical model after a won point.
                weight = self.gain * (.90 if reward > 0. else 1.)
                self.speed += weight * (desired - self.speed)
        return dict(used_speed=float(used_speed), next_speed=float(self.speed), visual_impact=None if impact is None
                    else dict(tick=impact.tick, incoming=impact.incoming, outgoing=impact.outgoing),
                    response=response, reference=median(self.reference) if self.reference else None)


class RacketPlayer(RallyPlayer):
    """Use the existing RGB-only body and flight controller with adjustable swing."""

    def __init__(self, arena_player, adaptation):
        self.racket = adaptation
        super().__init__(arena_player.adapter, arena_player.prior, arena_player.tracker_factory,
                         adjuster=arena_player.adjuster, retain_trace=True, stroke_speed=adaptation.speed,
                         transfer_strength=arena_player.transfer_strength,
                         impact_restitution=arena_player.impact_restitution)
        self.role_rows = arena_player.role_rows.copy()
        self.gravity = arena_player.gravity

    def reset(self):
        super().reset()
        self.stroke_speed = self.racket.speed

    def finish(self, reward):
        result = super().finish(reward)
        result['racket_adaptation'] = self.racket.finish([frame['ball'] for frame in self.trace], reward)
        return result
