"""A seeded block-and-placement opponent for the standard 3D table arena."""

import math

from tt3_arena import (BALL_RADIUS, GRAVITY, NET_HEIGHT, PADDLE_RESTITUTION,
                       TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, TABLE_RESTITUTION,
                       TT3Arena, _clamp, _dot, _unit)


LOFTS = (-.46, -.38, -.30, -.24, -.18, -.12, -.06, 0., .067, .134, .201, .268, .335, .402, .47, .54, .61)


def _to_bounce(z, vz):
    return (vz + math.sqrt(max(0., vz * vz + 2 * GRAVITY * (z - BALL_RADIUS)))) / GRAVITY


def _aim_normal(position, incoming, racket_velocity, placement):
    """Choose a face angle by simulating reflected flight for plausible lofts."""
    x, y, z = position
    vx, vy, vz = incoming
    target_x, target_y = placement
    relative = tuple(a - b for a, b in zip(incoming, racket_velocity))
    best, cost = (0., -1., 0.), math.inf
    for loft in LOFTS:
        normalization = math.hypot(1., loft)
        impulse = (1. + PADDLE_RESTITUTION) * (relative[1] - relative[2] * loft) / normalization
        flight = _to_bounce(z, vz + impulse * loft / normalization)
        lateral = _clamp(((target_x - x) / max(.10, flight) - vx) * normalization
                         / max(.5, impulse), (-.44, .44))
        normal = _unit((lateral, -1., loft))
        impulse = -(1. + PADDLE_RESTITUTION) * _dot(relative, normal)
        outgoing_y = vy + impulse * normal[1]
        if outgoing_y >= -.3:
            continue
        outgoing_z = vz + impulse * normal[2]
        net_time = -y / outgoing_y
        net_z = z + outgoing_z * net_time - GRAVITY * net_time**2 / 2
        flight = _to_bounce(z, outgoing_z)
        bx, by = x + (vx + impulse * normal[0]) * flight, y + outgoing_y * flight
        error = .6 * abs(bx - target_x) + abs(by - target_y)
        error += 9. * max(0., NET_HEIGHT + BALL_RADIUS + .04 - net_z)
        error += 9. * max(0., abs(bx) - TABLE_HALF_WIDTH + .038,
                           -TABLE_HALF_LENGTH + .055 - by, by + .24)
        # Prefer a ball still in play near the receiver's backcourt after its bounce.
        if by > -1.17:
            after = (-1.17 - by) / outgoing_y
            rebound = TABLE_RESTITUTION * (GRAVITY * flight - outgoing_z)
            ride = BALL_RADIUS + after * (rebound - GRAVITY * after / 2)
            error += 4. * max(0., .15 - ride)
        if error < cost:
            best, cost = normal, error
    return best


class PracticePartnerArena(TT3Arena):
    """The usual game with a mostly blocking opponent who varies return placement."""

    def _opponent_try(self):
        _, y, z = self.ball
        _, vy, vz = self.velocity
        self._rival_attempt = self._rng.random() < .97
        bounce_y = y + vy * _to_bounce(z, vz)
        self._rival_depth = _clamp(bounce_y + self._rng.uniform(.21, .36), (.71, 1.50))
        self._partner_touch_bias = self._rng.uniform(-.038, .038)
        self._partner_target = (self._rng.uniform(-.60, .60), -self._rng.uniform(.69, 1.12))
        # Most shots present a blocking face. The remainder add a short forward stroke.
        self._partner_swing = (self._rng.uniform(-.55, -.26) if self._rng.random() < .50
                               else self._rng.uniform(-.08, .06))

    def _move_rival(self, dt):
        if self._hitter != 1 or not self._rival_attempt or self.velocity[1] <= .35:
            super()._move_rival(dt)
            return
        x, y, z = self.ball
        vx, vy, vz = self.velocity
        engaging = self._bounced and y > self._rival_depth - .20
        swing = self._partner_swing if engaging else 0.
        face = self.rival[1] if engaging else self._rival_depth
        horizon = max(0., (face - y - BALL_RADIUS) / max(.33, vy - swing))
        bounce_time = _to_bounce(z, vz)
        if not self._bounced and bounce_time < horizon:
            after = horizon - bounce_time
            target_vz = TABLE_RESTITUTION * (GRAVITY * bounce_time - vz) - GRAVITY * after
            contact_z = BALL_RADIUS + target_vz * after + GRAVITY * after**2 / 2
        else:
            target_vz = vz - GRAVITY * horizon
            contact_z = z + vz * horizon - GRAVITY * horizon**2 / 2
        contact_x = x + vx * horizon
        target = (_clamp(contact_x + self._partner_touch_bias, (-.69, .69)),
                  self.rival[1] + swing * dt if engaging else self._rival_depth,
                  _clamp(contact_z, (.08, .66)))
        delta = tuple(_clamp(b - a, (-speed * dt, speed * dt)) for a, b, speed in
                      zip(self.rival, target, (1.75, 1.12, 1.2)))
        self.rival = tuple(a + step for a, step in zip(self.rival, delta))
        if engaging:
            self._rival_normal = _aim_normal((contact_x, face - BALL_RADIUS, contact_z),
                                             (vx, vy, target_vz), tuple(d / dt for d in delta),
                                             self._partner_target)
        else:
            self._rival_normal = (0., -1., 0.)
