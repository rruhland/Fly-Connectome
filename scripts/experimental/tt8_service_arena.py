"""Play both service roles with the same regulation MuJoCo table and arms."""

import numpy as np

from .tt4_arena import TABLE_HALF_LENGTH, TABLE_HEIGHT
from .tt6_body_arena import TT6Arena


MIN_TOSS_RISE = .16


class TT8ServiceArena(TT6Arena):
    """On learner service, release an upward toss for the controlled arm to hit."""

    def reset(self, *, seed=None, render=True, server=-1):
        if type(server) is not int or server not in (-1, 1):
            raise ValueError('server must be 1 (learner) or -1 (rival)')
        self.server = server
        self._waiting_for_strike = server == 1
        self.service_contact = False
        self.legal_serve = False
        super().reset(seed=seed, render=False)
        # The server owns any pre-strike fault, just as after a missed return.
        self._last_hitter = server
        if self._waiting_for_strike:
            self._toss_height = self._toss_apex = float(self.ball[2])
            self._event('serve_toss', side=server)
            self.events[-1]['step'] = 0
        return self.render_cameras() if render else None

    def _opening(self):
        if self.server == -1:
            super()._opening()
            return
        # Place the ball behind the learner's end line. Gravity and the regular
        # aerodynamic forces carry it from release through the descending strike.
        self.ball[:] = self._rng.uniform((.015, -1.497, 1.118), (.155, -1.467, 1.142))
        if self.body in ('gantry', 'offset'):
            self.ball[1] -= .08  # Clear the farther-back resting blades during descent.
        self.velocity[:] = self._rng.uniform((-.06, -.025, 2.83), (.06, .025, 2.99))

    def _contacts(self, incoming):
        if self._waiting_for_strike:
            self._toss_apex = max(self._toss_apex, self.ball[2])
        super()._contacts(incoming)

    def _racket_contact(self, side, location, incoming):
        if not self._waiting_for_strike or side != self.server:
            super()._racket_contact(side, location, incoming)
            return

        self.contacts += 1
        self.service_contact = True
        self._event('contact', side=side, location=location, incoming=incoming,
                    racket=tuple(self.paddle.tolist()))
        if incoming[2] >= 0. or self._toss_apex - self._toss_height < MIN_TOSS_RISE:
            self._end(-side, 'serve_illegal_toss_strike')
        elif self.ball[1] >= -TABLE_HALF_LENGTH or self.ball[2] < TABLE_HEIGHT:
            self._end(-side, 'serve_illegal_strike_position')
        else:
            self._waiting_for_strike = False
            self._event('serve_strike', side=side, location=location, incoming=incoming)
            self._choose_rival_response()

    def _table_contact(self, location, vertical, incoming):
        if self._waiting_for_strike:
            self._end(-self.server, 'serve_missed_toss')
            return
        received = self._received_bounce
        super()._table_contact(location, vertical, incoming)
        if self._serving and self._received_bounce and not received:
            self.legal_serve = True
            self._event('legal_serve', side=self.server, location=location)

    def _end(self, winner, reason):
        if self._waiting_for_strike and reason in ('floor', 'out'):
            reason = 'serve_missed_toss'
        super()._end(winner, reason)

    def point_result(self):
        return dict(super().point_result(), server=self.server, service_contact=self.service_contact,
                    legal_serve=self.legal_serve, serve_let=self.reason == 'serve_let')


class ServiceScore:
    """Use two-point service turns up to deuce and one-point turns thereafter."""

    def __init__(self, target=11, first_server=1):
        if type(target) is not int or target < 2:
            raise ValueError('expected a game target of at least two points')
        if type(first_server) is not int or first_server not in (-1, 1):
            raise ValueError('first server must be 1 (learner) or -1 (rival)')
        self.target = target
        self.first_server = first_server
        self._points = [0, 0]

    @property
    def score(self):
        return tuple(self._points)

    @property
    def finished(self):
        own, rival = self._points
        return max(own, rival) >= self.target and abs(own - rival) >= 2

    @property
    def winner(self):
        if self.finished:
            return 1 if self._points[0] > self._points[1] else -1
        return None

    @property
    def server(self):
        points = sum(self._points)
        deuce = 2 * (self.target - 1)
        turn = min(points, deuce) // 2 + max(0, points - deuce)
        return self.first_server * (-1 if turn % 2 else 1)

    def record(self, result):
        if self.finished:
            raise RuntimeError('the game is finished')
        if result['server'] != self.server:
            raise ValueError('the point was played by the wrong server')
        won, lost = result['points_for'], result['points_against']
        if (won, lost) not in ((1, 0), (0, 1), (0, 0)):
            raise ValueError('invalid point score')
        if result['serve_let'] or result['reason'] == 'timeout':
            if (won, lost) != (0, 0):
                raise ValueError('a let or timeout cannot award a point')
        elif (won, lost) == (0, 0):
            raise ValueError('the point has no result')
        self._points[0] += won
        self._points[1] += lost
        return self.score
