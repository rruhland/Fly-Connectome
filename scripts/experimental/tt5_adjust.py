"""Bounded contact calibration from the player's observed ball and blade."""

import numpy as np


class ContactAdjuster:
    """Carry small strike offsets between points using observed plane crossings."""

    def __init__(self, ready, *, enabled=True, gain=.45):
        self.ready = np.asarray(ready, dtype=float).copy()
        self.enabled = enabled
        self.gain = gain
        self.bias = np.zeros((2, 2))
        self.updates = 0
        self.reset()

    def reset(self):
        self.ball = {}
        self.blade = {}
        self.strikes = {}
        self.tick = None

    def observe(self, tick, ball_position, actor_position):
        self.tick = tick
        if ball_position is not None:
            self.ball[tick] = np.asarray(ball_position, dtype=float).copy()
        if actor_position is not None:
            self.blade[tick] = np.asarray(actor_position, dtype=float).copy()

    def _context(self, goal):
        return np.array((1., np.clip((goal[0] - self.ready[0]) / .40, -1., 1.)))

    def correct(self, goal, horizon):
        if horizon is None:
            return goal
        context = self._context(goal)
        if self.tick is not None and -2. < horizon < 8.:
            self.strikes[self.tick] = context
        if not self.enabled:
            return goal
        corrected = np.asarray(goal, dtype=float).copy()
        corrected[[0, 2]] += np.clip(context @ self.bias, -.12, .12)
        return corrected

    def _crossing(self):
        """Find an observed incoming passage through the moving blade's plane."""
        shared = sorted(self.ball.keys() & self.blade.keys())
        best = None
        for a, b in zip(shared, shared[1:]):
            if b - a > 2:
                continue
            previous, current = self.ball[a], self.ball[b]
            blade_a, blade_b = self.blade[a], self.blade[b]
            travel = (current - previous) / (b - a)
            if not -.30 < travel[1] < -.023 or abs(travel[0]) > .10 or abs(travel[2]) > .14:
                continue
            # The ball's center arrives one radius before the blade surface.
            da, db = previous[1] - blade_a[1] - .02, current[1] - blade_b[1] - .02
            if not da >= 0 >= db or da - db < .022:
                continue
            fraction = da / (da - db)
            time = a + fraction * (b - a)
            approaches = [tick for tick in self.strikes if time - 4 <= tick <= time]
            if not approaches:
                continue
            strike = self.strikes[max(approaches)]
            actor = blade_a + fraction * (blade_b - blade_a)
            ball = previous + fraction * (current - previous)
            residual = (ball - actor)[[0, 2]]
            if np.max(np.abs(residual)) > .32 or ball[2] < -.04:
                continue
            # Prefer adjacent samples and intersections near their midpoint.
            uncertainty = (b - a) + .5 * abs(fraction - .5)
            if best is None or uncertainty < best[0]:
                best = uncertainty, time, residual, strike
        return best

    def _departures(self):
        """Summarize visible departures from the player's end toward the net."""
        ticks = list(self.ball)
        departures = []
        for a, b, c in zip(ticks, ticks[1:], ticks[2:]):
            if c - a > 5:
                continue
            near = self.ball[a]
            if not self.ready[1] - .40 < near[1] < self.ready[1] + .38 or near[2] < -.02:
                continue
            speed1 = (self.ball[b][1] - near[1]) / (b - a)
            speed2 = (self.ball[c][1] - self.ball[b][1]) / (c - b)
            if min(speed1, speed2) > .027:
                departures.append(a)
        return departures

    def finish(self, reward):
        crossing = self._crossing()
        departures = self._departures()
        outgoing = departures[0] if departures else None
        time, residual = None, None
        if crossing is not None:
            _, time, residual, context = crossing
            if departures:
                outgoing = min(departures, key=lambda tick: abs(tick - time))
            clear_miss = outgoing is None or abs(outgoing - time) > 4.
            size = np.linalg.norm(residual)
            if self.enabled and clear_miss and reward <= 0. and size > .065:
                change = self.gain * np.clip(residual, -.14, .14)
                self.bias += np.outer(context, change) / (context @ context)
                self.bias[:] = np.clip(self.bias, -.115, .115)
                self.updates += 1
        return dict(crossing_tick=time, crossing_residual=None if residual is None else residual.tolist(),
                    outgoing_tick=outgoing, bias=self.bias.tolist(), updates=self.updates)
