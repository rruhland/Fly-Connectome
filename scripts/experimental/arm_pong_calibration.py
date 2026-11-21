"""Calibrate transferred visual shots against the paddle the arm actually moves."""

from dataclasses import dataclass
import math

from arm_plays_pong import ArmPongController, NATIVE_HZ
from button_pong_skill import CONTACT_X, reflected_y


@dataclass(frozen=True)
class CrossingForecast:
    tick: float
    y: float
    goal: float
    aimed: bool


class ShotCalibrationController(ArmPongController):
    """Reserve room for pixel prediction and arm execution errors near a shot.

    The starting shot direction and offset come from the Pong policy. The
    adapter sees the same pixels and native point score with either policy
    prior. Its evidence persists between points within a match.
    """

    def __init__(self, policy, motor, seed, *, cap=.091, adaptive=True, guard=False):
        super().__init__(policy, motor, seed)
        self.cap = cap
        self.adaptive = adaptive
        self.guard = guard
        self.execution_error = .012
        self.flight_error = .006
        self.score_margin = 0.
        self.crossings = []
        self.score_updates = []
        self.shot_frames = 0
        self.adjusted_frames = 0

    def begin_point(self):
        super().begin_point()
        self._forecast = None
        self._last_paddle = None
        self._last_shot = None
        self._visually_returned = False
        self._committed_until = 0

    @property
    def offset_limit(self):
        if not self.adaptive:
            return self.cap
        error = max(0., self.execution_error - .012) + max(0., self.flight_error - .006)
        return max(.064, min(.105, self.cap - .65 * error - self.score_margin))

    def _measure(self, screen, tick):
        forecast = self._forecast
        if forecast is None or tick < forecast.tick:
            return
        previous_tick, previous_y = self._last_paddle
        fraction = (forecast.tick - previous_tick) / (tick - previous_tick)
        paddle = previous_y + (screen.paddle_y - previous_y) * max(0., min(1., fraction))
        error = abs(paddle - forecast.goal)
        if self.adaptive:
            self.execution_error += .20 * (min(.065, error) - self.execution_error)
        self.crossings.append({'tick': tick, 'projected_y': forecast.y, 'goal_y': forecast.goal,
                               'paddle_y': paddle, 'execution_error': error, 'aimed': forecast.aimed})
        if forecast.aimed:
            self._last_shot = tick
            self._visually_returned = False
        self._forecast = None
        self._committed_until = tick + round(.26 * NATIVE_HZ)

    def aim(self, screen, tick):
        inherited_goal = self.policy.target(screen, tick)
        self._measure(screen, tick)
        self._last_paddle = tick, screen.paddle_y
        velocity = self.policy.ball_velocity
        paddle_stop = screen.paddle_y + .30 * self.policy.paddle_velocity
        goal = inherited_goal
        if (self._last_shot is not None and tick - self._last_shot <= .55 * NATIVE_HZ
                and velocity is not None and velocity[0] > .15
                and screen.ball is not None and .09 < screen.ball[0] < .40):
            self._visually_returned = True
        if screen.ball is not None and velocity is not None and velocity[0] < -.15:
            x, y = screen.ball
            remaining = (CONTACT_X - x) / velocity[0]
            if remaining > 0:
                intercept = reflected_y(y + self.policy.crossings.lead * velocity[1] * remaining)
                offset = inherited_goal - intercept
                aimed = abs(offset) > .035 and .105 < inherited_goal < .895
                if aimed:
                    self.shot_frames += 1
                    magnitude = min(abs(offset), self.offset_limit)
                    if self.guard:
                        direction = math.copysign(1., offset)
                        room = direction * (paddle_stop - intercept) + .56 * max(0., remaining - .075)
                        magnitude = min(magnitude, max(0., room))
                    goal = intercept + math.copysign(magnitude, offset)
                    self.adjusted_frames += int(abs(goal - inherited_goal) >= .002)
                if tick >= self._committed_until and remaining <= .18:
                    if self._forecast is not None and self.adaptive:
                        revision = abs(intercept - self._forecast.y)
                        self.flight_error += .10 * (min(.05, revision) - self.flight_error)
                    self._forecast = CrossingForecast(tick + remaining * NATIVE_HZ, intercept, goal, aimed)
        error = goal - paddle_stop
        return int(error > .012) - int(error < -.012)

    def finish_point(self, score):
        super().finish_point(score)
        if not self.adaptive or not score or self._last_shot is None:
            return
        # An optically unconfirmed return followed by a loss calls for a wider
        # interception margin. A confirmed return that failed to score allows
        # some of the aggression back. A winning point leaves the limit alone.
        if score < 0:
            change = -.003 if self._visually_returned else .006
            self.score_margin = max(-.010, min(.020, self.score_margin + change))
        self.score_updates.append({'score': score, 'return_visible': self._visually_returned,
                                   'offset_limit': self.offset_limit})

    def report(self):
        return {'cap_start': self.cap, 'cap_end': self.offset_limit, 'execution_error': self.execution_error,
                'flight_error': self.flight_error, 'score_margin': self.score_margin,
                'shot_frames': self.shot_frames, 'adjusted_frames': self.adjusted_frames,
                'crossings': self.crossings, 'score_updates': self.score_updates}
